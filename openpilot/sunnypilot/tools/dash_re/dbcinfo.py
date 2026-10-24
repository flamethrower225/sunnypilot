"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

What the Hyundai CAN DBC says about the instrument cluster, and how much of it openpilot can reach.

The DBC lists the receiving ECUs of every signal; "CLU" is the instrument cluster. Combined with the
panda harness topology, every cluster input falls into one of four tiers:

  tier 1  openpilot already transmits the message (panda safety TX list), so its fields can be changed in
          carcontroller with no firmware change
  tier 2  the camera (MFC, bus 2) sends it and the panda forwards it to the car untouched. openpilot could
          substitute it, but the message must first be added to the panda safety TX list and blocked from
          forwarding
  tier 3  an ECU on the car side (bus 0) sends it. openpilot can only read it: a second sender on the same
          bus would collide with the real one
  tier 4  it travels on M-CAN, the multimedia bus between the head unit and the cluster (navigation
          turn-by-turn, media and phone text, CarPlay/Android Auto channels). The camera harness is not on
          that bus at all; reading it needs a separate tap wired to a spare panda CAN port, and writing it
          also needs the real sender blocked (an inline gateway, like the harness relay does for the camera)
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

_DBC_DIR = Path(__file__).resolve().parents[4] / "opendbc_repo/opendbc/dbc"
HYUNDAI_DBC = _DBC_DIR / "generator/hyundai/hyundai_can.dbc"
HYUNDAI_MCAN_DBC = _DBC_DIR / "hyundai_2015_mcan.dbc"

CLUSTER = "CLU"

# opendbc_repo/opendbc/safety/modes/hyundai.h: HYUNDAI_TX_MSGS and HYUNDAI_LONG_TX_MSGS (radar-SCC, no ESCC).
# CLU11 (0x4F1) is in the TX list too, but only to spoof cruise buttons: it is the cluster's own output, not a
# dash input, so it is left out here.
OP_TX_STOCK_LONG = {0x340, 0x485}
OP_TX_OP_LONG = OP_TX_STOCK_LONG | {0x420, 0x421, 0x50A, 0x389, 0x4A2, 0x38D, 0x483}

# ECUs behind the camera side of the harness on CAN (non-CAN FD) Hyundai: the MFC camera and what it
# originates. Everything else in the DBC lives on the car side.
CAMERA_SENDERS = {"LDWS_LKAS"}
# messages with an unnamed sender in the DBC that are known to come from the camera
CAMERA_MESSAGES = {0x485, 0x484}  # LFAHDA_MFC, HDA11_MFC

TIER_DESCRIPTIONS = {
  1: "openpilot transmits it: change fields in carcontroller",
  2: "camera-originated, forwarded: needs a panda safety TX entry to substitute",
  3: "car-side sender on C-CAN: read-only",
  4: "M-CAN (head unit <-> cluster): needs a hardware tap to read, an inline gateway to write",
}


@dataclass
class Signal:
  name: str
  start: int
  size: int
  little_endian: bool
  signed: bool
  factor: float
  offset: float
  unit: str
  receivers: list[str]

  @property
  def bits(self) -> list[int]:
    """Bit positions as byte * 8 + bit (LSB first within a byte)."""
    if self.little_endian:
      return list(range(self.start, self.start + self.size))
    out, pos = [], self.start
    for _ in range(self.size):
      out.append(pos)
      pos = pos + 15 if pos % 8 == 0 else pos - 1
    return out


@dataclass
class Message:
  addr: int
  name: str
  length: int
  sender: str
  signals: list[Signal] = field(default_factory=list)

  @property
  def cluster_signals(self) -> list[Signal]:
    return [s for s in self.signals if CLUSTER in s.receivers]

  @property
  def covered_bits(self) -> set[int]:
    return {b for s in self.signals for b in s.bits}

  @property
  def camera_originated(self) -> bool:
    return self.sender in CAMERA_SENDERS or self.addr in CAMERA_MESSAGES


_BO = re.compile(r"^BO_ (\d+) (\w+)\s*:\s*(\d+) (\w+)")
_SG = re.compile(r'^\s*SG_ (\w+)\s*(?:\w+\s*)?:\s*(\d+)\|(\d+)@([01])([+-])\s*\(([^,]+),([^)]+)\)\s*\[[^]]*\]\s*"([^"]*)"\s*(.*)$')


def parse_dbc(path: Path = HYUNDAI_DBC) -> dict[int, Message]:
  msgs: dict[int, Message] = {}
  cur: Message | None = None
  for line in path.read_text(encoding="latin-1").splitlines():
    if m := _BO.match(line):
      cur = Message(int(m[1]), m[2], int(m[3]), m[4])
      msgs[cur.addr] = cur
    elif (m := _SG.match(line)) and cur is not None:
      receivers = [r for r in re.split(r"[,\s]+", m[9].strip()) if r]
      cur.signals.append(Signal(m[1], int(m[2]), int(m[3]), m[4] == "1", m[5] == "-",
                                float(m[6]), float(m[7]), m[8], sorted(set(receivers))))
    elif not line.strip():
      cur = None
  return msgs


def tier(msg: Message, op_long: bool = True, mcan: bool = False) -> int:
  if mcan:
    return 4
  if msg.addr in (OP_TX_OP_LONG if op_long else OP_TX_STOCK_LONG):
    return 1
  if msg.camera_originated:
    return 2
  return 3


def cluster_candidates(msgs: dict[int, Message], op_long: bool = True, mcan: bool = False) -> list[tuple[int, Message]]:
  """Messages the cluster reads, best-reachable first. LFAHDA_MFC has no receivers in the DBC but drives
  the HDA/LFA icons, so it is always included."""
  out = [(tier(m, op_long, mcan), m) for m in msgs.values()
         if m.cluster_signals or (not mcan and m.addr in CAMERA_MESSAGES)]
  return sorted(out, key=lambda x: (x[0], x[1].addr))


def all_cluster_candidates(op_long: bool = True) -> list[tuple[int, Message]]:
  return sorted(cluster_candidates(parse_dbc(HYUNDAI_DBC), op_long) + cluster_candidates(parse_dbc(HYUNDAI_MCAN_DBC), op_long, mcan=True),
                key=lambda x: (x[0], x[1].addr))
