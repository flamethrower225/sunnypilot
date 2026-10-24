"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Harness check: which bus the camera's steering messages arrive on, and whether that means the comma harness needs a pin swap.

openpilot makes the same call when it fingerprints a Hyundai, Kia or Genesis CAN-FD car (opendbc/car/hyundai/interface.py):
LKAS or LKAS_ALT on the camera bus means HDA2. When the harness is wired for a different trim those messages land on
another bus, that test silently fails, the car is set up as non-HDA2 and openpilot cannot steer it.

Pure Python with no openpilot runtime imports, so it can be tested anywhere. harnessd feeds it the 'can' stream.
"""
from enum import StrEnum

LKAS, LKAS_ALT, LFA, LFA_ALT = 0x50, 0x110, 0x12A, 0xCB
HDA2_MSGS = {LKAS: "LKAS", LKAS_ALT: "LKAS_ALT"}
NON_HDA2_MSGS = {LFA: "LFA", LFA_ALT: "LFA_ALT"}
STEERING_MSGS = {**HDA2_MSGS, **NON_HDA2_MSGS}

MIN_FRAMES = 20           # a message counts on a bus only after this many frames, so a stray frame can't decide anything
SENT_SRC = 64             # src at or above this is a panda TX echo (+128) or a blocked frame (+192), not traffic from the car
CLASSIC_CAN_MAX_LEN = 8   # longer frames only exist on CAN-FD
UNRECOGNIZED = ("", "MOCK")

PROVISIONAL_PERIOD_S = 2.0
FINAL_S = 15.0
GIVE_UP_S = 60.0

SWAP_ACTION = ("Flip the CAN pairs on the comma harness using the wiring diagram for your trim, then turn the car on again: "
               "the steering messages must arrive on bus {cam}.")


class Verdict(StrEnum):
  OK_HDA2 = "ok_hda2"
  SWAP_NEEDED = "swap_needed"
  NON_HDA2 = "non_hda2"
  SWAP_LIKELY = "swap_likely"
  INCONCLUSIVE = "inconclusive"


ALERT_VERDICTS = (Verdict.SWAP_NEEDED, Verdict.SWAP_LIKELY)


class HarnessTally:
  """Frames per (bus, address) from the car, plus the longest frame and the CAN time span. One dict update per frame."""

  def __init__(self):
    self.counts: dict[tuple[int, int], int] = {}
    self.max_len = 0
    self.first_ns: int | None = None
    self.last_ns = 0

  def ingest(self, t_ns: int, src: int, address: int, length: int) -> None:
    if src >= SENT_SRC:
      return
    if self.first_ns is None:
      self.first_ns = t_ns
    self.last_ns = t_ns
    key = (src, address)
    self.counts[key] = self.counts.get(key, 0) + 1
    if length > self.max_len:
      self.max_len = length

  @property
  def seconds(self) -> float:
    return 0.0 if self.first_ns is None else (self.last_ns - self.first_ns) / 1e9


def camera_bus(buses) -> int:
  """opendbc CanBusBase: every panda adds four buses, and the camera bus is the third one on the last panda."""
  return 2 + 4 * (max(buses, default=0) // 4)


def decision_from(fingerprint: str, brand: str, lka_steer: bool) -> dict:
  """What openpilot made of the car when it fingerprinted it. hda2 is only meaningful for Hyundai, Kia and Genesis."""
  recognized = fingerprint not in UNRECOGNIZED
  return {"recognized": recognized, "car": fingerprint if recognized else "",
          "hda2": bool(lka_steer) if recognized and brand == "hyundai" else None}


def bus_list(buses: list[int]) -> str:
  if len(buses) == 1:
    return f"bus {buses[0]}"
  return "buses " + ", ".join(str(b) for b in buses[:-1]) + f" and {buses[-1]}"


def name_list(names: list[str]) -> str:
  return " and ".join(names)


def _verdict(tally: HarnessTally, cam: int, seen: dict[int, list[int]]) -> tuple[Verdict, str, str, str]:
  def on_cam(msgs):
    return [name for addr, name in msgs.items() if cam in seen[addr]]

  def elsewhere(msgs):
    return sorted({bus for addr in msgs for bus in seen[addr] if bus != cam}), [name for addr, name in msgs.items() if seen[addr]]

  if not tally.counts:
    return (Verdict.INCONCLUSIVE, "No CAN traffic yet", "Nothing has arrived from the car yet.",
            "Turn the car on with the comma plugged in; the check starts as soon as CAN traffic arrives.")
  if tally.max_len <= CLASSIC_CAN_MAX_LEN:
    return (Verdict.INCONCLUSIVE, "Not a CAN-FD car",
            "This car's CAN is not CAN-FD; the check only applies to Hyundai, Kia and Genesis CAN-FD cars.",
            "Nothing to do here: this check does not apply to this car.")

  hda2_cam = on_cam(HDA2_MSGS)
  if hda2_cam:
    return (Verdict.OK_HDA2, "Harness OK (HDA2)",
            f"{name_list(hda2_cam)} arrive on bus {cam}, the camera bus, so openpilot can take over steering.",
            "No pin swap needed.")
  hda2_buses, hda2_names = elsewhere(HDA2_MSGS)
  if hda2_buses:
    return (Verdict.SWAP_NEEDED, f"Pin swap needed: steering on {bus_list(hda2_buses)}",
            f"{name_list(hda2_names)} (HDA2 steering) arrive on {bus_list(hda2_buses)}, not on bus {cam}. "
            "openpilot will set the car up as non-HDA2 and cannot steer it.",
            SWAP_ACTION.format(cam=cam))
  lfa_cam = on_cam(NON_HDA2_MSGS)
  if lfa_cam:
    return (Verdict.NON_HDA2, "Non-HDA2 wiring",
            f"{name_list(lfa_cam)} arrive on bus {cam} and no HDA2 steering messages were seen, so openpilot sets the car up "
            "as non-HDA2 (LFA steering).",
            "Correct for a non-HDA2 car. If this car has Highway Driving Assist II, the harness is not passing the HDA2 "
            "messages: check the harness type and pinout.")
  lfa_buses, lfa_names = elsewhere(NON_HDA2_MSGS)
  if lfa_buses:
    return (Verdict.SWAP_LIKELY, f"Pin swap likely: no steering on bus {cam}",
            f"No steering messages arrive on bus {cam}; {name_list(lfa_names)} arrive on {bus_list(lfa_buses)} instead.",
            SWAP_ACTION.format(cam=cam))
  return (Verdict.INCONCLUSIVE, "No steering messages yet",
          f"None of LKAS, LKAS_ALT, LFA or LFA_ALT arrived on any bus in {tally.seconds:.0f} s.",
          "Make sure the car is fully on (ready to drive, not just accessory) and give the camera a few seconds to start; "
          "the check keeps listening for a minute.")


def _disagrees(verdict: Verdict, decision: dict) -> bool:
  expected = {Verdict.OK_HDA2: True, Verdict.NON_HDA2: False}.get(verdict)
  return expected is not None and decision["hda2"] is not None and decision["hda2"] != expected


def classify(tally: HarnessTally, decision: dict | None, now: float) -> dict:
  """The HarnessCheck param: the verdict in words, the camera bus, and what arrived on each bus. final is set by harnessd."""
  decision = decision or decision_from("", "", False)
  buses = sorted({bus for bus, _ in tally.counts})
  cam = camera_bus(buses)
  seen = {addr: sorted(bus for (bus, a), n in tally.counts.items() if a == addr and n >= MIN_FRAMES) for addr in STEERING_MSGS}
  verdict, title, detail, action = _verdict(tally, cam, seen)

  seconds = tally.seconds
  breakdown = []
  for bus in buses:
    on_bus = {a: n for (b, a), n in tally.counts.items() if b == bus}
    steering = {name: {"frames": on_bus[addr], "hz": round(on_bus[addr] / seconds, 1) if seconds > 0 else 0.0}
                for addr, name in STEERING_MSGS.items() if on_bus.get(addr, 0) >= MIN_FRAMES}
    breakdown.append({"bus": bus, "frames": sum(on_bus.values()), "ids": len(on_bus), "steering": steering})

  return {"verdict": str(verdict), "title": title, "detail": detail, "action": action, "camera_bus": cam,
          "can_fd": tally.max_len > CLASSIC_CAN_MAX_LEN, "seconds": round(seconds, 1), "t": now, "buses": breakdown,
          "decision": decision, "disagrees": _disagrees(verdict, decision), "final": False}


def is_final(seconds: float, verdict: str) -> bool:
  return seconds >= GIVE_UP_S or (seconds >= FINAL_S and verdict != Verdict.INCONCLUSIVE)


def alert_text(result: dict | None, drive_start: float) -> str | None:
  """The onroad alert's title, or None: only a pin-swap result written during this drive raises it."""
  if not result or result.get("verdict") not in ALERT_VERDICTS or "title" not in result:
    return None
  if result.get("t", 0.0) < drive_start:
    return None
  return result["title"]
