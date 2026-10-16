"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Analysis of CAN captures: where each message comes from, what its bits do, and which bits change when the
driver does something the dash reacts to. Built on numpy so it runs on the device itself.

Bit numbering everywhere is byte * 8 + bit with bit 0 the least significant, the same numbering a DBC uses
for little-endian (@1) start bits. A payload is held as a little-endian uint64, so bit k of that integer is
bit k in this numbering.
"""
import gzip
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from openpilot.sunnypilot.tools.dash_re.dbcinfo import Message

# panda tags frames it transmitted with src = bus + 128; those are openpilot's own output, not the car's
PANDA_TX_OFFSET = 128

# A capture is a gzip text file, one record per line:
#   C <t_ns> <bus> <addr hex> <data hex>    a CAN frame
#   M <t_ns> <label>                        a label (typed, or placed by a guided session)
CAPTURE_SUFFIX = ".dashcap.gz"


@dataclass(frozen=True)
class Frame:
  t: int  # ns
  bus: int
  addr: int
  dat: bytes


@dataclass(frozen=True)
class Marker:
  t: int  # ns
  label: str


def write_frame(f, frame: Frame) -> None:
  f.write(f"C {frame.t} {frame.bus} {frame.addr:x} {frame.dat.hex()}\n")


def write_marker(f, marker: Marker) -> None:
  f.write(f"M {marker.t} {marker.label}\n")


def read_capture(path: str | Path) -> tuple[list[Frame], list[Marker]]:
  frames, markers = [], []
  with gzip.open(path, "rt") as f:
    for line in f:
      parts = line.rstrip("\n").split(" ")
      if parts[0] == "C" and len(parts) in (4, 5):
        frames.append(Frame(int(parts[1]), int(parts[2]), int(parts[3], 16), bytes.fromhex(parts[4]) if len(parts) == 5 else b""))
      elif parts[0] == "M" and len(parts) >= 3:
        markers.append(Marker(int(parts[1]), " ".join(parts[2:])))
  return frames, markers


def read_rlog(path: str) -> tuple[list[Frame], list[Marker]]:
  """openpilot drive logs record every CAN frame; bookmarks (the flag button on the device) become markers."""
  from openpilot.tools.lib.logreader import LogReader

  frames, markers = [], []
  for msg in LogReader(path):
    which = msg.which()
    if which == "can":
      for c in msg.can:
        frames.append(Frame(msg.logMonoTime, c.src, c.address, bytes(c.dat)))
    elif which in ("userBookmark", "bookmarkButton"):
      markers.append(Marker(msg.logMonoTime, which))
  return frames, markers


def load(path: str) -> tuple[list[Frame], list[Marker]]:
  return read_capture(path) if path.endswith(".gz") else read_rlog(path)


def bit(dat: bytes, pos: int) -> int:
  return (dat[pos // 8] >> (pos % 8)) & 1 if pos // 8 < len(dat) else 0


# *** per-message time series ***

@dataclass
class Series:
  t: np.ndarray  # int64 ns
  v: np.ndarray  # uint64 payloads
  lengths: set[int]

  def bits(self, a: int = 0, b: int | None = None) -> np.ndarray:
    """(n, 64) array of 0/1."""
    return np.unpackbits(self.v[a:b].view(np.uint8).reshape(-1, 8), axis=1, bitorder="little")

  def raw(self, n: int) -> list[bytes]:
    return [int(x).to_bytes(8, "little") for x in self.v[:n]]


def build_series(frames: Iterable[Frame]) -> dict[tuple[int, int], Series]:
  ts: dict[tuple[int, int], list[int]] = defaultdict(list)
  vs: dict[tuple[int, int], list[int]] = defaultdict(list)
  lens: dict[tuple[int, int], set[int]] = defaultdict(set)
  for fr in frames:
    if fr.bus >= PANDA_TX_OFFSET:
      continue
    key = (fr.bus, fr.addr)
    ts[key].append(fr.t)
    vs[key].append(int.from_bytes(fr.dat[:8].ljust(8, b"\0"), "little"))
    lens[key].add(len(fr.dat))
  return {k: Series(np.array(ts[k], dtype=np.int64), np.array(vs[k], dtype=np.uint64), lens[k]) for k in ts}


# *** origin ***

def origins(series: dict[tuple[int, int], Series]) -> dict[int, set[int]]:
  """Buses each address was received on. The panda only receives what the car side sends on bus 0 and what
  the camera sends on bus 2; what it forwards or openpilot sends is not received back. So an address seen
  only on bus 2 originates at the camera."""
  out: dict[int, set[int]] = defaultdict(set)
  for bus, addr in series:
    out[addr].add(bus)
  return dict(out)


def origin_label(buses: set[int]) -> str:
  return {frozenset({2}): "camera", frozenset({0}): "car", frozenset({1}): "bus1"}.get(
    frozenset(buses), "+".join(f"bus{b}" for b in sorted(buses)))


# *** per-message inventory ***

@dataclass
class MsgStats:
  bus: int
  addr: int
  count: int
  hz: float
  lengths: set[int]
  toggles: np.ndarray  # per bit, how often it changed
  counters: list[tuple[int, int]] = field(default_factory=list)  # (start bit, size)
  checksum_bytes: list[int] = field(default_factory=list)

  @property
  def changing_bits(self) -> list[int]:
    return [int(b) for b in np.flatnonzero(self.toggles)]

  @property
  def noise_bits(self) -> set[int]:
    """Counter and checksum bits, which change for reasons unrelated to any action."""
    return {s + i for s, sz in self.counters for i in range(sz)} | {b * 8 + i for b in self.checksum_bytes for i in range(8)}


def _find_counters(history: list[bytes]) -> list[tuple[int, int]]:
  """Fields that step by +1 (wrapping) in nearly every frame. Checks 2-8 bit fields at every position."""
  if len(history) < 8:
    return []
  found: list[tuple[int, int]] = []
  for size in (8, 4, 3, 2):
    mod = 1 << size
    for start in range(64 - size + 1):
      if any(start < s + sz and s < start + size for s, sz in found):
        continue
      vals = [sum(bit(d, start + i) << i for i in range(size)) for d in history]
      steps = sum((b - a) % mod == 1 for a, b in zip(vals, vals[1:], strict=False))
      if steps >= 0.9 * (len(vals) - 1) and len(set(vals)) == min(mod, len(vals)):
        found.append((start, size))
  return sorted(found)


def _find_checksum_bytes(history: list[bytes], counters: list[tuple[int, int]]) -> list[int]:
  """Whole bytes that change in almost every frame and take many values but are not counters."""
  if len(history) < 8:
    return []
  counter_bits = {s + i for s, sz in counters for i in range(sz)}
  out = []
  for byte in range(8):
    if set(range(byte * 8, byte * 8 + 8)) <= counter_bits:
      continue
    vals = [d[byte] for d in history]
    changes = sum(a != b for a, b in zip(vals, vals[1:], strict=False))
    if changes >= 0.8 * (len(vals) - 1) and len(set(vals)) >= min(32, len(vals) // 2):
      out.append(byte)
  return out


def inventory(series: dict[tuple[int, int], Series], history_len: int = 200) -> dict[tuple[int, int], MsgStats]:
  stats = {}
  for (bus, addr), s in series.items():
    n = len(s.t)
    dt = (s.t[-1] - s.t[0]) / 1e9 if n > 1 else 0.0
    bits = s.bits()
    toggles = (bits[1:] != bits[:-1]).sum(axis=0) if n > 1 else np.zeros(64, dtype=np.int64)
    history = s.raw(history_len)
    counters = _find_counters(history)
    stats[(bus, addr)] = MsgStats(bus, addr, n, (n - 1) / dt if dt > 0 else 0.0, s.lengths, toggles, counters,
                                  _find_checksum_bytes(history, counters))
  return stats


def undocumented_changing_bits(st: MsgStats, msg: Message | None) -> list[int]:
  """Bits that move but no DBC signal covers. On a dash message these are the first place to look."""
  covered = msg.covered_bits if msg else set()
  return [b for b in st.changing_bits if b not in covered and b not in st.noise_bits]


# *** action vs. baseline ***

@dataclass
class BitChange:
  bus: int
  addr: int
  bit: int
  before: str  # "0", "1" or "toggling"
  after: str
  score: float


def _describe(frac: float) -> str:
  return "1" if frac >= 0.95 else "0" if frac <= 0.05 else "toggling"


def diff_windows(series: dict[tuple[int, int], Series], before: tuple[int, int], after: tuple[int, int],
                 ignore_bits: dict[tuple[int, int], set[int]] | None = None) -> list[BitChange]:
  """Bits whose behavior differs between two time windows, strongest first. A bit that sits at 0 before and
  1 after scores 1.0; a bit that only starts or stops toggling scores lower."""
  ignore_bits = ignore_bits or {}
  out = []
  for key, s in series.items():
    b0, b1 = np.searchsorted(s.t, before)
    a0, a1 = np.searchsorted(s.t, after)
    if b1 <= b0 or a1 <= a0:
      continue
    fb, fa = s.bits(b0, b1).mean(axis=0), s.bits(a0, a1).mean(axis=0)
    for b in np.flatnonzero(np.abs(fa - fb) > 0.05):
      if int(b) in ignore_bits.get(key, set()):
        continue
      db, da = _describe(fb[b]), _describe(fa[b])
      if db != da:
        score = float(abs(fa[b] - fb[b])) if "toggling" in (db, da) else 1.0
        out.append(BitChange(key[0], key[1], int(b), db, da, round(score, 3)))
  return sorted(out, key=lambda c: (-c.score, c.bus, c.addr, c.bit))


def marker_diffs(series: dict[tuple[int, int], Series], markers: list[Marker], before_s: float = 2.0, after_s: float = 3.0,
                 settle_s: float = 0.0, stats: dict[tuple[int, int], MsgStats] | None = None) -> list[tuple[Marker, list[BitChange]]]:
  """For each label, compare the window before it with the window right after it."""
  stats = stats if stats is not None else inventory(series)
  ignore = {k: st.noise_bits for k, st in stats.items()}
  b, a, settle = int(before_s * 1e9), int(after_s * 1e9), int(settle_s * 1e9)
  return [(m, diff_windows(series, (m.t - b, m.t), (m.t + settle, m.t + settle + a), ignore)) for m in markers]


# *** ISO-TP (multi-frame payloads such as head unit -> cluster text) ***

def isotp_payloads(frames: Iterable[Frame], addr: int, bus: int | None = None) -> list[tuple[int, bytes]]:
  """Reassemble ISO-TP payloads (single, first and consecutive frames) sent on one address."""
  out: list[tuple[int, bytes]] = []
  buf, expect, t0, seq = bytearray(), 0, 0, 0
  for fr in frames:
    if fr.addr != addr or (bus is not None and fr.bus != bus) or not fr.dat:
      continue
    pci = fr.dat[0] >> 4
    if pci == 0:  # single frame
      n = fr.dat[0] & 0xF
      out.append((fr.t, bytes(fr.dat[1:1 + n])))
      expect = 0
    elif pci == 1 and len(fr.dat) >= 2:  # first frame
      expect = ((fr.dat[0] & 0xF) << 8) | fr.dat[1]
      buf, t0, seq = bytearray(fr.dat[2:]), fr.t, 1
    elif pci == 2 and expect:  # consecutive frame
      if (fr.dat[0] & 0xF) != seq % 16:
        expect = 0  # lost a frame; drop this payload
        continue
      buf += fr.dat[1:]
      seq += 1
      if len(buf) >= expect:
        out.append((t0, bytes(buf[:expect])))
        expect = 0
  return out


def printable(payload: bytes) -> str:
  """Best-effort text view; head units commonly send UTF-16LE or ASCII."""
  for enc in ("utf-16-le", "ascii"):
    try:
      text = payload.decode(enc)
    except UnicodeDecodeError:
      continue
    if text and sum(ch.isprintable() for ch in text) >= 0.8 * len(text):
      return text
  return payload.hex(" ")
