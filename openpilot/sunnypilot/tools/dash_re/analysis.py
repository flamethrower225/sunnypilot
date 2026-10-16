"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Offline analysis of CAN captures: where each message comes from, what its bits do, and which bits change
when the driver does something the dash reacts to.

Bit numbering everywhere is byte * 8 + bit with bit 0 the least significant, the same numbering a DBC uses
for little-endian (@1) start bits.
"""
import gzip
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from openpilot.sunnypilot.tools.dash_re.dbcinfo import Message

# panda tags frames it transmitted with src = bus + 128; those are openpilot's own output, not the car's
PANDA_TX_OFFSET = 128

# A capture is a gzip text file, one record per line:
#   C <t_ns> <bus> <addr hex> <data hex>    a CAN frame
#   M <t_ns> <label>                        a label typed by the driver
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
      parts = line.rstrip("\n").split(" ", 4)
      if parts[0] == "C" and len(parts) == 5:
        frames.append(Frame(int(parts[1]), int(parts[2]), int(parts[3], 16), bytes.fromhex(parts[4])))
      elif parts[0] == "C" and len(parts) == 4:  # zero-length frame
        frames.append(Frame(int(parts[1]), int(parts[2]), int(parts[3], 16), b""))
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


# *** origin ***

def origins(frames: Iterable[Frame]) -> dict[int, set[int]]:
  """Buses each address was received on. With the harness in place the panda only receives what the car
  side sends on bus 0 and what the camera sends on bus 2; anything it forwards or openpilot sends is not
  received back, so an address seen only on bus 2 originates at the camera."""
  out: dict[int, set[int]] = defaultdict(set)
  for fr in frames:
    if fr.bus < PANDA_TX_OFFSET:
      out[fr.addr].add(fr.bus)
  return dict(out)


def origin_label(buses: set[int]) -> str:
  if buses == {2}:
    return "camera"
  if buses == {0}:
    return "car"
  if buses == {1}:
    return "bus1"
  return "+".join(f"bus{b}" for b in sorted(buses))


# *** per-message inventory ***

@dataclass
class MsgStats:
  bus: int
  addr: int
  count: int = 0
  t_first: int = 0
  t_last: int = 0
  lengths: set[int] = field(default_factory=set)
  toggles: list[int] = field(default_factory=lambda: [0] * 64)  # per bit, how often it changed
  values: list[set[int]] = field(default_factory=lambda: [set() for _ in range(64)])
  last: bytes = b""
  counters: list[tuple[int, int]] = field(default_factory=list)  # (start bit, size)
  checksum_bytes: list[int] = field(default_factory=list)

  @property
  def hz(self) -> float:
    dt = (self.t_last - self.t_first) / 1e9
    return (self.count - 1) / dt if dt > 0 and self.count > 1 else 0.0

  @property
  def changing_bits(self) -> list[int]:
    return [b for b in range(64) if self.toggles[b]]


def _find_counters(history: list[bytes]) -> list[tuple[int, int]]:
  """Fields that step by +1 (wrapping) in nearly every frame. Checks 2-8 bit fields at every position."""
  if len(history) < 8:
    return []
  found: list[tuple[int, int]] = []
  nbits = 8 * max(len(d) for d in history)
  for size in (8, 4, 3, 2):
    mod = 1 << size
    for start in range(nbits - size + 1):
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
  for byte in range(max(len(d) for d in history)):
    if counter_bits & set(range(byte * 8, byte * 8 + 8)) == set(range(byte * 8, byte * 8 + 8)):
      continue
    vals = [d[byte] for d in history if byte < len(d)]
    changes = sum(a != b for a, b in zip(vals, vals[1:], strict=False))
    if changes >= 0.8 * (len(vals) - 1) and len(set(vals)) >= min(32, len(vals) // 2):
      out.append(byte)
  return out


def inventory(frames: Iterable[Frame], history_len: int = 200) -> dict[tuple[int, int], MsgStats]:
  stats: dict[tuple[int, int], MsgStats] = {}
  history: dict[tuple[int, int], list[bytes]] = defaultdict(list)
  for fr in frames:
    if fr.bus >= PANDA_TX_OFFSET:
      continue
    key = (fr.bus, fr.addr)
    st = stats.get(key)
    if st is None:
      st = stats[key] = MsgStats(fr.bus, fr.addr, t_first=fr.t)
    st.count += 1
    st.t_last = fr.t
    st.lengths.add(len(fr.dat))
    for b in range(min(64, len(fr.dat) * 8)):
      v = bit(fr.dat, b)
      st.values[b].add(v)
      if st.last and bit(st.last, b) != v:
        st.toggles[b] += 1
    st.last = fr.dat
    if len(history[key]) < history_len:
      history[key].append(fr.dat)
  for key, st in stats.items():
    st.counters = _find_counters(history[key])
    st.checksum_bytes = _find_checksum_bytes(history[key], st.counters)
  return stats


def undocumented_changing_bits(st: MsgStats, msg: Message | None) -> list[int]:
  """Bits that move but no DBC signal covers. On a dash message these are the first place to look."""
  covered = msg.covered_bits if msg else set()
  noise = {s + i for s, sz in st.counters for i in range(sz)} | {b * 8 + i for b in st.checksum_bytes for i in range(8)}
  return [b for b in st.changing_bits if b not in covered and b not in noise]


# *** action vs. baseline ***

@dataclass
class BitChange:
  bus: int
  addr: int
  bit: int
  before: str  # "0", "1" or "toggling"
  after: str
  score: float


def _window(frames: list[Frame], t0: int, t1: int) -> Iterator[Frame]:
  # frames are time ordered; a linear scan keeps this dependency free and is fast enough for drive-length captures
  for fr in frames:
    if fr.t >= t1:
      break
    if fr.t >= t0:
      yield fr


def _bit_profile(frames: Iterable[Frame]) -> dict[tuple[int, int], tuple[list[int], int]]:
  """For every message: per-bit count of ones, and the number of frames."""
  out: dict[tuple[int, int], tuple[list[int], int]] = {}
  for fr in frames:
    if fr.bus >= PANDA_TX_OFFSET:
      continue
    ones, n = out.get((fr.bus, fr.addr), ([0] * 64, 0))
    for b in range(min(64, len(fr.dat) * 8)):
      ones[b] += bit(fr.dat, b)
    out[(fr.bus, fr.addr)] = (ones, n + 1)
  return out


def _describe(frac: float) -> str:
  return "1" if frac >= 0.95 else "0" if frac <= 0.05 else "toggling"


def diff_windows(frames: list[Frame], before: tuple[int, int], after: tuple[int, int],
                 ignore_bits: dict[tuple[int, int], set[int]] | None = None) -> list[BitChange]:
  """Bits whose behavior differs between two time windows, strongest first. A bit that sits at 0 before and
  1 after scores 1.0; a bit that only starts or stops toggling scores lower."""
  ignore_bits = ignore_bits or {}
  prof_b = _bit_profile(_window(frames, *before))
  prof_a = _bit_profile(_window(frames, *after))
  out = []
  for key in prof_b.keys() & prof_a.keys():
    (ones_b, n_b), (ones_a, n_a) = prof_b[key], prof_a[key]
    for b in range(64):
      if b in ignore_bits.get(key, set()):
        continue
      fb, fa = ones_b[b] / n_b, ones_a[b] / n_a
      db, da = _describe(fb), _describe(fa)
      if db != da:
        score = abs(fa - fb) if "toggling" in (db, da) else 1.0
        out.append(BitChange(key[0], key[1], b, db, da, round(score, 3)))
  return sorted(out, key=lambda c: (-c.score, c.bus, c.addr, c.bit))


def noise_bits(stats: dict[tuple[int, int], MsgStats]) -> dict[tuple[int, int], set[int]]:
  """Counter and checksum bits, which change for reasons unrelated to any action."""
  return {key: {s + i for s, sz in st.counters for i in range(sz)} | {b * 8 + i for b in st.checksum_bytes for i in range(8)}
          for key, st in stats.items()}


def marker_diffs(frames: list[Frame], markers: list[Marker], window_s: float = 2.0, settle_s: float = 0.3,
                 stats: dict[tuple[int, int], MsgStats] | None = None) -> list[tuple[Marker, list[BitChange]]]:
  """For each label, compare the window before it with the window right after it."""
  w, settle = int(window_s * 1e9), int(settle_s * 1e9)
  ignore = noise_bits(stats if stats is not None else inventory(frames))
  return [(m, diff_windows(frames, (m.t - w, m.t), (m.t + settle, m.t + settle + w), ignore)) for m in markers]


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
