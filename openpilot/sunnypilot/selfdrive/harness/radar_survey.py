"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Radar survey: which Hyundai/Kia/Genesis radar track ranges show up on which bus, at what frame size and rate, with one raw frame
per range. It answers whether a car's radar tracks could be decoded, without a route or SSH.

The ranges and sizes come from the hyundai-radar-tracks research (tools/radar/PLATFORM_MATRIX.md on that upstream branch).
Address and size alone don't prove a layout: the 2026 Sportage sends 0x3A5-0x3C4 with a payload the common decoder can't read.
"""
from __future__ import annotations

from statistics import median
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
  from openpilot.sunnypilot.selfdrive.harness.harness_check import HarnessTally


class RadarRange(NamedTuple):
  name: str
  start: int
  slots: int    # addresses in the range
  size: int     # frame size of the known layout, bytes
  kind: str     # "radar", "aux" (radar side data) or "camera" (object list from the camera, not radar)


RANGES = (
  RadarRange("500", 0x500, 64, 8, "radar"),      # legacy front radar, 32 or 64 slots
  RadarRange("602", 0x602, 22, 8, "radar"),      # legacy alternate front radar
  RadarRange("210", 0x210, 16, 32, "radar"),     # CAN-FD front radar, detailed or compact
  RadarRange("3A5", 0x3A5, 32, 24, "radar"),     # CAN-FD common front radar
  RadarRange("3D0", 0x3D0, 5, 32, "aux"),        # auxiliary scan list next to some 3A5 radars
  RadarRange("235", 0x235, 20, 32, "camera"),    # forward camera object list
)
SAMPLE_ADDRS = frozenset(r.start for r in RANGES)
MIN_FRAMES = 20   # same bar as the steering messages: a stray frame doesn't count


def _range_entry(tally: HarnessTally, r: RadarRange, bus: int) -> dict | None:
  present = [a for a in range(r.start, r.start + r.slots) if tally.counts.get((bus, a), 0) >= MIN_FRAMES]
  if len(present) < max(2, r.slots // 2):
    return None
  sample = tally.samples.get((bus, r.start), b"")
  return {"range": r.name, "kind": r.kind, "bus": bus, "present": len(present), "expected": r.slots,
          "sizes": sorted({tally.sizes[(bus, a)] for a in present}), "hz": round(median(tally.hz((bus, a)) for a in present), 1),
          "sample": sample.hex()}


def _verdict(entries: list[dict]) -> tuple[str, str, str]:
  radar = {e["range"]: e for e in entries if e["kind"] == "radar"}
  for name in ("210", "3A5"):
    e = radar.get(name)
    if e is None:
      continue
    known = next(r.size for r in RANGES if r.name == name)
    if e["sizes"] != [known]:
      sizes = "/".join(str(s) for s in e["sizes"])
      return ("unknown_layout", f"Radar at {name} with {sizes} B frames: unknown layout",
              f"The {name} range is on bus {e['bus']}, but the known layout uses {known} B frames. The tracks need decoding first.")
    if name == "210":
      return ("supported_210", f"210 radar on bus {e['bus']}: decodable",
              "The hyundai-radar-tracks decoder reads this layout (detailed or compact).")
    return ("size_match_3a5", f"3A5 radar on bus {e['bus']}: layout unconfirmed",
            "Frame size matches the common 3A5 layout, but the 2026 Sportage sends these addresses with a payload that layout " +
            "can't read. Check the sample frame before decoding.")
  legacy = [radar[n] for n in ("500", "602") if n in radar]
  if legacy:
    e = legacy[0]
    return ("legacy", f"Legacy {e['range']} radar on bus {e['bus']}", f"{e['present']} track slots at {e['hz']} Hz.")
  return ("none", "No radar tracks found",
          "None of the known radar ranges are on the buses this harness exposes, or the radar is off.")


def survey(tally: HarnessTally) -> dict:
  buses = sorted({bus for bus, _ in tally.counts})
  entries = [e for r in RANGES for bus in buses if (e := _range_entry(tally, r, bus)) is not None]
  verdict, title, detail = _verdict(entries)
  return {"verdict": verdict, "title": title, "detail": detail, "ranges": entries}
