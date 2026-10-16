#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Narrow CAN traffic down to what drives the instrument cluster.

  candidates          what the DBC says the cluster reads, sorted by how reachable it is (no capture needed)
  inventory FILE      every message in a capture: origin, rate, changing bits, counters, checksums,
                      and bits no DBC signal explains
  diff FILE           for each typed label, the bits that changed right after it
  isotp FILE ADDR     reassemble multi-frame payloads on one address (head unit -> cluster text)

FILE is a capture from capture.py (*.dashcap.gz) or an openpilot rlog/qlog.
"""
import argparse
import signal
from collections import Counter

from openpilot.sunnypilot.tools.dash_re import analysis
from openpilot.sunnypilot.tools.dash_re.dbcinfo import (HYUNDAI_DBC, HYUNDAI_MCAN_DBC, TIER_DESCRIPTIONS, all_cluster_candidates,
                                                        parse_dbc, tier)


def cmd_candidates(args) -> None:
  cands = all_cluster_candidates(op_long=not args.stock_long)
  counts = Counter(t for t, _ in cands)
  print(f"{len(cands)} messages carry signals the cluster reads\n")
  for t in sorted(counts):
    print(f"tier {t}: {counts[t]:3d} messages  ({TIER_DESCRIPTIONS[t]})")
  for t, m in cands:
    if t > args.max_tier:
      continue
    sigs = m.cluster_signals or m.signals
    print(f"\n[tier {t}] {m.addr:#05x} {m.name}  sender={m.sender}")
    for s in sigs:
      print(f"    {s.name:<32} bits {s.start}|{s.size}  {s.unit}")


def cmd_inventory(args) -> None:
  frames, markers = analysis.load(args.file)
  dbc = parse_dbc(HYUNDAI_DBC)
  series = analysis.build_series(frames)
  origins = analysis.origins(series)
  stats = analysis.inventory(series)
  print(f"{len(frames)} frames, {len(stats)} (bus, address) pairs, {len(markers)} labels\n")
  print(f"{'bus':>3} {'addr':>6} {'name':<14} {'origin':<9} {'tier':>4} {'hz':>6} {'len':>4}  changing / counters / checksum / undocumented")
  for (bus, addr), st in sorted(stats.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    if args.bus is not None and bus != args.bus:
      continue
    m = dbc.get(addr)
    t = tier(m, op_long=not args.stock_long) if m else (2 if origins[addr] == {2} else 3)
    if args.dash_only and not (m and (m.cluster_signals or m.camera_originated) or origins[addr] == {2}):
      continue
    undoc = analysis.undocumented_changing_bits(st, m)
    lengths = ",".join(map(str, sorted(st.lengths)))
    where = f"{bus:>3} {addr:#06x} {(m.name if m else '?'):<14} {analysis.origin_label(origins[addr]):<9} {t:>4}"
    bits = f"{len(st.changing_bits)} / {st.counters or '-'} / {st.checksum_bytes or '-'} / {undoc or '-'}"
    print(f"{where} {st.hz:6.1f} {lengths:>4}  {bits}")
  unknown_camera = sorted(a for a, b in origins.items() if b == {2} and a not in dbc)
  if unknown_camera:
    print(f"\ncamera-originated addresses missing from the DBC (new tier 2 candidates): {[hex(a) for a in unknown_camera]}")


def cmd_diff(args) -> None:
  frames, markers = analysis.load(args.file)
  if not markers:
    print("no labels in this capture")
    return
  dbc = parse_dbc(HYUNDAI_DBC)
  series = analysis.build_series(frames)
  origins = analysis.origins(series)
  stats = analysis.inventory(series)
  for marker, changes in analysis.marker_diffs(series, markers, args.window, args.window, args.settle, stats):
    if args.label and args.label.lower() not in marker.label.lower():
      continue
    if args.bus is not None:
      changes = [c for c in changes if c.bus == args.bus]
    print(f"\n== {marker.label} ==  ({len(changes)} bits changed)")
    for c in changes[:args.top]:
      m = dbc.get(c.addr)
      sig = next((s.name for s in m.signals if c.bit in s.bits), "(undocumented)") if m else "(unknown msg)"
      dash = " CLU" if m and any(c.bit in s.bits for s in m.cluster_signals) else ""
      where = f"bus{c.bus} {c.addr:#06x} {(m.name if m else '?'):<14} {analysis.origin_label(origins[c.addr]):<7}"
      print(f"  {where} bit {c.bit:2d}  {c.before:>8} -> {c.after:<8} {c.score:.2f}  {sig}{dash}")


def cmd_isotp(args) -> None:
  frames, _ = analysis.load(args.file)
  addr = int(args.addr, 0)
  mcan = parse_dbc(HYUNDAI_MCAN_DBC).get(addr)
  if mcan:
    print(f"{addr:#x} is {mcan.name} on M-CAN (sender {mcan.sender})")
  t0 = frames[0].t if frames else 0
  for t, payload in analysis.isotp_payloads(frames, addr, args.bus):
    print(f"{(t - t0) / 1e9:9.3f}s  {len(payload):4d} B  {analysis.printable(payload)}")


def main() -> None:
  signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # quiet exit when piped into head/less
  parser = argparse.ArgumentParser(description=__doc__.split("more details.\n", 1)[-1], formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--stock-long", action="store_true", help="tiers for stock longitudinal (openpilot sends fewer messages)")
  sub = parser.add_subparsers(dest="cmd", required=True)

  c = sub.add_parser("candidates")
  c.add_argument("--max-tier", type=int, default=2, help="list signals up to this tier (default 2)")
  c.set_defaults(func=cmd_candidates)

  i = sub.add_parser("inventory")
  i.add_argument("file")
  i.add_argument("--bus", type=int)
  i.add_argument("--dash-only", action="store_true", help="only messages the cluster reads or the camera sends")
  i.set_defaults(func=cmd_inventory)

  d = sub.add_parser("diff")
  d.add_argument("file")
  d.add_argument("--label", help="only labels containing this text")
  d.add_argument("--bus", type=int)
  d.add_argument("--window", type=float, default=2.0, help="seconds compared on each side of a label")
  d.add_argument("--settle", type=float, default=0.3, help="seconds skipped after a label (typing delay)")
  d.add_argument("--top", type=int, default=25)
  d.set_defaults(func=cmd_diff)

  t = sub.add_parser("isotp")
  t.add_argument("file")
  t.add_argument("addr", help="e.g. 0x4f2")
  t.add_argument("--bus", type=int)
  t.set_defaults(func=cmd_isotp)

  args = parser.parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
