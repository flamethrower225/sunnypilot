#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Record CAN on the device with driver labels, or watch bits change live. Listen-only: it reads the 'can'
stream pandad already publishes and never transmits.

Run over SSH while openpilot is running (ignition on):

  cd /data/openpilot
  python -m openpilot.sunnypilot.tools.dash_re.capture record           # type a label + Enter at each action
  python -m openpilot.sunnypilot.tools.dash_re.capture watch --bus 2    # live bit changes from the camera
"""
import argparse
import gzip
import os
import select
import sys
import time
from pathlib import Path

from openpilot.cereal import messaging
from openpilot.sunnypilot.tools.dash_re.analysis import CAPTURE_SUFFIX, PANDA_TX_OFFSET, Frame, Marker, write_frame, write_marker
from openpilot.sunnypilot.tools.dash_re.dbcinfo import HYUNDAI_DBC, parse_dbc

DEFAULT_DIR = Path("/data/dash_re") if Path("/data").is_dir() else Path.home() / "dash_re"


def read_label() -> str | None:
  if select.select([sys.stdin], [], [], 0)[0]:
    line = sys.stdin.readline()
    return line.strip() or "mark"
  return None


def record(args) -> None:
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}{'-' + args.name if args.name else ''}{CAPTURE_SUFFIX}"
  sock = messaging.sub_sock("can", timeout=100)
  deadline = time.monotonic() + args.max_minutes * 60
  max_bytes = args.max_mb * 1024 * 1024

  print(f"recording to {path}")
  print("type a label and press Enter at each action (e.g. 'lka button', 'climate 72'); Ctrl+C to stop")
  n_frames = n_marks = 0
  with gzip.open(path, "wt", compresslevel=3) as f:
    try:
      while time.monotonic() < deadline and path.stat().st_size < max_bytes:
        for msg in messaging.drain_sock(sock, wait_for_one=True):
          for c in msg.can:
            if c.src < PANDA_TX_OFFSET or args.include_tx:
              write_frame(f, Frame(msg.logMonoTime, c.src, c.address, bytes(c.dat)))
              n_frames += 1
        if (label := read_label()) is not None:
          write_marker(f, Marker(time.monotonic_ns(), label))
          n_marks += 1
          print(f"  [{n_marks}] {label}")
      else:
        print("size or time limit reached")
    except KeyboardInterrupt:
      pass
  print(f"saved {n_frames} frames, {n_marks} labels: {path}")


def watch(args) -> None:
  """Print every change to the selected messages, with the changed bits and DBC signal names."""
  dbc = parse_dbc(HYUNDAI_DBC)
  addrs = {int(a, 0) for a in args.addr} if args.addr else None
  sock = messaging.sub_sock("can", timeout=100)
  last: dict[tuple[int, int], bytes] = {}
  ignore = {(int(a.split(":")[0], 0), int(a.split(":")[1])) for a in args.ignore_bit}
  print("watching; noisy bits can be muted with --ignore-bit ADDR:BIT. Ctrl+C to stop")
  try:
    while True:
      for msg in messaging.drain_sock(sock, wait_for_one=True):
        for c in msg.can:
          if c.src != args.bus or (addrs is not None and c.address not in addrs):
            continue
          key, dat = (c.src, c.address), bytes(c.dat)
          prev = last.get(key)
          last[key] = dat
          if prev is None or prev == dat:
            continue
          changed = [b for b in range(min(len(dat), len(prev)) * 8)
                     if ((dat[b // 8] ^ prev[b // 8]) >> (b % 8)) & 1 and (c.address, b) not in ignore]
          if not changed:
            continue
          m = dbc.get(c.address)
          names = sorted({s.name for s in m.signals for b in changed if b in s.bits}) if m else []
          name = m.name if m else "?"
          print(f"{time.strftime('%H:%M:%S')} bus{c.src} {c.address:#05x} {name:<14} {dat.hex(' ')}  bits {changed} {names}")
      if (label := read_label()) is not None:
        print(f"---- {label} ----")
  except KeyboardInterrupt:
    pass


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("more details.\n", 1)[-1], formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="cmd", required=True)

  rec = sub.add_parser("record", help="record all CAN with typed labels")
  rec.add_argument("--name", default="", help="suffix for the file name, e.g. climate")
  rec.add_argument("--out-dir", default=os.environ.get("DASH_RE_DIR", str(DEFAULT_DIR)))
  rec.add_argument("--max-minutes", type=float, default=30)
  rec.add_argument("--max-mb", type=float, default=200)
  rec.add_argument("--include-tx", action="store_true", help="also keep what openpilot sends (src >= 128)")
  rec.set_defaults(func=record)

  w = sub.add_parser("watch", help="print bit changes live")
  w.add_argument("--bus", type=int, default=2, help="0 = car side, 2 = camera side (default)")
  w.add_argument("--addr", nargs="*", help="only these addresses, e.g. 0x53e 0x485")
  w.add_argument("--ignore-bit", nargs="*", default=[], help="ADDR:BIT pairs to mute, e.g. 0x340:36")
  w.set_defaults(func=watch)

  args = parser.parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
