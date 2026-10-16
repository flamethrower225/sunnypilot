#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Dash research service. Runs while the car is on and the DashReEnabled toggle is set. Listen-only: it reads
the 'can' stream pandad already publishes and never transmits.

Talks to the settings UI through params:
  DashReFindings  written every few seconds: what is on each bus, measured against what the cluster reads
  DashReMark      written by the UI when the driver marks an action: {"id", "label", "t"} (t = monotonic ns)
  DashReResults   written after each mark resolves: the bits that changed, newest first
"""
import time

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.tools.dash_re.engine import DashReEngine

FINDINGS_PERIOD_S = 5.
MARK_POLL_S = 0.25


def main() -> None:
  params = Params()
  engine = DashReEngine()
  engine.results = params.get("DashReResults") or []
  sock = messaging.sub_sock("can", timeout=100)

  last_mark_id = (params.get("DashReMark") or {}).get("id")
  next_findings = time.monotonic() + FINDINGS_PERIOD_S
  next_mark_poll = 0.
  cloudlog.info("dashred: listening")

  while True:
    for msg in messaging.drain_sock(sock, wait_for_one=True):
      for c in msg.can:
        engine.ingest(msg.logMonoTime, c.src, c.address, bytes(c.dat))

    now = time.monotonic()
    if now >= next_mark_poll:
      next_mark_poll = now + MARK_POLL_S
      mark = params.get("DashReMark")
      if mark and mark.get("id") != last_mark_id:
        last_mark_id = mark.get("id")
        engine.mark(str(mark.get("label", "mark")), int(mark.get("t", 0)))
      if engine.process_marks():
        params.put("DashReResults", engine.results)

    if now >= next_findings:
      next_findings = now + FINDINGS_PERIOD_S
      params.put("DashReFindings", engine.findings())


if __name__ == "__main__":
  main()
