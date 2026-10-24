#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Harness check service. Runs while the car is on and the HarnessCheckEnabled toggle is set. Listen-only: it reads the 'can'
stream pandad already publishes and never transmits.

Writes HarnessCheck every couple of seconds of CAN time, and a final result once the verdict is settled (15 s, or 60 s if no
steering message has shown up). Then it stops reading CAN and idles until the car turns off.
"""
import time

from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.car.structs import car
from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.selfdrive.harness.harness_check import HarnessSession, Stamp, boot_id, decision_from

IDLE_S = 1.0


def read_decision(params: Params) -> dict:
  """What openpilot made of the car this drive. CarParams may still hold the last drive's car for the first second or two."""
  raw = params.get("CarParams")
  if not raw:
    return decision_from("", "", False)
  CP = messaging.log_from_bytes(raw, car.CarParams)
  return decision_from(CP.carFingerprint, CP.brand, bool(CP.flags & HyundaiFlags.CANFD_LKA_STEER_MSG))


def main() -> None:
  params = Params()
  boot = boot_id()
  session = HarnessSession()
  sock = messaging.sub_sock("can", timeout=100)
  cloudlog.info("harnessd: listening")

  while not session.final:
    for msg in messaging.drain_sock(sock, wait_for_one=True):
      for c in msg.can:
        session.tally.ingest(msg.logMonoTime, c.src, c.address, len(c.dat), c.dat)

    if not session.due():
      continue
    try:
      stamp = Stamp(time.time(), time.monotonic(), boot)  # noqa: TID251  wall clock is only shown as the date of the check
      result = session.result(read_decision(params), stamp)
      params.put("HarnessCheck", result)
    except Exception:
      cloudlog.exception("harnessd: check failed")
      continue
    session.finish(result)
    if session.changed:
      cloudlog.info(f"harnessd: {result['verdict']} after {result['seconds']} s, camera bus {result['camera_bus']}: {result['title']}")

  del sock
  cloudlog.info(f"harnessd: done after {session.tally.seconds:.1f} s, {session.verdict}")
  while True:
    time.sleep(IDLE_S)


if __name__ == "__main__":
  main()
