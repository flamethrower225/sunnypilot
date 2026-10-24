"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The Developer panel's view of the harness check: what harnessd last wrote to HarnessCheck, as a page a person can read.
Every tag starts its own line in the settings viewer, so nothing is bolded inline.
"""
import time

from openpilot.selfdrive.ui.sunnypilot.comm_issue_watch import _safe


def _decision_line(decision: dict | None, disagrees: bool) -> str:
  if not decision or not decision.get("recognized"):
    return "openpilot has not recognised this car (no fingerprint match), so it will not steer it until that is fixed too."
  line = f"openpilot identified the car as {_safe(decision.get('car', ''))}"
  hda2 = decision.get("hda2")
  if hda2 is None:
    return line + "."
  line += f" and set it up as {'HDA2' if hda2 else 'non-HDA2'}."
  if disagrees:
    line += " That disagrees with the wiring above: the car may have been identified from an earlier drive or a different harness."
  return line


def _bus_line(bus: dict, camera_bus: int | None) -> str:
  label = f"bus {bus.get('bus')}" + (" (camera bus)" if bus.get("bus") == camera_bus else "")
  steering = bus.get("steering") or {}
  seen = ", ".join(f"{_safe(name)} {s.get('hz', 0)} Hz" for name, s in steering.items()) or "no steering messages"
  ids = bus.get("ids", 0)
  return f"{label}: {bus.get('frames', 0)} frames, {ids} message ID{'' if ids == 1 else 's'}; {seen}"


def harness_status_html(result: dict | None) -> str:
  if not result:
    return ("<h2>No harness check yet</h2><p>Turn the car on with the comma plugged in; the result shows here about 15 seconds " +
            "later.</p>")
  parts = [f"<h2>{_safe(result.get('title', 'Harness check'))}</h2>"]
  if not result.get("final", False):
    parts.append("<p>Still listening: this is a provisional result and may change in the next few seconds.</p>")
  if result.get("detail"):
    parts.append(f"<p>{_safe(result['detail'])}</p>")
  if result.get("action"):
    parts.append(f"<h3>What to do</h3><p>{_safe(result['action'])}</p>")

  parts.append(f"<h3>What openpilot decided</h3><p>{_decision_line(result.get('decision'), result.get('disagrees', False))}</p>")

  buses = result.get("buses") or []
  if buses:
    parts.append("<h3>What arrived on each bus</h3><ul>" +
                 "".join(f"<li>{_bus_line(b, result.get('camera_bus'))}</li>" for b in buses) + "</ul>")

  if "t" in result:
    when = time.strftime("%d-%b-%Y %H:%M:%S", time.localtime(result["t"])).upper()
    parts.append(f"<p>Checked over {result.get('seconds', 0)} s of CAN traffic, {when}.</p>")
  return "".join(parts)
