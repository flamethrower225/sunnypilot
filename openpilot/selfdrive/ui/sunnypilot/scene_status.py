"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The Developer panel's view of Camera Object Detection: what scened last wrote to SceneDetectionStatus (every 30 s while driving),
as a page a person can read: how fast it runs, how often it was held back, how its distances are calibrated, and what it saw.
"""
from openpilot.selfdrive.ui.sunnypilot.comm_issue_watch import _safe

PASS_NAMES = {
  "lights_near": "traffic lights, near",
  "lights_far": "traffic lights, far",
  "road": "road camera",
  "wide": "wide camera",
}
THROTTLED_HINT = 30.0       # % of passes held back before it's worth a word
SCALE_HINT = 0.12           # a range scale this far from 1 points to camera mounting or calibration
SLOW_PASS_MS = 120.0


def _hints(s: dict) -> list[str]:
  hints = []
  if not s.get("compiled", True):
    hints.append("the detector is running its ONNX fallback, which is slow: the compiled model was not built for this device")
  if s.get("throttled_pct", 0.0) > THROTTLED_HINT:
    hints.append(f"{s['throttled_pct']}% of passes were held back for dropped model frames or heat")
  for key, camera in (("range_scale", "road"), ("range_scale_wide", "wide")):
    scale = s.get(key, 1.0)
    if abs(scale - 1.0) > SCALE_HINT:
      hints.append(f"the {camera} camera's distances needed a {100 * (scale - 1):+.0f}% correction against the radar: " +
                   "check the device mount and recalibrate")
  slow = [name for name, p in s.get("passes", {}).items() if p.get("avg_ms", 0.0) > SLOW_PASS_MS]
  if slow:
    hints.append("slow passes: " + ", ".join(PASS_NAMES.get(n, n) for n in slow))
  return hints


def scene_status_html(status: dict | None, enabled: bool) -> str:
  if not status:
    if not enabled:
      return "<h2>Camera Object Detection is off</h2><p>Turn it on in Visuals; its status shows here after a drive.</p>"
    return "<h2>No status yet</h2><p>Camera Object Detection writes its status here every 30 seconds while driving.</p>"
  s = status
  parts = [f"<h2>Camera Object Detection</h2><p>As of {_safe(s.get('time', ''))}, {s.get('uptime_s', 0) // 60} min into the drive"
           + ("" if enabled else " (it has been turned off since)") + ".</p>"]

  lines = [
    f"running at {s.get('pass_rate', 0)} passes per second, {s.get('throttled_pct', 0)}% held back, " +
    f"{s.get('lights_share_pct', 0)}% of passes on traffic lights",
    f"detector: {'compiled' if s.get('compiled', True) else 'ONNX fallback'}",
  ]
  for name, p in s.get("passes", {}).items():
    lines.append(f"{_safe(PASS_NAMES.get(name, name))}: {p.get('count', 0)} passes, {p.get('avg_ms', 0)} ms each")
  matches = s.get("calibration_matches", 0)
  lines.append(f"distance scale: road {s.get('range_scale', 1.0):.3f}, wide {s.get('range_scale_wide', 1.0):.3f} "
               + (f"(from {matches} radar matches)" if matches else "(not yet calibrated against the radar)"))
  lines.append(f"most at once: {s.get('max_vehicles', 0)} cars, {s.get('max_people', 0)} people; {s.get('stop_signs', 0)} stop signs")
  parts.append("<ul>" + "".join(f"<li>{line}</li>" for line in lines) + "</ul>")

  hints = _hints(s)
  if hints:
    parts.append("<p>Points to:</p><ul>" + "".join(f"<li>{_safe(h)}</li>" for h in hints) + "</ul>")
  events = s.get("events", [])
  if events:
    parts.append("<h3>Recently seen</h3><ul>"
                 + "".join(f"<li>{_safe(e.get('time', ''))} {_safe(e.get('event', ''))}</li>" for e in events) + "</ul>")
  return "".join(parts)
