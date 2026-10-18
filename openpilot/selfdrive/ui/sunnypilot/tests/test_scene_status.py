"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.selfdrive.ui.sunnypilot.scene_status import scene_status_html

STATUS = {
  "time": "2026-10-18 08:12:00", "uptime_s": 1260, "compiled": True, "pass_rate": 5.6, "throttled_pct": 4.2, "lights_share_pct": 50,
  "passes": {"road": {"count": 2100, "avg_ms": 61.0}, "lights_near": {"count": 1800, "avg_ms": 140.0}},
  "range_scale": 1.18, "range_scale_wide": 1.0, "calibration_matches": 220, "max_vehicles": 9, "max_people": 3, "stop_signs": 4,
  "events": [{"time": "08:11:52", "event": "light flashing red at 41 m"}, {"time": "08:10:03", "event": "stop sign at 30 m"}],
}


def test_empty():
  assert "is off" in scene_status_html(None, False)
  assert "No status yet" in scene_status_html({}, True)


def test_full_status_and_hints():
  html = scene_status_html(STATUS, True)
  assert "21 min into the drive" in html and "5.6 passes per second" in html
  assert "road camera: 2100 passes, 61.0 ms each" in html and "from 220 radar matches" in html
  assert "+18% correction" in html and "slow passes: traffic lights, near" in html   # the hints
  assert "light flashing red at 41 m" in html and "turned off" not in html
  assert "turned off" in scene_status_html(STATUS, False)


def test_text_is_escaped():
  html = scene_status_html({**STATUS, "events": [{"time": "x", "event": "<b>bold</b>"}]}, True)
  assert "<b>bold" not in html
