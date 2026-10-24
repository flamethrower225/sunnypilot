"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pytest

from openpilot.selfdrive.ui.sunnypilot.harness_status import harness_status_html
from openpilot.sunnypilot.selfdrive.harness.harness_check import LFA, LKAS, HarnessTally, Stamp, classify, decision_from

NOW = 1_800_000_000.0
STAMP = Stamp(NOW, 500.0, "boot-a")


def result(lkas_bus: int | None = 2, lfa_bus: int | None = None, decision=None, final: bool = True) -> dict:
  t = HarnessTally()
  for bus in (0, 1, 2):
    for i in range(100):
      t.ingest(i * 10_000_000, bus, 0x1A0, 32)
  for bus, addr in ((lkas_bus, LKAS), (lfa_bus, LFA)):
    if bus is not None:
      for i in range(100):
        t.ingest(i * 10_000_000, bus, addr, 32)
  r = classify(t, decision, STAMP)
  r["final"] = final
  return r


def test_no_result_yet():
  for empty in (None, {}):
    assert "No harness check yet" in harness_status_html(empty)


@pytest.mark.parametrize("lkas_bus, lfa_bus", [(2, None), (0, None), (None, 2), (None, 1), (None, None)])
def test_every_verdict_shows_title_detail_and_action(lkas_bus, lfa_bus):
  r = result(lkas_bus, lfa_bus)
  html = harness_status_html(r)
  assert f"<h2>{r['title']}</h2>" in html
  assert r["detail"] in html and r["action"] in html
  assert "<h3>What to do</h3>" in html


def test_provisional_result_says_still_listening():
  assert "Still listening" in harness_status_html(result(final=False))
  assert "Still listening" not in harness_status_html(result(final=True))


def test_per_bus_list_marks_the_camera_bus_and_rates():
  html = harness_status_html(result(lkas_bus=2))
  assert "<li>bus 2 (camera bus): 200 frames, 2 message IDs; LKAS " in html
  assert " Hz" in html
  assert "<li>bus 0: 100 frames, 1 message ID; no steering messages</li>" in html


def test_decision_lines():
  ok = harness_status_html(result(2, decision=decision_from("KIA_SPORTAGE_HEV_2026", "hyundai", True)))
  assert "openpilot identified the car as KIA_SPORTAGE_HEV_2026 and set it up as HDA2." in ok
  assert "disagrees" not in ok

  odd = harness_status_html(result(2, decision=decision_from("KIA_SPORTAGE_HEV_2026", "hyundai", False)))
  assert "set it up as non-HDA2" in odd and "disagrees with the wiring" in odd

  unknown = harness_status_html(result(2, decision=decision_from("MOCK", "mock", False)))
  assert "has not recognised this car" in unknown


def test_dynamic_text_is_escaped():
  r = result(2)
  r["title"] = "<b>bad</b>"
  r["decision"] = {"recognized": True, "car": "<CAR>", "hda2": None}
  html = harness_status_html(r)
  assert "<b>bad</b>" not in html and "<CAR>" not in html


def test_radar_section_lists_each_range():
  r = result(2)
  r["radar"] = {"verdict": "size_match_3a5", "title": "3A5 radar on bus 1: layout unconfirmed", "detail": "Check the sample frame.",
                "ranges": [{"range": "3A5", "kind": "radar", "bus": 1, "present": 32, "expected": 32, "sizes": [24], "hz": 20.0,
                            "sample": "a5003b00"},
                           {"range": "235", "kind": "camera", "bus": 1, "present": 20, "expected": 20, "sizes": [32], "hz": 33.0,
                            "sample": ""}]}
  html = harness_status_html(r)
  assert "<h3>Radar</h3><p>3A5 radar on bus 1: layout unconfirmed: Check the sample frame.</p>" in html
  assert "<li>3A5 on bus 1: 32 of 32 addresses, 24 B, 20.0 Hz; sample a5003b00</li>" in html
  assert "<li>235 (camera objects, not radar) on bus 1: 20 of 20 addresses, 32 B, 33.0 Hz</li>" in html


def test_no_radar_ranges_still_shows_the_verdict():
  r = result(2)
  r["radar"] = {"verdict": "none", "title": "No radar tracks found", "detail": "Radar is off.", "ranges": []}
  html = harness_status_html(r)
  assert "<h3>Radar</h3><p>No radar tracks found: Radar is off.</p>" in html and "<ul><li>3A5" not in html


def test_tolerates_missing_fields():
  html = harness_status_html({"verdict": "swap_needed", "title": "Pin swap needed"})
  assert "Pin swap needed" in html
