"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pytest

from openpilot.sunnypilot.selfdrive.harness.harness_check import HarnessTally
from openpilot.sunnypilot.selfdrive.harness.radar_survey import RANGES, survey

BY_NAME = {r.name: r for r in RANGES}


def radar(tally: HarnessTally, name: str, bus: int, size: int | None = None, slots: int | None = None, hz: float = 20.0,
          frames: int = 60) -> None:
  r = BY_NAME[name]
  size = r.size if size is None else size
  for addr in range(r.start, r.start + (r.slots if slots is None else slots)):
    for i in range(frames):
      tally.ingest(int(i / hz * 1e9), bus, addr, size, bytes([addr & 0xFF, i & 0xFF]) + bytes(size - 2))


def found(result: dict, name: str) -> list[dict]:
  return [r for r in result["ranges"] if r["range"] == name]


class TestVerdict:
  def test_210_radar_is_supported(self):
    t = HarnessTally()
    radar(t, "210", 1)
    r = survey(t)
    assert r["verdict"] == "supported_210"
    assert "210" in r["title"]

  def test_3a5_at_24_bytes_matches_by_size_only(self):
    t = HarnessTally()
    radar(t, "3A5", 1)
    r = survey(t)
    assert r["verdict"] == "size_match_3a5"
    assert "2026 Sportage" in r["detail"]

  def test_3a5_at_another_size_is_an_unknown_layout(self):
    t = HarnessTally()
    radar(t, "3A5", 1, size=32)
    r = survey(t)
    assert r["verdict"] == "unknown_layout"
    assert "32" in r["title"]

  def test_210_at_another_size_is_an_unknown_layout(self):
    t = HarnessTally()
    radar(t, "210", 0, size=24)
    assert survey(t)["verdict"] == "unknown_layout"

  def test_legacy_500_with_32_slots(self):
    t = HarnessTally()
    radar(t, "500", 1, slots=32)
    r = survey(t)
    assert r["verdict"] == "legacy"
    assert found(r, "500")[0]["present"] == 32

  def test_legacy_602(self):
    t = HarnessTally()
    radar(t, "602", 1, hz=10.0)
    assert survey(t)["verdict"] == "legacy"

  def test_nothing_found(self):
    t = HarnessTally()
    for bus in (0, 1, 2):
      for i in range(60):
        t.ingest(int(i * 1e7), bus, 0x1A0, 32)
    r = survey(t)
    assert r["verdict"] == "none" and r["ranges"] == []

  def test_camera_objects_alone_are_not_radar(self):
    t = HarnessTally()
    radar(t, "235", 1, hz=33.0)
    r = survey(t)
    assert r["verdict"] == "none"
    assert found(r, "235")[0]["kind"] == "camera"

  def test_210_wins_over_camera_and_aux_traffic(self):
    t = HarnessTally()
    radar(t, "235", 1, hz=33.0)
    radar(t, "3D0", 1)
    radar(t, "210", 1)
    assert survey(t)["verdict"] == "supported_210"


class TestRanges:
  def test_a_few_stray_addresses_do_not_make_a_range(self):
    t = HarnessTally()
    radar(t, "3A5", 1, slots=3)
    assert survey(t)["ranges"] == []

  def test_frames_below_min_frames_do_not_count(self):
    t = HarnessTally()
    radar(t, "210", 1, frames=5)
    assert survey(t)["verdict"] == "none"

  def test_each_bus_is_reported_separately(self):
    t = HarnessTally()
    radar(t, "3A5", 0)
    radar(t, "3A5", 2)
    assert sorted(r["bus"] for r in found(survey(t), "3A5")) == [0, 2]

  def test_size_rate_and_sample(self):
    t = HarnessTally()
    radar(t, "3A5", 1, hz=20.0)
    entry = found(survey(t), "3A5")[0]
    assert entry["present"] == 32 and entry["expected"] == 32
    assert entry["sizes"] == [24]
    assert entry["hz"] == pytest.approx(20.0, rel=0.01)
    assert entry["sample"].startswith("a5")       # first address 0x3A5, last frame captured
    assert len(entry["sample"]) == 48               # 24 bytes as hex

  def test_mixed_sizes_are_all_listed(self):
    t = HarnessTally()
    radar(t, "210", 1, slots=10)
    r = BY_NAME["210"]
    for addr in range(r.start + 10, r.start + r.slots):
      for i in range(60):
        t.ingest(int(i / 20 * 1e9), 1, addr, 24, bytes(24))
    assert found(survey(t), "210")[0]["sizes"] == [24, 32]

  def test_empty_tally(self):
    r = survey(HarnessTally())
    assert r["verdict"] == "none" and r["ranges"] == []
    assert set(r) == {"verdict", "title", "detail", "ranges"}
