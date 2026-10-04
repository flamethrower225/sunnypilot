"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

import pyray as rl

from openpilot.cereal import log, messaging
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import distance_bands, gauge_scale, zone_bands
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.lock import LeadLock, MIN_BOX_H, MIN_BOX_W
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone, LeadGapMetrics, LeadGapReading, ZoneHysteresis, \
  STOP_DISTANCE, classify_zone, format_gap, get_t_follow, get_target_gap, primary_value, speed_delta, speed_value

DT = 0.05  # 20 fps, as on the comma 3X
INF = math.inf
P = log.LongitudinalPersonality
STANDARD = 1.45


class TestClassifyZone(OpenpilotTestCase):
  @parameterized.expand([
    ("following", 1.8, 48.0, INF, STANDARD, GapZone.GOOD),
    ("tailgating", 0.5, 15.0, INF, STANDARD, GapZone.CRITICAL),
    ("cut_in", 0.8, 21.0, INF, STANDARD, GapZone.CLOSE),
    ("tight_standard", 1.2, 30.0, INF, STANDARD, GapZone.CAUTION),
    ("tight_is_good_on_aggressive", 1.2, 30.0, INF, 1.25, GapZone.GOOD),
    ("open_road", 3.5, 90.0, INF, STANDARD, GapZone.OPEN),
    ("ttc_critical", 2.0, 50.0, 2.0, STANDARD, GapZone.CRITICAL),
    ("ttc_close", 2.0, 50.0, 3.5, STANDARD, GapZone.CLOSE),
    ("ttc_caution", 2.0, 50.0, 5.0, STANDARD, GapZone.CAUTION),
    ("ttc_overrides_open", 3.5, 90.0, 5.0, STANDARD, GapZone.CAUTION),
  ])
  def test_moving(self, _, gap, d_rel, ttc, t_follow, expected):
    assert classify_zone(gap, d_rel, ttc, t_follow, low_speed=False) == expected

  @parameterized.expand([
    ("bumper", 2.0, GapZone.CRITICAL),
    ("close", 4.0, GapZone.CLOSE),
    ("short_of_stop_distance", 5.0, GapZone.CAUTION),
    ("queued", 8.0, GapZone.GOOD),
    ("never_open_at_a_crawl", 50.0, GapZone.GOOD),
  ])
  def test_low_speed_uses_distance(self, _, d_rel, expected):
    assert classify_zone(INF, d_rel, INF, STANDARD, low_speed=True) == expected


class TestZoneHysteresis(OpenpilotTestCase):
  def test_appear_and_critical_are_immediate(self):
    h = ZoneHysteresis(DT)
    assert h.update(GapZone.GOOD) == GapZone.GOOD
    assert h.update(GapZone.CRITICAL) == GapZone.CRITICAL
    assert h.update(GapZone.NONE) == GapZone.NONE

  def test_escalation_holds_briefly(self):
    h = ZoneHysteresis(DT)
    h.update(GapZone.GOOD)
    zones = [h.update(GapZone.CLOSE) for _ in range(4)]
    assert zones == [GapZone.GOOD] * 3 + [GapZone.CLOSE]

  def test_relaxing_holds_longer(self):
    h = ZoneHysteresis(DT)
    h.update(GapZone.GOOD)
    for _ in range(4):
      h.update(GapZone.CLOSE)
    zones = [h.update(GapZone.GOOD) for _ in range(12)]
    assert zones[:11] == [GapZone.CLOSE] * 11
    assert zones[11] == GapZone.GOOD

  def test_flicker_on_a_threshold_is_ignored(self):
    h = ZoneHysteresis(DT)
    h.update(GapZone.GOOD)
    for i in range(40):
      assert h.update(GapZone.CAUTION if i % 2 else GapZone.GOOD) == GapZone.GOOD


class TestLeadGapMetrics(OpenpilotTestCase):
  def _settle(self, m, n=60, **kw):
    args = {"present": True, "d_rel": 48.0, "v_rel": 0.0, "v_lead": 26.8, "v_ego": 26.8, "personality": P.standard, **kw}
    for _ in range(n):
      reading = m.update(**args)
    return reading

  def test_steady_follow(self):
    r = self._settle(LeadGapMetrics(DT))
    assert r.present and r.zone == GapZone.GOOD
    assert abs(r.gap - 48.0 / 26.8) < 0.01
    assert abs(r.target_gap - (1.45 + STOP_DISTANCE / 26.8)) < 0.01
    assert not r.show_ttc

  def test_new_lead_skips_smoothing(self):
    m = LeadGapMetrics(DT)
    r = m.update(True, 30.0, 0.0, 20.0, 20.0, P.standard)
    assert r.d_rel == 30.0

  def test_cut_in_snaps_and_escalates(self):
    m = LeadGapMetrics(DT)
    self._settle(m)
    r = m.update(True, 18.0, -2.0, 24.8, 26.8, P.standard)
    assert r.d_rel == 18.0
    for _ in range(4):
      r = m.update(True, 18.0, -2.0, 24.8, 26.8, P.standard)
    assert r.zone == GapZone.CLOSE

  def test_jitter_is_smoothed(self):
    m = LeadGapMetrics(DT)
    self._settle(m)
    r = m.update(True, 49.5, 0.0, 26.8, 26.8, P.standard)
    assert 48.0 < r.d_rel < 49.5

  def test_lost_lead_keeps_last_numbers(self):
    m = LeadGapMetrics(DT)
    self._settle(m)
    r = m.update(False, 0.0, 0.0, 0.0, 26.8, P.standard)
    assert not r.present and r.zone == GapZone.NONE
    assert abs(r.d_rel - 48.0) < 0.01

  def test_closing_shows_ttc(self):
    r = self._settle(LeadGapMetrics(DT), d_rel=36.6, v_rel=-7.6, v_lead=17.0, v_ego=24.6)
    assert r.show_ttc and abs(r.ttc - 36.6 / 7.6) < 0.05
    assert r.zone == GapZone.CAUTION

  def test_personality_from_capnp_enum(self):
    msg = messaging.new_message('selfdriveState')
    msg.selfdriveState.personality = 'relaxed'
    r = self._settle(LeadGapMetrics(DT), personality=msg.selfdriveState.personality)
    assert r.t_follow == 1.75


class TestFormatting(OpenpilotTestCase):
  def test_t_follow(self):
    assert get_t_follow(P.aggressive) == 1.25
    assert get_t_follow(P.relaxed) == 1.75
    assert get_t_follow(99) == 1.45
    assert get_target_gap(1.45, 0.0) == 1.45 + STOP_DISTANCE

  def test_format_gap(self):
    assert format_gap(1.84) == "1.8"
    assert format_gap(12.0) == "9.9"
    assert format_gap(INF) == "–"

  def test_primary_value_switches_to_distance_at_a_crawl(self):
    moving = LeadGapReading(present=True, gap=1.75, d_rel=47.0, low_speed=False)
    crawling = LeadGapReading(present=True, gap=4.0, d_rel=6.7, low_speed=True)
    assert primary_value(moving, is_metric=False) == ("1.8", "s")
    assert primary_value(crawling, is_metric=True) == ("7", "m")
    assert primary_value(crawling, is_metric=False) == ("22", "ft")

  def test_speeds(self):
    assert speed_value(26.8224, is_metric=False) == 60
    assert speed_value(-1.0, is_metric=True) == 0
    assert speed_delta(-7.6, is_metric=False) == -17
    assert speed_delta(0.1, is_metric=False) == 0


class TestScales(OpenpilotTestCase):
  @parameterized.expand([(1.25,), (1.45,), (1.75,)])
  def test_zone_bands_are_contiguous(self, t_follow):
    for max_gap in (3.0, 4.0):
      bands = zone_bands(t_follow, max_gap)
      assert bands[0][1] == 0.0 and bands[-1][2] == max_gap
      assert all(a[2] == b[1] for a, b in zip(bands, bands[1:], strict=False))

  def test_distance_bands_are_contiguous(self):
    bands = distance_bands(18.0)
    assert bands[0][1] == 0.0 and bands[-1][2] == 18.0
    assert all(a[2] == b[1] for a, b in zip(bands, bands[1:], strict=False))

  def test_gauge_scale(self):
    moving = LeadGapReading(present=True, gap=5.0, t_follow=1.45, target_gap=1.67)
    scale = gauge_scale(moving, False, 3.0, (0, 6, 12, 18), (0, 20, 40, 60))
    assert scale.unit == "s" and scale.value == 3.0 and scale.target == 1.67

    crawling = LeadGapReading(present=True, d_rel=30.0, low_speed=True)
    scale = gauge_scale(crawling, True, 3.0, (0, 6, 12, 18), (0, 20, 40, 60))
    assert scale.unit == "m" and scale.max_value == 18.0 and scale.value == 18.0 and scale.target is None
    assert [label for _, label in scale.ticks] == ["0", "6", "12", "18"]


class TestLeadLockBox(OpenpilotTestCase):
  def test_far_lead_box_has_a_minimum_size(self):
    box = LeadLock.pad_box(1000, 400, 1010, 408, rl.Rectangle(0, 0, 2100, 1020))
    assert box[2] - box[0] == MIN_BOX_W
    assert box[3] - box[1] >= MIN_BOX_H

  def test_close_lead_box_is_capped(self):
    box = LeadLock.pad_box(100, 100, 2000, 900, rl.Rectangle(0, 0, 2100, 1020))
    assert box[2] - box[0] == 2100 * 0.6
