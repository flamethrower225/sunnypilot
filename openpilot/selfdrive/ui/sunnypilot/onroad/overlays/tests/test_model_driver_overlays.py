"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

import numpy as np

from openpilot.cereal import log, messaging
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import ZONE_RGB
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.disengage_meter import combine_disengage, slice_hazard, risk_color, \
  horizon_labels, confidence_color, percent_text
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.driver_flags import Debounce, flag_inputs, awareness_percent, awareness_color, \
  PHONE_THRESH, SLEEP_THRESH, FACE_THRESH, ON_DELAY, OFF_DELAY
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.hard_brake_flash import HoldLatch, pulse, HOLD_TIME, PULSE_HZ

DT = 0.05  # 20 fps, as on the comma 3X


class TestCombineDisengage(OpenpilotTestCase):
  def test_matches_the_model_confidence_formula(self):
    brake, gas, steer = [0.1, 0.2, 0.3], [0.05, 0.1, 0.1], [0.02, 0.05, 0.2]
    expected = 1 - (1 - np.array(brake)) * (1 - np.array(gas)) * (1 - np.array(steer))
    np.testing.assert_allclose(combine_disengage(brake, gas, steer), expected)

  def test_one_source_alone(self):
    np.testing.assert_allclose(combine_disengage([0.1, 0.3], [0.0, 0.0], [0.0, 0.0]), [0.1, 0.3])

  def test_empty_everywhere_hides(self):
    assert combine_disengage([], [], []).size == 0

  @parameterized.expand([
    ("gas_missing", [0.1, 0.2], [], [0.1, 0.1], [0.19, 0.28]),
    ("steer_missing", [0.1, 0.2], [0.1, 0.1], [], [0.19, 0.28]),
    ("only_brake", [0.1, 0.2], [], [], [0.1, 0.2]),
  ])
  def test_missing_head_is_skipped(self, _, brake, gas, steer, expected):
    np.testing.assert_allclose(combine_disengage(brake, gas, steer), expected)

  def test_cut_to_the_shortest_present_list(self):
    assert combine_disengage([0.1] * 5, [0.1] * 3, [0.1] * 5).size == 3

  def test_cumulative_never_falls(self):
    out = combine_disengage([0.2, 0.1, 0.3], [0.0] * 3, [0.0] * 3)
    assert np.all(np.diff(out) >= 0)
    np.testing.assert_allclose(out, [0.2, 0.2, 0.3])

  def test_bad_values_are_clamped(self):
    out = combine_disengage([float("nan"), 1.5, -0.2], [0.0] * 3, [0.0] * 3)
    assert np.all((out >= 0) & (out <= 1))
    assert out[0] == 0.0 and out[1] == 1.0


class TestSliceHazard(OpenpilotTestCase):
  def test_constant_hazard_round_trips(self):
    h = 0.04
    cumulative = 1 - (1 - h) ** np.arange(1, 6)
    np.testing.assert_allclose(slice_hazard(cumulative), [h] * 5)

  def test_risk_lands_in_the_slice_where_it_happens(self):
    hazard = slice_hazard([0.0, 0.0, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(hazard, [0.0, 0.0, 0.5, 0.0, 0.0])

  def test_already_certain_does_not_divide_by_zero(self):
    hazard = slice_hazard([1.0, 1.0, 1.0])
    assert np.all(np.isfinite(hazard)) and np.all((hazard >= 0) & (hazard <= 1))

  def test_empty(self):
    assert slice_hazard([]).size == 0


class TestRiskColor(OpenpilotTestCase):
  def test_ends_are_zone_colors(self):
    np.testing.assert_allclose(risk_color(0.0), ZONE_RGB[GapZone.GOOD])
    np.testing.assert_allclose(risk_color(1.0), ZONE_RGB[GapZone.CRITICAL])

  def test_model_thresholds_land_between_the_zones(self):
    green_to_yellow = risk_color(ModelConstants.RYG_GREEN)
    assert green_to_yellow[0] > ZONE_RGB[GapZone.GOOD][0] and green_to_yellow[1] > 200  # past green, not yet red
    np.testing.assert_allclose(risk_color(ModelConstants.RYG_YELLOW), ZONE_RGB[GapZone.CLOSE])

  def test_gets_redder_as_risk_rises(self):
    redness = [risk_color(r)[0] - risk_color(r)[1] for r in np.linspace(0.0, 0.2, 40)]
    assert all(b >= a - 1e-9 for a, b in zip(redness, redness[1:], strict=False))
    assert redness[0] < 0 < redness[-1]


class TestLabelsAndText(OpenpilotTestCase):
  def test_labels_follow_model_times(self):
    assert horizon_labels([2.0, 4.0, 6.0, 8.0, 10.0], 5) == ["2s", "4s", "6s", "8s", "10s"]

  def test_labels_fall_back_when_times_are_missing(self):
    assert horizon_labels([], 3) == ["2s", "4s", "6s"]
    assert horizon_labels([2.0, 4.0], 4) == ["2s", "4s", "6s", "8s"]

  @parameterized.expand([
    ("zero", 0.0, "0%"),
    ("tiny", 0.003, "<1%"),
    ("small", 0.12, "12%"),
    ("big", 0.654, "65%"),
  ])
  def test_percent_text(self, _, p, expected):
    assert percent_text(p) == expected

  def test_confidence_dot(self):
    C = log.ModelDataV2.ConfidenceClass
    assert confidence_color(C.green) == ZONE_RGB[GapZone.GOOD]
    assert confidence_color(C.yellow) == ZONE_RGB[GapZone.CAUTION]
    assert confidence_color(C.red) == ZONE_RGB[GapZone.CRITICAL]


class TestHoldLatch(OpenpilotTestCase):
  def test_quiet_until_first_signal(self):
    latch = HoldLatch(HOLD_TIME)
    assert not latch.update(False, 0.0)
    assert not latch.update(False, 5.0)

  def test_follows_the_signal_immediately(self):
    latch = HoldLatch(HOLD_TIME)
    assert latch.update(True, 10.0)

  def test_holds_then_releases(self):
    latch = HoldLatch(HOLD_TIME)
    latch.update(True, 10.0)
    assert latch.update(False, 10.0 + DT)
    assert latch.update(False, 10.0 + HOLD_TIME - DT)
    assert not latch.update(False, 10.0 + HOLD_TIME)

  def test_flicker_is_bridged(self):
    latch = HoldLatch(HOLD_TIME)
    shown = []
    for i in range(60):
      active = i % 4 == 0  # on one frame in four
      shown.append(latch.update(active, i * DT))
    assert all(shown)

  def test_signal_during_the_hold_restarts_it(self):
    latch = HoldLatch(HOLD_TIME)
    latch.update(True, 0.0)
    latch.update(True, 0.8)
    assert latch.update(False, 1.7)
    assert not latch.update(False, 1.8)

  def test_reset(self):
    latch = HoldLatch(HOLD_TIME)
    latch.update(True, 0.0)
    latch.reset()
    assert not latch.update(False, 0.1)


class TestPulse(OpenpilotTestCase):
  def test_range_and_period(self):
    values = [pulse(i * 0.005) for i in range(400)]
    assert min(values) >= 0.0 and max(values) <= 1.0
    assert min(values) < 0.01 and max(values) > 0.99
    assert math.isclose(pulse(0.123), pulse(0.123 + 1 / PULSE_HZ), abs_tol=1e-9)

  def test_two_beats_a_second(self):
    peaks = [i for i in range(1, 200) if pulse(i * 0.01 - 0.01) < pulse(i * 0.01) > pulse(i * 0.01 + 0.01)]
    assert len(peaks) == 4  # 2 s of 2 Hz


class TestDebounce(OpenpilotTestCase):
  def run_signal(self, d: Debounce, signal: list[bool]) -> list[bool]:
    return [d.update(s, i * DT) for i, s in enumerate(signal)]

  def test_needs_the_on_delay(self):
    d = Debounce(ON_DELAY, OFF_DELAY)
    frames_to_on = round(ON_DELAY / DT)
    out = self.run_signal(d, [True] * (frames_to_on + 3))
    assert not any(out[:frames_to_on])
    assert out[frames_to_on]

  def test_blip_never_shows(self):
    d = Debounce(ON_DELAY, OFF_DELAY)
    out = self.run_signal(d, [False] * 5 + [True] * 4 + [False] * 40)
    assert not any(out)

  def test_dropout_does_not_clear_it(self):
    d = Debounce(ON_DELAY, OFF_DELAY)
    out = self.run_signal(d, [True] * 10 + [False] * 10 + [True] * 10)
    assert all(out[8:])

  def test_off_after_the_off_delay(self):
    d = Debounce(ON_DELAY, OFF_DELAY)
    signal = [True] * 10 + [False] * 40
    out = self.run_signal(d, signal)
    off_at = out.index(False, 10)
    assert math.isclose((off_at - 10) * DT, OFF_DELAY, abs_tol=DT)

  def test_return_during_off_delay_restarts_it(self):
    d = Debounce(ON_DELAY, OFF_DELAY)
    # two 0.75 s gaps split by a brief return: neither reaches the full second, so it never goes off
    out = self.run_signal(d, [True] * 10 + [False] * 15 + [True] * 2 + [False] * 15)
    assert all(out[8:])

  def test_reset(self):
    d = Debounce(ON_DELAY, OFF_DELAY)
    self.run_signal(d, [True] * 10)
    d.reset()
    assert not d.state


def driver_data(phone: float = 0.0, sleep: float = 0.0, face: float = 1.0):
  msg = messaging.new_message("driverStateV2")
  data = msg.driverStateV2.leftDriverData
  data.phoneProb, data.sleepProb, data.faceProb = phone, sleep, face
  return messaging.log_from_bytes(msg.to_bytes()).driverStateV2.leftDriverData


class TestFlagInputs(OpenpilotTestCase):
  def test_thresholds_come_from_the_monitoring_policy(self):
    assert PHONE_THRESH == 0.5
    assert FACE_THRESH == 0.7
    assert SLEEP_THRESH == 0.5

  @parameterized.expand([
    ("clear", 0.1, 0.1, 0.99, False, False),
    ("phone", 0.9, 0.1, 0.99, True, False),
    ("sleep", 0.1, 0.9, 0.99, False, True),
    ("both", 0.9, 0.9, 0.99, True, True),
    ("just_under_phone", 0.49, 0.0, 0.99, False, False),
    ("no_face_means_no_flags", 0.9, 0.9, 0.2, False, False),
  ])
  def test_flags(self, _, phone, sleep, face, want_phone, want_sleep):
    flags = flag_inputs(driver_data(phone, sleep, face))
    assert flags["phone"] == want_phone
    assert flags["sleep"] == want_sleep


class TestAwareness(OpenpilotTestCase):
  @staticmethod
  def dm_state(policy: str, vision: int, wheeltouch: int):
    msg = messaging.new_message("driverMonitoringState")
    msg.driverMonitoringState.activePolicy = policy
    msg.driverMonitoringState.visionPolicyState.awarenessPercent = vision
    msg.driverMonitoringState.wheeltouchPolicyState.awarenessPercent = wheeltouch
    return messaging.log_from_bytes(msg.to_bytes()).driverMonitoringState

  def test_reads_the_active_policy(self):
    assert awareness_percent(self.dm_state("vision", 72, 15)) == 72
    assert awareness_percent(self.dm_state("wheeltouch", 72, 15)) == 15

  @parameterized.expand([
    ("full", 100, GapZone.GOOD),
    ("fading", 50, GapZone.CAUTION),
    ("about_to_alert", 10, GapZone.CRITICAL),
  ])
  def test_color(self, _, percent, zone):
    assert awareness_color(percent) == ZONE_RGB[zone]
