"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from unittest import mock

import numpy as np

from openpilot.cereal import messaging
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import STOP_DISTANCE, get_t_follow
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays import detection, pace_border as pb
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.follow_ghost import desired_gap
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.tests.test_ar_overlays import overlay_env
from openpilot.selfdrive.ui.ui_state import ui_state

T_FOLLOW = get_t_follow(1)  # standard personality
MODEL_T = np.linspace(0, 10, 33)


class TestDemand(OpenpilotTestCase):
  def test_settled_at_the_target_gap_needs_nothing(self):
    gap = desired_gap(25.0, 25.0, T_FOLLOW)
    assert pb.lead_demand(gap, 25.0, 25.0, 0.0, T_FOLLOW) == 0.0

  def test_closing_on_a_slower_lead(self):
    # 10 m/s faster with 40 m left before the target gap: v^2 / 2d = 1.25 m/s^2
    gap = desired_gap(25.0, 15.0, T_FOLLOW)
    assert abs(pb.lead_demand(gap + 40.0, 25.0, 15.0, 0.0, T_FOLLOW) - 1.25) < 1e-6

  def test_inside_the_gap_without_closing_still_asks_to_ease_off(self):
    gap = desired_gap(25.0, 25.0, T_FOLLOW)
    half = pb.lead_demand(gap / 2, 25.0, 25.0, 0.0, T_FOLLOW)
    quarter = pb.lead_demand(gap * 0.75, 25.0, 25.0, 0.0, T_FOLLOW)
    assert half > quarter > 0.0

  def test_closing_inside_the_gap_uses_the_stop_distance(self):
    d = pb.lead_demand(STOP_DISTANCE + 10.0, 20.0, 10.0, 0.0, T_FOLLOW)
    assert d >= 10.0 ** 2 / (2 * 10.0)

  def test_a_braking_lead_adds_to_the_demand(self):
    gap = desired_gap(25.0, 15.0, T_FOLLOW)
    calm = pb.lead_demand(gap + 40.0, 25.0, 15.0, 0.0, T_FOLLOW)
    assert abs(pb.lead_demand(gap + 40.0, 25.0, 15.0, -4.0, T_FOLLOW) - (calm + 2.0)) < 1e-6

  def test_stop_demand_leaves_a_margin(self):
    assert abs(pb.stop_demand(82.0, 20.0) - 400 / (2 * 80.0)) < 1e-6
    assert pb.stop_demand(0.0, 5.0) == 25 / (2 * pb.MIN_ROOM)  # past the margin stays finite

  def test_urgency_and_colors(self):
    assert pb.urgency(0.3) == 0.0 and pb.urgency(pb.SHORTFALL_FULL + 1) == 1.0
    assert 0 < pb.urgency(2.0) < 1
    assert pb.urgency_color(0.0) == pb.YELLOW and pb.urgency_color(0.5) == pb.ORANGE and pb.urgency_color(1.0) == pb.RED


# ---- the overlay, fed real messages ----

_clock = [0.0]


def feed(v_ego=25.0, a_ego=0.0, lead=None, stop_at=None, green=False):
  msgs = {n: messaging.new_message(n) for n in ('carState', 'radarState', 'modelV2', 'selfdriveState', 'longitudinalPlanSP')}
  msgs['carState'].carState.vEgo, msgs['carState'].carState.aEgo = v_ego, a_ego
  if lead is not None:
    d_rel, v_lead = lead
    lo = msgs['radarState'].radarState.leadOne
    lo.present, lo.dRel, lo.vLeadK, lo.vRel = True, d_rel, v_lead, v_lead - v_ego
  mv = msgs['modelV2'].modelV2
  if stop_at is None:
    mv.position.x, mv.velocity.x = (v_ego * MODEL_T).tolist(), [v_ego] * 33
  else:
    decel = v_ego ** 2 / (2 * stop_at)
    v = np.clip(v_ego - decel * MODEL_T, 0, None)
    mv.position.x = np.where(v > 0, v_ego * MODEL_T - 0.5 * decel * MODEL_T ** 2, stop_at).tolist()
    mv.velocity.x = v.tolist()
  mv.orientationRate.z = [0.0] * 33
  msgs['longitudinalPlanSP'].longitudinalPlanSP.e2eAlerts.greenLightAlert = green
  for m in msgs.values():
    m.valid = True
  _clock[0] += 0.05
  ui_state.sm.update_msgs(_clock[0], list(msgs.values()))


class TestPaceBorder(OpenpilotTestCase):
  def setup_method(self):
    detection._feed = None  # each test starts with a fresh stop detector

  def run_frames(self, border, n, **kw):
    for _ in range(n):
      feed(**kw)
      border.update()

  def test_closing_on_a_slow_lead_lights_up_and_braking_calms_it(self):
    with overlay_env(pace_border=True, started=True, started_frame=0):
      border = pb.PaceBorder()
      self.run_frames(border, 40, lead=(40.0, 10.0))  # 15 m/s faster, well inside the target gap
      assert border.reason == "lead" and border.level > 0.9
      self.run_frames(border, 80, lead=(40.0, 10.0), a_ego=-8.0)  # the car is already braking hard enough
      assert border.level < 0.05

  def test_comfortable_following_shows_nothing(self):
    with overlay_env(pace_border=True, started=True, started_frame=0):
      border = pb.PaceBorder()
      self.run_frames(border, 40, lead=(desired_gap(25.0, 25.0, T_FOLLOW) + 5, 25.0))
      assert border.level == 0.0

  def test_planned_stop_without_a_lead(self):
    with overlay_env(pace_border=True, started=True, started_frame=0):
      border = pb.PaceBorder()
      self.run_frames(border, 30, stop_at=60.0)  # 25 m/s and a stop 60 m out: needs ~5.4 m/s^2, the car isn't slowing
      assert border.reason == "stop" and border.level > 0.9

  def test_green_light_pulses_green_and_off_switch_resets(self):
    with overlay_env(pace_border=True, started=True, started_frame=0):
      border = pb.PaceBorder()
      self.run_frames(border, 3, v_ego=0.0)
      self.run_frames(border, 1, v_ego=0.0, green=True)
      assert border._go_until > 0 and border.level == 0.0
    with overlay_env(pace_border=False, started=True, started_frame=0):
      self.run_frames(border, 1, lead=(20.0, 5.0))
      assert border.level == 0.0 and border._go_until == 0.0

  def test_flash_rate_never_jumps(self):
    with overlay_env(pace_border=True, started=True, started_frame=0), mock.patch.object(pb.time, "monotonic") as mono:
      mono.return_value = 1000.0
      border = pb.PaceBorder()
      phases = []
      for i in range(60):
        mono.return_value = 1000 + i * 0.05
        feed(lead=(40.0, 10.0) if i > 20 else None)
        border.update()
        phases.append(border._phase)
      steps = [(b - a) % 1.0 for a, b in zip(phases, phases[1:], strict=False)]
      assert max(steps) <= pb.FLASH_HZ[1] * 0.05 + 1e-9  # the phase only ever advances by rate * dt


class TestPapi(OpenpilotTestCase):
  def setup_method(self):
    detection._feed = None

  def test_lights_follow_the_deviation(self):
    assert [pb.red_lights(d, previous=pb.red_lights(d)) for d in (-3.0, -1.0, 0.0, 1.0, 3.0)] == [0, 1, 2, 3, 4]
    assert pb.light_colors(2) == [pb.PAPI_WHITE, pb.PAPI_WHITE, pb.PAPI_RED, pb.PAPI_RED]

  def test_hysteresis_keeps_a_light_from_flickering_on_a_boundary(self):
    assert pb.red_lights(0.55, previous=2) == 2  # just past the 3rd threshold: stays at 2
    assert pb.red_lights(0.7, previous=2) == 3
    assert pb.red_lights(0.45, previous=3) == 3  # just under it on the way back: stays at 3
    assert pb.red_lights(0.3, previous=3) == 2

  def run_frames(self, border, n, **kw):
    for _ in range(n):
      feed(**kw)
      border.update()

  def test_on_profile_shows_two_and_two_without_the_alarm(self):
    gap = desired_gap(25.0, 15.0, T_FOLLOW)
    need = pb.lead_demand(gap + 40.0, 25.0, 15.0, 0.0, T_FOLLOW)  # 1.25 m/s^2
    with overlay_env(pace_border=True, started=True, started_frame=0):
      border = pb.PaceBorder()
      self.run_frames(border, 60, lead=(gap + 40.0, 15.0), a_ego=-need)
      assert border.papi_shown and border.reds == 2 and border.level == 0.0

  def test_too_little_and_too_much_braking(self):
    with overlay_env(pace_border=True, started=True, started_frame=0):
      border = pb.PaceBorder()
      self.run_frames(border, 60, lead=(40.0, 10.0))  # far short of what's needed
      assert border.reds == 4 and border.level > 0.9
      self.run_frames(border, 80, lead=(40.0, 10.0), a_ego=-12.0)  # braking far harder than needed
      assert border.reds == 0 and border.level < 0.05

  def test_nothing_to_slow_for_hides_it(self):
    with overlay_env(pace_border=True, started=True, started_frame=0):
      border = pb.PaceBorder()
      self.run_frames(border, 40, lead=(200.0, 30.0))
      assert not border.papi_shown
