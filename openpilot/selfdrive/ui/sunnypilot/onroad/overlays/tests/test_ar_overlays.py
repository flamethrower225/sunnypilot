"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
from contextlib import contextmanager
from unittest import mock

import numpy as np

from openpilot.cereal import log, messaging
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.curve_hologram import CurveHologram, advisory_speed, curvature_speed, \
  curve_direction, display_speed, find_apex, lat_acc_color, lateral_accels, rail_severity, rails_active
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.follow_ghost import FollowGhost, desired_gap, footprint_outline, ghost_distance, \
  ghost_visibility, place
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.lead_wake import HOLD_TIME, HoldTimer, LeadWake, WakePoint, kinematic_wake_points, \
  lead_braking, model_wake_points, visible_wake_points
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app

DT = 0.05  # 20 fps, as on the comma 3X
MPH = 0.44704
P = log.LongitudinalPersonality
MODEL_T = np.linspace(0, 10, 33)


# ---- running the overlays' update() without a window ----

@contextmanager
def overlay_env(**flags):
  """Patches in the font and frame rate a window would give, and the enable flags."""
  with mock.patch.object(gui_app, "font", return_value=None), mock.patch.object(gui_app, "_target_fps", 20), \
       mock.patch.object(ui_state, "is_metric", False):
    patches = [mock.patch.object(ui_state, k, v) for k, v in flags.items()]
    for p in patches:
      p.start()
    try:
      yield
    finally:
      for p in patches:
        p.stop()


_clock = [0.0]


def feed(**msgs):
  _clock[0] += DT
  ui_state.sm.update_msgs(_clock[0], list(msgs.values()))


def radar_msg(present=True, d_rel=50.0, y_rel=0.0, v_lead=26.8, a_lead=0.0):
  m = messaging.new_message('radarState')
  m.valid = True
  lead = m.radarState.leadOne
  lead.present, lead.dRel, lead.yRel, lead.vLeadK, lead.aLeadK = present, d_rel, y_rel, v_lead, a_lead
  return m


def car_msg(v_ego=26.8):
  m = messaging.new_message('carState')
  m.carState.vEgo = v_ego
  return m


def personality_msg(personality=P.standard):
  m = messaging.new_message('selfdriveState')
  m.selfdriveState.personality = personality
  return m


def model_msg(lat_acc=None, v=26.8, lead_x=None, lead_y=0.0, lead_a=0.0, lead_prob=0.9, bend=1):
  """modelV2 with a path whose predicted lateral acceleration is lat_acc[i] at point i, and a lead whose predicted x (from the
  camera) at t = 0, 2, ... is lead_x."""
  m = messaging.new_message('modelV2')
  m.valid = True
  mv = m.modelV2
  lat = np.zeros(33) if lat_acc is None else np.asarray(lat_acc, dtype=float)
  xs = np.maximum(v * MODEL_T, 3.0 * MODEL_T)
  heading = np.cumsum(bend * lat / max(v, 0.1) * np.gradient(MODEL_T))
  mv.position.x = xs.tolist()
  mv.position.y = np.cumsum(np.sin(heading) * np.gradient(xs)).tolist()
  mv.velocity.x = np.full(33, v).tolist()
  mv.orientationRate.z = (-bend * lat / max(v, 0.1)).tolist()
  leads = mv.init('leadsV3', 3)
  if lead_x is not None:
    leads[0].prob = lead_prob
    leads[0].t = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    leads[0].x = list(lead_x)
    leads[0].y = [lead_y] * 6
    leads[0].a = [lead_a] * 6
  return m


def v_x(v=26.8):
  """Model point distances at a steady speed."""
  return v * MODEL_T


def bell(peak, center, width, floor=0.1):
  """Lateral acceleration at each model point for a bend centered `center` m ahead."""
  return floor + (peak - floor) * np.exp(-0.5 * ((v_x() - center) / width) ** 2)


# ---- curve hologram ----

class TestCurveMath(OpenpilotTestCase):
  def test_lateral_accels(self):
    lat = lateral_accels([0.1, -0.2, 0.0], [20.0, 20.0, 20.0])
    assert np.allclose(lat, [2.0, 4.0, 0.0])

  def test_lateral_accels_truncates_to_the_shorter_series(self):
    assert lateral_accels([0.1, 0.1, 0.1], [10.0, 10.0]).shape == (2,)
    assert lateral_accels([], [10.0]).shape == (0,)

  @parameterized.expand([
    ("straight", 0.0, (52, 209, 122)),
    ("still_green", 1.2, (52, 209, 122)),
    ("yellow", 1.5, (255, 210, 63)),
    ("orange", 2.0, (255, 138, 43)),
    ("red", 2.4, (255, 77, 79)),
    ("harder_than_red", 5.0, (255, 77, 79)),
  ])
  def test_lat_acc_color(self, _, lat_acc, expected):
    assert tuple(round(c) for c in lat_acc_color(lat_acc)) == expected

  def test_color_runs_green_to_red_without_a_gap(self):
    colors = [lat_acc_color(g) for g in np.linspace(1.0, 2.6, 100)]
    assert all(abs(a[i] - b[i]) < 40 for a, b in zip(colors, colors[1:], strict=False) for i in range(3))

  @parameterized.expand([
    ("below", 0.99, False, False),
    ("above", 1.01, False, True),
    ("hold_between", 0.95, True, True),
    ("drop", 0.89, True, False),
    ("no_flicker_on_threshold", 0.95, False, False),
  ])
  def test_rails_hysteresis(self, _, peak, was_active, expected):
    assert rails_active(peak, was_active) == expected

  def test_rail_severity_warns_early_but_never_exceeds_the_peak(self):
    lat = np.array([0.2, 1.0, 2.7, 1.0])
    sev = rail_severity(lat, tint=0.65)
    assert np.allclose(sev, [1.755, 1.755, 2.7, 1.0])
    assert sev.max() == lat.max()
    assert rail_severity([]).shape == (0,)

  @parameterized.expand([("bend_at_50m", 50.0), ("bend_at_75m", 75.0), ("bend_at_110m", 110.0)])
  def test_apex_is_the_peak_of_the_bend(self, _, center):
    xs, lat = v_x(), bell(2.6, center, 16.0)
    idx = find_apex(xs, lat)
    assert idx is not None and abs(xs[idx] - center) < 5.0
    assert lat[idx] >= 0.97 * lat.max()

  def test_apex_ignores_a_peak_beyond_120_m(self):
    xs = v_x()
    lat = 0.1 + 2.6 * np.exp(-0.5 * ((xs - 200.0) / 20.0) ** 2)
    assert find_apex(xs, lat) is None

  def test_apex_ignores_a_bend_we_are_leaving(self):
    xs = v_x()
    lat = 2.5 * np.exp(-xs / 25.0)  # strongest right here, easing off ahead
    assert find_apex(xs, lat) is None

  def test_apex_is_not_a_distant_lesser_bump_when_the_peak_is_out_of_window(self):
    xs = v_x()
    lat = 0.1 + 1.2 * np.exp(-0.5 * ((xs - 60.0) / 8.0) ** 2) + 2.8 * np.exp(-0.5 * ((xs - 200.0) / 20.0) ** 2)
    assert find_apex(xs, lat) is None

  def test_constant_radius_bend_gets_its_gate_at_the_first_point_past_10_m(self):
    xs = v_x()
    idx = find_apex(xs, np.full(33, 2.0))
    assert idx is not None and xs[idx] >= 10.0 and xs[idx - 1] < 10.0

  def test_apex_picks_the_strongest_among_the_near_peak_points(self):
    xs = np.array([0.0, 20.0, 40.0, 60.0, 80.0])
    lat = np.array([0.0, 2.0, 2.6, 2.58, 0.5])
    assert find_apex(xs, lat) == 2

  @parameterized.expand([("flat", [0.0] * 5), ("empty", [])])
  def test_apex_needs_a_bend(self, _, lat):
    assert find_apex(np.arange(len(lat)) * 20.0, lat) is None

  def test_curve_direction(self):
    xs = np.array([0.0, 10.0, 20.0, 30.0])
    assert curve_direction(xs, [0.0, 0.0, 1.0, 3.0], 2) == 1    # path curls to the right (y positive)
    assert curve_direction(xs, [0.0, 0.0, -1.0, -3.0], 2) == -1

  def test_curvature_speed_is_the_speed_for_2_m_s2(self):
    v = 26.8
    yaw_rate = 3.0 / v  # 3 m/s² at this speed, curvature 3 / v²
    v_adv = curvature_speed(yaw_rate, v)
    assert math.isclose(v_adv, math.sqrt(2.0 / (yaw_rate / v)))
    assert math.isclose(v_adv ** 2 * (yaw_rate / v), 2.0)
    assert 21.0 < v_adv < 22.0  # about 48 mph
    assert curvature_speed(-yaw_rate, v) == v_adv

  def test_straight_road_has_no_advisory_speed(self):
    assert curvature_speed(0.0, 26.8) == math.inf

  def test_scc_vision_target_wins_while_it_is_active(self):
    assert advisory_speed(True, 17.0, 0.11, 26.8) == 17.0
    assert advisory_speed(False, 17.0, 0.11, 26.8) == curvature_speed(0.11, 26.8)
    assert advisory_speed(True, 0.0, 0.11, 26.8) == curvature_speed(0.11, 26.8)  # nothing computed yet

  @parameterized.expand([
    ("mph", 22.4, False, 50),
    ("kph", 22.4, True, 80),
    ("up_at_the_midpoint", 12.5 * MPH, False, 15),
    ("down_below_it", 12.4 * MPH, False, 10),
    ("at_least_one_step", 1.0, False, 5),
    ("a_crawl_in_kph", 0.5, True, 5),
  ])
  def test_display_speed_rounds_to_5(self, _, v, is_metric, expected):
    assert display_speed(v, is_metric) == expected

  def test_display_speed_none_when_unbounded(self):
    assert display_speed(math.inf, False) is None


class TestCurveHologramUpdate(OpenpilotTestCase):
  def _run(self, holo, frames, lat, **kw):
    for _ in range(frames):
      feed(modelV2=model_msg(lat, **kw), longitudinalPlanSP=messaging.new_message('longitudinalPlanSP'))
      holo.update()

  def test_straight_road_shows_nothing(self):
    with overlay_env(curve_hologram=True):
      holo = CurveHologram()
      self._run(holo, 20, np.full(33, 0.2))
      assert holo._rail_alpha.x < 0.01 and holo._gate_alpha.x < 0.01

  def test_bend_fades_rails_and_gate_in_then_out(self):
    with overlay_env(curve_hologram=True):
      holo = CurveHologram()
      self._run(holo, 3, bell(2.6, 50.0, 16.0))
      assert 0.0 < holo._rail_alpha.x < 0.8  # fading in, not popping
      self._run(holo, 30, bell(2.6, 50.0, 16.0))
      assert holo._rail_alpha.x > 0.95 and holo._gate_alpha.x > 0.95
      assert holo._speed == 55 and holo._direction == 1  # 26.8 * sqrt(2 / 2.6) = 23.5 m/s = 52.6 mph
      assert abs(holo._apex_x.x - 50.0) < 6.0
      self._run(holo, 40, np.full(33, 0.2))
      assert holo._rail_alpha.x < 0.01 and holo._gate_alpha.x < 0.01

  def test_left_bend_points_left(self):
    with overlay_env(curve_hologram=True):
      holo = CurveHologram()
      self._run(holo, 5, bell(2.6, 50.0, 16.0), bend=-1)
      assert holo._direction == -1

  def test_rails_hold_between_the_thresholds(self):
    with overlay_env(curve_hologram=True):
      holo = CurveHologram()
      self._run(holo, 30, np.full(33, 1.1))
      assert holo._on
      self._run(holo, 30, np.full(33, 0.95))
      assert holo._on and holo._rail_alpha.x > 0.95
      self._run(holo, 30, np.full(33, 0.85))
      assert not holo._on

  def test_scc_vision_target_is_the_label_while_active(self):
    scc = messaging.new_message('longitudinalPlanSP')
    scc.longitudinalPlanSP.smartCruiseControl.vision.active = True
    scc.longitudinalPlanSP.smartCruiseControl.vision.vTarget = 20.0
    with overlay_env(curve_hologram=True):
      holo = CurveHologram()
      for _ in range(5):
        feed(modelV2=model_msg(bell(2.6, 50.0, 16.0)), longitudinalPlanSP=scc)
        holo.update()
      assert holo._speed == 45  # 44.7 mph

  def test_disabled_flag_resets_and_draws_nothing(self):
    with overlay_env(curve_hologram=True):
      holo = CurveHologram()
      self._run(holo, 30, bell(2.6, 50.0, 16.0))
    with overlay_env(curve_hologram=False):
      holo.update()
      assert holo._rail_alpha.x == 0.0 and holo._gate_alpha.x == 0.0 and not holo._on
      holo.render(mock.MagicMock(), mock.MagicMock())  # returns before touching the projector


# ---- follow ghost ----

class TestGhostMath(OpenpilotTestCase):
  def test_desired_gap_matches_the_mpc(self):
    v, tf = 26.8224, 1.45
    assert math.isclose(desired_gap(v, v, tf), tf * v + 6.0)  # same speed: the quadratic terms cancel
    assert math.isclose(desired_gap(v, 0.0, tf), v ** 2 / 5 + tf * v + 6.0)  # stopped lead
    assert math.isclose(desired_gap(30.0, 20.0, 1.75), 900 / 5 + 1.75 * 30 + 6 - 400 / 5)

  def test_desired_gap_grows_with_personality_and_speed(self):
    assert desired_gap(25, 25, 1.25) < desired_gap(25, 25, 1.45) < desired_gap(25, 25, 1.75)
    assert desired_gap(20, 20, 1.45) < desired_gap(30, 30, 1.45)

  @parameterized.expand([
    ("far_lead", 100.0, 26.8224, 24.59, 1.45, 100.0 - (26.8224 ** 2 / 5 + 1.45 * 26.8224 + 6 - 24.59 ** 2 / 5)),
    ("steady_follow", 70.0, 26.8224, 26.8224, 1.45, 70.0 - (1.45 * 26.8224 + 6)),
    ("too_close_is_behind_us", 30.0, 26.8224, 26.8224, 1.45, 30.0 - (1.45 * 26.8224 + 6)),
    ("standstill", 20.0, 0.0, 0.0, 1.45, 14.0),
    ("faster_lead_never_closer_than_stop_distance", 40.0, 10.0, 30.0, 1.45, 34.0),
  ])
  def test_ghost_distance(self, _, d_rel, v_ego, v_lead, t_follow, expected):
    assert math.isclose(ghost_distance(d_rel, v_ego, v_lead, t_follow), expected, abs_tol=1e-6)

  def test_too_close_puts_the_ghost_behind_the_bumper(self):
    assert ghost_distance(30.0, 26.8224, 26.8224, 1.45) < 0.0

  @parameterized.expand([
    ("behind", -5.0, 0.0),
    ("at_threshold", 4.0, 0.0),
    ("halfway", 5.0, 0.5),
    ("full", 6.0, 1.0),
    ("far", 60.0, 1.0),
    ("fading_far", 120.0, 0.5),
    ("too_far", 135.0, 0.0),
  ])
  def test_ghost_visibility(self, _, x, expected):
    assert math.isclose(ghost_visibility(x), expected, abs_tol=1e-6)

  def test_footprint_is_a_car_sized_rounded_rectangle(self):
    pts = footprint_outline(4.5, 1.8, 0.5)
    us, vs = [p[0] for p in pts], [p[1] for p in pts]
    assert math.isclose(max(us), 2.25) and math.isclose(min(us), -2.25)
    assert math.isclose(max(vs), 0.9) and math.isclose(min(vs), -0.9)
    # corners are cut: no point is at a corner of the bounding box
    assert all(not (abs(u) > 2.2 and abs(v) > 0.85) for u, v in pts)
    # convex, in order: every turn has the same sign
    turns = [(b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
             for a, b, c in zip(pts, pts[1:] + pts[:1], pts[2:] + pts[:2], strict=True)]
    assert all(t >= -1e-9 for t in turns) or all(t <= 1e-9 for t in turns)

  def test_short_footprint_keeps_its_corners_inside(self):
    pts = footprint_outline(0.8, 1.8, 0.5)
    assert max(abs(p[0]) for p in pts) <= 0.4 + 1e-9

  def test_place_on_a_straight_path(self):
    assert place([(1.0, 0.5)], 20.0, 1.0, 0.0) == [(21.0, 1.5)]

  def test_place_follows_the_heading_to_the_right(self):
    (x, y), = place([(1.0, 0.0)], 20.0, 0.0, math.pi / 2)  # pointing right: forward becomes +y
    assert math.isclose(x, 20.0, abs_tol=1e-9) and math.isclose(y, 1.0)
    (x, y), = place([(0.0, 1.0)], 20.0, 0.0, math.pi / 2)  # and the car's right side points back
    assert math.isclose(x, 19.0) and math.isclose(y, 0.0, abs_tol=1e-9)


class TestFollowGhostUpdate(OpenpilotTestCase):
  def _run(self, ghost, frames, **radar):
    for _ in range(frames):
      feed(radarState=radar_msg(**radar), carState=car_msg(), selfdriveState=personality_msg())
      ghost.update()

  def test_far_lead_shows_the_ghost_ahead(self):
    with overlay_env(follow_ghost=True):
      ghost = FollowGhost()
      self._run(ghost, 40, d_rel=100.0, v_lead=55 * MPH)
      assert ghost._alpha.x > 0.95
      assert math.isclose(ghost._x.x, ghost_distance(100.0, 26.8, 55 * MPH, 1.45), abs_tol=0.01)

  def test_ghost_uses_the_selected_personality(self):
    with overlay_env(follow_ghost=True):
      for personality, t_follow in ((P.aggressive, 1.25), (P.relaxed, 1.75)):
        ghost = FollowGhost()
        for _ in range(3):
          feed(radarState=radar_msg(d_rel=100.0, v_lead=26.8), carState=car_msg(), selfdriveState=personality_msg(personality))
          ghost.update()
        assert math.isclose(ghost._x.x, ghost_distance(100.0, 26.8, 26.8, t_follow), abs_tol=1e-3)

  def test_ghost_fades_out_when_we_are_there(self):
    with overlay_env(follow_ghost=True):
      ghost = FollowGhost()
      self._run(ghost, 40, d_rel=100.0, v_lead=26.8)
      assert ghost._alpha.x > 0.95
      self._run(ghost, 40, d_rel=50.0, v_lead=26.8)  # desired gap is 44.9 m, but we got there with the ghost at 5 m
      self._run(ghost, 40, d_rel=44.0, v_lead=26.8)
      assert ghost._alpha.x < 0.01

  def test_no_lead_fades_out(self):
    with overlay_env(follow_ghost=True):
      ghost = FollowGhost()
      self._run(ghost, 40, d_rel=100.0, v_lead=26.8)
      self._run(ghost, 40, present=False)
      assert ghost._alpha.x < 0.01

  def test_new_lead_snaps_and_small_changes_are_smoothed(self):
    with overlay_env(follow_ghost=True):
      ghost = FollowGhost()
      self._run(ghost, 1, d_rel=100.0, v_lead=26.8)
      assert math.isclose(ghost._x.x, ghost_distance(100.0, 26.8, 26.8, 1.45), abs_tol=1e-3)
      before = ghost._x.x
      self._run(ghost, 1, d_rel=102.0, v_lead=26.8)
      assert before < ghost._x.x < before + 2.0
      self._run(ghost, 1, d_rel=160.0, v_lead=26.8)  # a different car
      assert math.isclose(ghost._x.x, ghost_distance(160.0, 26.8, 26.8, 1.45), abs_tol=1e-3)

  def test_disabled_flag_resets(self):
    with overlay_env(follow_ghost=True):
      ghost = FollowGhost()
      self._run(ghost, 40, d_rel=100.0, v_lead=26.8)
    with overlay_env(follow_ghost=False):
      ghost.update()
      assert ghost._alpha.x == 0.0
      ghost.render(mock.MagicMock(), mock.MagicMock())


# ---- lead wake ----

class TestLeadBraking(OpenpilotTestCase):
  @parameterized.expand([
    ("radar_hard_brake", -2.0, 0.0, 0.0, True),
    ("radar_just_below", -1.51, 0.0, 0.0, True),
    ("radar_at_threshold", -1.5, 0.0, 0.0, False),
    ("steady", 0.0, 0.9, 0.0, False),
    ("model_brake", 0.0, 0.9, -2.0, True),
    ("model_not_sure_of_lead", 0.0, 0.5, -3.0, False),
    ("model_barely_sure", 0.0, 0.51, -3.0, True),
    ("model_gentle", 0.0, 0.9, -1.0, False),
    ("radar_wins_when_model_unsure", -2.0, 0.1, 0.0, True),
  ])
  def test_lead_braking(self, _, a_lead_k, prob, a0, expected):
    assert lead_braking(a_lead_k, prob, a0) == expected


class TestWakePoints(OpenpilotTestCase):
  T = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

  def test_model_points_remove_the_camera_offset_and_keep_y_as_is(self):
    pts = model_wake_points(self.T, [51.52, 47.52, 35.52, 15.52, 5.0, 2.0], [1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    assert [p.t for p in pts] == [2.0, 4.0, 6.0]
    assert [round(p.x, 2) for p in pts] == [46.0, 34.0, 14.0]
    # leadsV3 y is right positive (radard.py: yRel = -lead.y[0]), the same convention the projector wants
    assert [round(p.y, 2) for p in pts] == [1.1, 1.2, 1.3]

  def test_model_y_agrees_with_radar_y_rel(self):
    # a lead 2 m to the right: radarState yRel = -2.0 (left positive), leadsV3 y = +2.0
    from_model = model_wake_points(self.T, [50.0] * 6, [2.0] * 6)
    from_radar = kinematic_wake_points(50.0, -2.0, 20.0, 20.0, 0.0)
    assert [p.y for p in from_model] == [p.y for p in from_radar] == [2.0, 2.0, 2.0]

  def test_model_points_interpolate_between_samples(self):
    pts = model_wake_points(self.T, [51.52, 47.52, 35.52, 15.52, 5.0, 2.0], [0.0] * 6, times=(3.0,))
    assert math.isclose(pts[0].x, 41.52 - 1.52)

  def test_malformed_model_lead_gives_no_points(self):
    assert model_wake_points([], [], []) == []
    assert model_wake_points(self.T, [1.0, 2.0], [1.0, 2.0]) == []

  def test_kinematic_points_brake_to_a_stop(self):
    pts = kinematic_wake_points(60.0, 0.0, 20.0, 20.0, -2.0)
    # relative: lead travels 20t - t² and we travel 20t, so x = 60 - t² until the lead stops at t = 10
    assert [round(p.x, 2) for p in pts] == [56.0, 44.0, 24.0]
    stopped = kinematic_wake_points(60.0, 0.0, 5.0, 4.0, -2.0, times=(6.0,))[0]
    assert math.isclose(stopped.x, 60.0 + 4.0 - 5.0 * 6.0)  # lead stopped after 2 s having gone 4 m

  def test_kinematic_y_flips_the_radar_sign(self):
    assert kinematic_wake_points(50.0, 1.5, 20.0, 20.0, -2.0)[0].y == -1.5

  def test_points_under_the_hood_are_dropped(self):
    pts = [WakePoint(2.0, 30.0, 0.0), WakePoint(4.0, 3.0, 0.0), WakePoint(6.0, -9.0, 0.0)]
    assert visible_wake_points(pts) == pts[:1]


class TestHoldTimer(OpenpilotTestCase):
  def test_idle_until_active(self):
    assert not HoldTimer(HOLD_TIME, DT).update(False)

  def test_holds_for_the_hold_time_after_clearing(self):
    t = HoldTimer(HOLD_TIME, DT)
    assert t.update(True)
    held = [t.update(False) for _ in range(20)]
    assert held[:16] == [True] * 16 and held[16:] == [False] * 4  # 0.8 s at 20 fps is 16 frames

  def test_active_again_restarts_the_hold(self):
    t = HoldTimer(HOLD_TIME, DT)
    t.update(True)
    for _ in range(10):
      t.update(False)
    assert t.update(True)
    assert all(t.update(False) for _ in range(16))
    assert not t.update(False)

  def test_reset(self):
    t = HoldTimer(HOLD_TIME, DT)
    t.update(True)
    t.reset()
    assert not t.update(False)


class TestLeadWakeUpdate(OpenpilotTestCase):
  X = [51.52, 47.52, 35.52, 15.52, 5.0, 2.0]  # a lead 50 m ahead braking at 2 m/s² (from the camera)

  def _run(self, wake, frames, braking=True, present=True, **kw):
    a = -2.0 if braking else 0.0
    for _ in range(frames):
      feed(radarState=radar_msg(present=present, a_lead=a), carState=car_msg(),
           modelV2=model_msg(lead_x=self.X, lead_a=a, lead_y=kw.get("lead_y", 0.0), lead_prob=kw.get("prob", 0.9)))
      wake.update()

  def test_braking_lead_brings_the_wake_in_with_a_fade(self):
    with overlay_env(lead_braking_wake=True):
      wake = LeadWake()
      self._run(wake, 2)
      assert 0.0 < wake._alpha.x < 0.9
      self._run(wake, 30)
      assert wake._alpha.x > 0.95
      assert [p.t for p in wake._points] == [2.0, 4.0, 6.0]
      assert [round(p.x, 1) for p in wake._points] == [46.0, 34.0, 14.0]

  def test_lead_y_is_not_flipped(self):
    with overlay_env(lead_braking_wake=True):
      wake = LeadWake()
      self._run(wake, 3, lead_y=1.7)
      assert all(math.isclose(p.y, 1.7, abs_tol=1e-5) for p in wake._points)

  def test_wake_holds_then_fades_after_braking_stops(self):
    with overlay_env(lead_braking_wake=True):
      wake = LeadWake()
      self._run(wake, 30)
      self._run(wake, 16, braking=False)
      assert wake._alpha.x > 0.95  # still held on the 16th frame (0.8 s)
      self._run(wake, 1, braking=False)
      alpha_after_hold = wake._alpha.x
      assert alpha_after_hold > 0.5  # starts fading, doesn't snap off
      self._run(wake, 40, braking=False)
      assert wake._alpha.x < 0.01

  def test_steady_lead_shows_nothing(self):
    with overlay_env(lead_braking_wake=True):
      wake = LeadWake()
      self._run(wake, 30, braking=False)
      assert wake._alpha.x < 0.01 and not wake._points

  def test_no_lead_no_wake_whatever_the_model_says(self):
    with overlay_env(lead_braking_wake=True):
      wake = LeadWake()
      self._run(wake, 30, present=False)
      assert wake._alpha.x < 0.01

  def test_radar_brake_with_an_unsure_model_falls_back_to_radar_kinematics(self):
    with overlay_env(lead_braking_wake=True):
      wake = LeadWake()
      for _ in range(3):
        feed(radarState=radar_msg(d_rel=60.0, v_lead=20.0, a_lead=-2.0, y_rel=-1.0), carState=car_msg(20.0),
             modelV2=model_msg(lead_x=self.X, lead_prob=0.1))
        wake.update()
      assert [round(p.x, 1) for p in wake._points] == [56.0, 44.0, 24.0]
      assert all(p.y == 1.0 for p in wake._points)

  def test_disabled_flag_resets(self):
    with overlay_env(lead_braking_wake=True):
      wake = LeadWake()
      self._run(wake, 30)
    with overlay_env(lead_braking_wake=False):
      wake.update()
      assert wake._alpha.x == 0.0 and not wake._points
      wake.render(mock.MagicMock(), mock.MagicMock())
