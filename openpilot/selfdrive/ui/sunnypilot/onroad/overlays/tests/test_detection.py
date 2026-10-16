"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from unittest import mock

import numpy as np
import pyray as rl

from openpilot.cereal import custom, messaging
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.onroad.hud_renderer import UI_CONFIG
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_scope import scope_rect
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.detection import CONFIRM_FRAMES, GO_CONFIRM, LEAD_LOST_TIME, RELEASE_TIME, \
  SHOW_CLEARED, SHOW_GO, DetectionEvent, DetectionFeed, EventDetector, EventKind, StopDetector, StopPhase, predicted_stop_distance, \
  stop_is_for_car
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.disengage_meter import PANEL_WIDTH as RISK_WIDTH
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.stop_indicator import TURN_SIGNAL_CLEARANCE, age_text, column, clock, event_detail, \
  planner_follows_model

DT = 0.05
T = np.linspace(0, 10, 33)


def plan_with_stop(v_ego: float, stop_at: float | None):
  """Model position/velocity: constant decel to rest at stop_at meters, or cruising at v_ego."""
  if stop_at is None:
    return (v_ego * T).tolist(), np.full(33, v_ego).tolist()
  decel = v_ego ** 2 / (2 * stop_at)
  t_stop = v_ego / decel
  v = np.clip(v_ego - decel * T, 0, None)
  x = np.where(T < t_stop, v_ego * T - 0.5 * decel * T ** 2, stop_at)
  return x.tolist(), v.tolist()


class Clock:
  def __init__(self):
    self.t = 100.0

  def tick(self):
    self.t += DT
    return 1_700_000_000 + self.t, self.t


class TestPredictedStop(OpenpilotTestCase):
  def test_cruising_has_no_stop(self):
    assert predicted_stop_distance(*plan_with_stop(20.0, None)) is None

  @parameterized.expand([(40.0,), (80.0,), (95.0,)])
  def test_finds_the_planned_stop(self, stop_at):
    x = predicted_stop_distance(*plan_with_stop(20.0, stop_at))
    assert x is not None and abs(x - stop_at) < 0.08 * stop_at

  def test_only_stops_inside_the_10_s_plan(self):
    # braking gently from 20 m/s for 140 m takes 14 s, so it isn't in the model's plan yet
    assert predicted_stop_distance(*plan_with_stop(20.0, 140.0)) is None

  def test_already_stopped(self):
    assert predicted_stop_distance([0.0, 0.1, 0.2], [0.0, 0.0, 0.0]) == 0.0

  def test_car_is_the_reason_only_if_it_is_at_the_stop(self):
    assert stop_is_for_car(80.0, True, 72.0)
    assert not stop_is_for_car(80.0, True, 120.0)
    assert not stop_is_for_car(80.0, False, 0.0)


class TestStopDetector(OpenpilotTestCase):
  def setUp(self):
    self.clock = Clock()
    self.det = StopDetector()

  def step(self, v_ego=20.0, stop_at=None, lead=False, lead_d=0.0, plan=None):
    wall, mono = self.clock.tick()
    x, v = plan if plan is not None else plan_with_stop(max(v_ego, 0.1), stop_at)
    return self.det.update(wall, mono, DT, x, v, v_ego, lead, lead_d)

  def approach(self, stop_at=90.0, **kw):
    events = []
    for _ in range(CONFIRM_FRAMES):
      events += self.step(stop_at=stop_at, **kw)
    return events

  def test_stop_is_stamped_at_the_first_frame_it_was_seen(self):
    first_wall, first_mono = self.clock.t + 1_700_000_000 + DT, self.clock.t + DT
    events = self.approach(stop_at=90.0)
    assert [e.kind for e in events] == [EventKind.STOP]
    stop = events[0]
    assert abs(stop.mono_time - first_mono) < 1e-6 and abs(stop.wall_time - first_wall) < 1e-6
    assert abs(stop.distance - 90.0) < 8 and stop.car_ahead is False
    assert self.det.phase == StopPhase.APPROACH

  def test_stop_behind_a_car(self):
    events = self.approach(stop_at=60.0, lead=True, lead_d=55.0)
    assert events[0].car_ahead is True

  def test_flicker_never_confirms(self):
    for i in range(40):
      assert self.step(stop_at=90.0 if i % 2 else None) == []
    assert self.det.phase == StopPhase.IDLE

  def test_creeping_is_not_an_approach(self):
    for _ in range(10):
      assert self.step(v_ego=1.0, stop_at=5.0) == []

  def test_cleared_when_the_model_drops_the_stop(self):
    self.approach()
    events = []
    for _ in range(int(RELEASE_TIME / DT) + 1):
      events += self.step(stop_at=None)
    assert [e.kind for e in events] == [EventKind.CLEARED]
    for _ in range(int(SHOW_CLEARED / DT) + 2):
      self.step(stop_at=None)
    assert self.det.phase == StopPhase.IDLE

  def test_stop_then_go(self):
    self.approach()
    self.step(v_ego=0.0, plan=([0.0] * 33, [0.0] * 33))
    assert self.det.phase == StopPhase.STOPPED
    held = ([0.0] * 32 + [10.0], [0.0] * 33)
    for _ in range(20):
      assert self.step(v_ego=0.0, plan=held) == []
    go_plan = plan_with_stop(8.0, None)
    events = []
    for _ in range(int(GO_CONFIRM / DT) + 1):
      events += self.step(v_ego=0.0, plan=go_plan)
    assert [e.kind for e in events] == [EventKind.GO] and events[0].car_ahead is False
    for _ in range(int(SHOW_GO / DT) + 2):
      self.step(v_ego=5.0, plan=go_plan)
    assert self.det.phase == StopPhase.IDLE

  def test_stopped_without_seeing_the_approach(self):
    self.step(v_ego=0.0, plan=([0.0] * 33, [0.0] * 33))
    assert self.det.phase == StopPhase.STOPPED and self.det.onset is None

  def test_new_stop_right_after_go(self):
    self.approach()
    self.step(v_ego=0.0, plan=([0.0] * 33, [0.0] * 33))
    for _ in range(int(GO_CONFIRM / DT) + 1):
      self.step(v_ego=0.0, plan=plan_with_stop(8.0, None))
    assert self.det.phase == StopPhase.GO
    assert [e.kind for e in self.approach(stop_at=70.0)] == [EventKind.STOP]


class TestEventDetector(OpenpilotTestCase):
  def setUp(self):
    self.clock = Clock()
    self.det = EventDetector()

  def step(self, lead=False, lead_d=50.0, lat=None, hard_brake=False, green=False, depart=False):
    wall, mono = self.clock.tick()
    lat = [0.1] * 33 if lat is None else lat
    return self.det.update(wall, mono, lead, lead_d, True, lat, (20.0 * T).tolist(), hard_brake, green, depart)

  def kinds(self, events):
    return [e.kind for e in events]

  def test_lead_acquired_after_it_holds(self):
    assert self.step(lead=True) == []
    events = []
    for _ in range(10):
      events += self.step(lead=True)
    assert self.kinds(events) == [EventKind.LEAD]

  def test_lead_lost_after_a_second(self):
    for _ in range(10):
      self.step(lead=True)
    events = []
    for _ in range(int(LEAD_LOST_TIME / DT) + 1):
      events += self.step(lead=False)
    assert self.kinds(events) == [EventKind.LEAD_LOST]

  def test_cut_in(self):
    for _ in range(10):
      self.step(lead=True, lead_d=50.0)
    events = self.step(lead=True, lead_d=18.0)
    assert self.kinds(events) == [EventKind.CUT_IN] and events[0].distance == 18.0

  def test_curve_rising_edge_with_hysteresis(self):
    bend = [0.1] * 10 + [2.2] * 5 + [0.1] * 18
    assert self.kinds(self.step(lat=bend)) == [EventKind.CURVE]
    assert self.step(lat=[1.4] * 33) == []           # still above the off threshold
    assert self.step(lat=[0.5] * 33) == []
    assert self.kinds(self.step(lat=bend)) == [EventKind.CURVE]

  def test_alerts_fire_once_per_edge(self):
    assert self.kinds(self.step(hard_brake=True, green=True, depart=True)) == \
      [EventKind.HARD_BRAKE, EventKind.GREEN_LIGHT, EventKind.LEAD_DEPART]
    assert self.step(hard_brake=True, green=True, depart=True) == []


class FakeSM:
  def __init__(self, msgs):
    self.frame = 0
    self.recv_frame = {}
    self.valid = {}
    self.data = {}
    self.feed(msgs)

  def feed(self, msgs, model=True):
    self.frame += 1
    for name, msg in msgs.items():
      if name == 'modelV2' and not model:
        continue
      self.recv_frame[name] = self.frame
      self.valid[name] = True
      self.data[name] = getattr(msg, name)

  def __getitem__(self, name):
    return self.data[name]


def scene(v_ego=20.0, stop_at=None):
  msgs = {name: messaging.new_message(name) for name in ('modelV2', 'radarState', 'carState', 'longitudinalPlanSP')}
  x, v = plan_with_stop(v_ego, stop_at)
  msgs['modelV2'].modelV2.position.x = x
  msgs['modelV2'].modelV2.velocity.x = v
  msgs['modelV2'].modelV2.orientationRate.z = [0.0] * 33
  msgs['carState'].carState.vEgo = v_ego
  return msgs


class TestDetectionFeed(OpenpilotTestCase):
  def test_advances_once_per_model_frame(self):
    msgs = scene(stop_at=90.0)
    sm = FakeSM(msgs)
    feed, clock_ = DetectionFeed(), Clock()
    for _ in range(CONFIRM_FRAMES - 1):
      feed.update(sm, 0, *clock_.tick())
      for _ in range(3):  # several UI frames per model frame shouldn't advance the detector
        feed.update(sm, 0, *clock_.tick())
      sm.feed(msgs)
    assert len(feed.log) == 0
    feed.update(sm, 0, *clock_.tick())
    assert [e.kind for e in feed.log] == [EventKind.STOP]

  def test_ignores_messages_from_before_going_onroad(self):
    sm = FakeSM(scene(stop_at=90.0))
    feed = DetectionFeed()
    for _ in range(5):
      feed.update(sm, 1000, *Clock().tick())
    assert feed.stop.phase == StopPhase.IDLE


class TestStopIndicatorHelpers(OpenpilotTestCase):
  def test_event_detail(self):
    stop = DetectionEvent(EventKind.STOP, 0.0, 0.0, distance=42.67, car_ahead=False)
    assert event_detail(stop, is_metric=False) == "140 ft • no car"
    lead = DetectionEvent(EventKind.LEAD, 0.0, 0.0, distance=62.0, radar=True)
    assert event_detail(lead, is_metric=True) == "62 m • radar"
    curve = DetectionEvent(EventKind.CURVE, 0.0, 0.0, distance=85.0, value=2.06)
    assert event_detail(curve, is_metric=True) == "0.21 g • 85 m"
    assert event_detail(DetectionEvent(EventKind.HARD_BRAKE, 0.0, 0.0), is_metric=True) == ""

  @parameterized.expand([(m, scope, fuel) for m in (False, True) for scope in (False, True) for fuel in (False, True)])
  def test_column_sits_between_the_speed_limit_sign_and_the_left_turn_signal(self, is_metric, radar_scope, rocket_fuel):
    hud = rl.Rectangle(0, 0, 2160, 1080)
    with mock.patch.object(ui_state, "is_metric", is_metric), mock.patch.object(ui_state, "radar_scope", radar_scope), \
         mock.patch.object(ui_state, "rocket_fuel", rocket_fuel):
      x, w = column(hud)
    sign_w = UI_CONFIG.set_speed_width_metric if is_metric else UI_CONFIG.set_speed_width_imperial
    sign_right = 60 + 30 - 6 + 2 * sign_w + (9 if is_metric else 0)
    assert x >= sign_right + 20  # clear of the speed limit sign (and the round metric sign's overhang)
    center = hud.width / 2
    assert x + w <= center - (80 + 150) - 10  # clear of the left turn signal
    assert x + w <= center - RISK_WIDTH / 2 - 10  # and of the risk meter centered under the speed
    if radar_scope:
      scope = scope_rect(hud, rocket_fuel)
      assert x >= scope.x + scope.width + 20
    assert w >= 260
    assert TURN_SIGNAL_CLEARANCE >= 80 + 150

  @parameterized.expand([(0.0, "<1m"), (59.9, "<1m"), (60.0, "1m"), (59 * 60 + 59, "59m"), (3600.0, "1h+"), (5 * 3600.0, "1h+")])
  def test_age_text_is_in_whole_minutes(self, seconds, expected):
    assert age_text(seconds) == expected

  def test_clock_has_tenths(self):
    text = clock(1_700_000_000.37)
    assert len(text) == 10 and text[2] == ":" and text.endswith(".3")

  def test_planner_follows_model(self):
    msgs = {name: messaging.new_message(name) for name in ('selfdriveState', 'longitudinalPlanSP')}
    sm = {name: getattr(msg, name) for name, msg in msgs.items()}
    assert not planner_follows_model(sm)
    msgs['selfdriveState'].selfdriveState.experimentalMode = True
    assert planner_follows_model(sm)
    dec = msgs['longitudinalPlanSP'].longitudinalPlanSP.dec
    dec.active = True
    dec.state = custom.LongitudinalPlanSP.DynamicExperimentalControl.DynamicExperimentalControlState.acc
    assert not planner_follows_model(sm)
    dec.state = custom.LongitudinalPlanSP.DynamicExperimentalControl.DynamicExperimentalControlState.blended
    assert planner_follows_model(sm)
