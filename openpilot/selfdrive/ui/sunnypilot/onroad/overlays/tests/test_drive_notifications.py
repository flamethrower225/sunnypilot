"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from dataclasses import replace

from openpilot.common.constants import CV
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays import drive_notifications as dn
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.drive_notifications import BREAK_AFTER, DriveSnapshot, Icon, NotificationEngine, \
  NotifyKind, adjacent_lanes, curve_decel_needed
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.tests.test_ar_overlays import next_sm_time

MPH = CV.MPH_TO_MS
DT = 0.05

# 60 mph in a 65 behind a lead doing 45, with a free lane on the left
HIGHWAY = DriveSnapshot(now=0.0, v_ego=50 * MPH, speed_limit=65 * MPH, cruise_speed=70 * MPH, lead_present=True, lead_d=40.0,
                        lead_v=45 * MPH, left_lane=True, right_lane=True)


def run(engine: NotificationEngine, snap: DriveSnapshot, seconds: float, start: float = 0.0):
  """Feeds the same scene for a while, returns every notification that went on screen."""
  shown = []
  t = start
  while t < start + seconds - 1e-9:
    note = engine.update(replace(snap, now=t))
    if note is not None:
      shown.append(note)
    t += DT
  return shown, t


def engine_with(*kinds: NotifyKind) -> NotificationEngine:
  return NotificationEngine([r for r in dn.default_rules() if r.kind in kinds])


class TestAdjacentLanes(OpenpilotTestCase):
  def test_both_sides(self):
    assert adjacent_lanes([-5.4, -1.8, 1.8, 5.4], [0.8, 0.9, 0.9, 0.8], [-9.0, 9.0]) == (True, True)

  def test_an_unseen_outer_line_is_no_lane(self):
    assert adjacent_lanes([-5.4, -1.8, 1.8, 5.4], [0.1, 0.9, 0.9, 0.8], [-9.0, 9.0]) == (False, True)

  def test_a_shoulder_line_is_no_lane(self):
    # the road edge sits right past our left line, so the "lane" beyond it is the shoulder
    assert adjacent_lanes([-5.4, -1.8, 1.8, 5.4], [0.8, 0.9, 0.9, 0.8], [-3.0, 9.0]) == (False, True)

  def test_lane_width_has_to_be_a_lane(self):
    assert adjacent_lanes([-2.5, -1.8, 1.8, 9.0], [0.8, 0.9, 0.9, 0.8], []) == (False, False)
    assert adjacent_lanes([], [], []) == (False, False)


class TestSlowLead(OpenpilotTestCase):
  def test_suggests_passing_left_after_a_while(self):
    engine = engine_with(NotifyKind.SLOW_LEAD)
    shown, _ = run(engine, HIGHWAY, dn.SLOW_SUSTAIN - 0.5)
    assert shown == []
    shown, _ = run(engine, HIGHWAY, 1.0, start=dn.SLOW_SUSTAIN - 0.5)
    assert len(shown) == 1 and shown[0].icon == Icon.LEFT and "20 mph" in shown[0].detail

  def test_blind_spot_and_lanes_pick_the_advice(self):
    def advice(**kw):
      shown, _ = run(engine_with(NotifyKind.SLOW_LEAD), replace(HIGHWAY, **kw), dn.SLOW_SUSTAIN + 0.5)
      return shown[0].icon if shown else None
    assert advice(left_blindspot=True) == Icon.WAIT
    assert advice(left_lane=False) == Icon.RIGHT
    assert advice(left_lane=False, right_blindspot=True) is None
    assert advice(left_lane=False, right_lane=False) is None

  def test_nothing_while_changing_lanes_or_without_a_reason(self):
    for kw in ({"changing_lanes": True}, {"lead_v": 62 * MPH}, {"lead_d": 150.0}, {"v_ego": 20 * MPH, "lead_v": 15 * MPH},
               {"speed_limit": 0.0, "cruise_speed": 0.0}):
      shown, _ = run(engine_with(NotifyKind.SLOW_LEAD), replace(HIGHWAY, **kw), dn.SLOW_SUSTAIN + 1)
      assert shown == [], kw

  def test_set_speed_stands_in_for_a_missing_limit(self):
    shown, _ = run(engine_with(NotifyKind.SLOW_LEAD), replace(HIGHWAY, speed_limit=0.0), dn.SLOW_SUSTAIN + 1)
    assert len(shown) == 1 and "set speed" in shown[0].detail

  def test_cooldown_unless_the_lead_changes(self):
    engine = engine_with(NotifyKind.SLOW_LEAD)
    shown, t = run(engine, HIGHWAY, dn.SLOW_SUSTAIN + dn.SlowLeadRule.ttl + 5)
    assert len(shown) == 1 and engine.current is None
    shown, t = run(engine, HIGHWAY, 20.0, start=t)
    assert shown == []
    # a different, slower car pulls in ahead: worth saying again, after it has held us up a while too
    shown, _ = run(engine, replace(HIGHWAY, lead_d=20.0), dn.SLOW_SUSTAIN + 1, start=t)
    assert len(shown) == 1


class TestOtherRules(OpenpilotTestCase):
  def test_stopped_traffic_is_sticky_and_outranks_slow_lead(self):
    engine = engine_with(NotifyKind.SLOW_LEAD, NotifyKind.STOPPED_TRAFFIC)
    shown, t = run(engine, HIGHWAY, dn.SLOW_SUSTAIN + 1)
    assert [n.kind for n in shown] == [NotifyKind.SLOW_LEAD]
    stopped = replace(HIGHWAY, lead_v=0.0, lead_d=120.0)
    shown, t = run(engine, stopped, 1.0, start=t)
    assert [n.kind for n in shown] == [NotifyKind.STOPPED_TRAFFIC]
    # once the car ahead moves on, the card goes after its minimum time
    shown, t = run(engine, replace(HIGHWAY, lead_present=False), dn.CARD_TTL_MIN + 0.5, start=t)
    assert engine.current is None

  def test_sharp_curve(self):
    curve = replace(HIGHWAY, lead_present=False, v_ego=30.0, apex_x=80.0, apex_speed=18.0)
    assert abs(curve_decel_needed(30.0, 18.0, 80.0) - (30.0 ** 2 - 18.0 ** 2) / (2 * 70.0)) < 1e-9
    shown, _ = run(engine_with(NotifyKind.SHARP_CURVE), curve, 1.0)
    assert len(shown) == 1 and "40 mph" in shown[0].text
    # already braking hard enough, or the bend is gentle, or too far away
    for kw in ({"a_ego": -6.0}, {"apex_speed": 28.0}, {"apex_x": 400.0}, {"apex_x": None}):
      shown, _ = run(engine_with(NotifyKind.SHARP_CURVE), replace(curve, **kw), 1.0)
      assert shown == [], kw

  def test_speed_limit_change_is_reported_once_settled(self):
    engine = engine_with(NotifyKind.SPEED_LIMIT)
    shown, t = run(engine, HIGHWAY, 3.0)
    assert len(shown) == 1 and shown[0].text == "65 mph"
    shown, t = run(engine, HIGHWAY, 10.0, start=t)
    assert shown == []
    shown, t = run(engine, replace(HIGHWAY, speed_limit=45 * MPH), 0.5, start=t)
    assert shown == []  # not settled yet
    shown, t = run(engine, replace(HIGHWAY, speed_limit=45 * MPH), 2.0, start=t)
    assert len(shown) == 1 and shown[0].icon == Icon.DOWN and "65" in shown[0].detail

  def test_over_the_limit_only_when_speed_limit_doesnt_warn(self):
    fast = replace(HIGHWAY, lead_present=False, v_ego=80 * MPH)
    shown, _ = run(engine_with(NotifyKind.OVER_LIMIT), fast, dn.OVER_SUSTAIN + 1)
    assert len(shown) == 1 and "15 mph" in shown[0].text
    shown, _ = run(engine_with(NotifyKind.OVER_LIMIT), replace(fast, speed_limit_warns=True), dn.OVER_SUSTAIN + 1)
    assert shown == []
    shown, _ = run(engine_with(NotifyKind.OVER_LIMIT), replace(fast, v_ego=69 * MPH), dn.OVER_SUSTAIN + 1)
    assert shown == []

  def test_break_every_two_hours(self):
    engine = engine_with(NotifyKind.BREAK)
    shown, _ = run(engine, replace(HIGHWAY, drive_time=BREAK_AFTER - 10), 1.0)
    assert shown == []
    shown, _ = run(engine, replace(HIGHWAY, drive_time=BREAK_AFTER + 1), 1.0, start=100.0)
    assert len(shown) == 1 and "2 h" in shown[0].detail
    shown, _ = run(engine, replace(HIGHWAY, drive_time=BREAK_AFTER + 600), 20.0, start=200.0)
    assert shown == []
    shown, _ = run(engine, replace(HIGHWAY, drive_time=2 * BREAK_AFTER + 1), 1.0, start=8000.0)
    assert len(shown) == 1

  def test_a_lesser_notice_waits_its_turn(self):
    engine = engine_with(NotifyKind.SLOW_LEAD, NotifyKind.SPEED_LIMIT)
    shown, t = run(engine, HIGHWAY, dn.SLOW_SUSTAIN + 1)
    assert [n.kind for n in shown] == [NotifyKind.SPEED_LIMIT, NotifyKind.SLOW_LEAD]  # the slow lead takes over
    # a new limit while the slow lead card is up waits for it to finish
    shown, t = run(engine, replace(HIGHWAY, speed_limit=55 * MPH), 3.0, start=t)
    assert shown == []
    shown, t = run(engine, replace(HIGHWAY, speed_limit=55 * MPH), dn.SlowLeadRule.ttl, start=t)
    assert [n.kind for n in shown] == [NotifyKind.SPEED_LIMIT]


# ---- the overlay, fed real messages ----

def highway_msgs(left_blindspot=False, blinker=False):
  import numpy as np
  from openpilot.cereal import messaging
  msgs = {n: messaging.new_message(n) for n in ('carState', 'radarState', 'modelV2', 'longitudinalPlanSP')}
  cs = msgs['carState'].carState
  cs.vEgo, cs.leftBlindspot, cs.leftBlinker = 50 * MPH, left_blindspot, blinker
  lead = msgs['radarState'].radarState.leadOne
  lead.present, lead.dRel, lead.vLeadK = True, 40.0, 45 * MPH
  mv = msgs['modelV2'].modelV2
  xs = (50 * MPH * np.linspace(0, 10, 33)).tolist()
  mv.position.x, mv.position.y, mv.velocity.x, mv.orientationRate.z = xs, [0.0] * 33, [50 * MPH] * 33, [0.0] * 33
  lines = mv.init('laneLines', 4)
  for line, y in zip(lines, (-5.4, -1.8, 1.8, 5.4), strict=True):
    line.x, line.y = xs, [y] * 33
  mv.laneLineProbs = [0.8, 0.9, 0.9, 0.8]
  edges = mv.init('roadEdges', 2)
  for edge, y in zip(edges, (-9.0, 9.0), strict=True):
    edge.x, edge.y = xs, [y] * 33
  r = msgs['longitudinalPlanSP'].longitudinalPlanSP.speedLimit.resolver
  r.speedLimit, r.speedLimitValid = 65 * MPH, True
  for m in msgs.values():
    m.valid = True
  return list(msgs.values())


_sm_clock = [0.0]


class TestOverlay(OpenpilotTestCase):
  def drive(self, seconds, **kw):
    from unittest import mock
    from openpilot.selfdrive.ui.sunnypilot.onroad.overlays import detection
    from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.tests.test_ar_overlays import overlay_env
    from openpilot.selfdrive.ui.ui_state import ui_state
    detection._feed = None
    with overlay_env(drive_notifications=True, started=True, started_frame=0, started_time=0.0, speed_limit_mode=0), \
         mock.patch.object(dn.time, "monotonic") as mono:
      mono.return_value = 1000.0
      overlay = dn.DriveNotifications()
      for i in range(int(seconds / DT)):
        mono.return_value = 1000.0 + i * DT
        _sm_clock[0] = next_sm_time(DT)
        ui_state.sm.update_msgs(_sm_clock[0], highway_msgs(**kw))
        overlay.update()
      return overlay, detection.get_detection_feed().log

  def test_slow_lead_card_and_log(self):
    overlay, log = self.drive(dn.SLOW_SUSTAIN + 1)
    assert overlay.engine.current is not None and overlay.engine.current.note.icon == Icon.LEFT
    assert [e.kind for e in log][:2] == [dn.EventKind.SLOW_LEAD, dn.EventKind.SPEED_LIMIT]
    assert log[0].note == "Pass on the left"

  def test_blind_spot_and_blinker_come_from_car_state(self):
    overlay, _ = self.drive(dn.SLOW_SUSTAIN + 1, left_blindspot=True)
    assert overlay.engine.current.note.icon == Icon.WAIT
    overlay, _ = self.drive(dn.SLOW_SUSTAIN + 1, blinker=True)
    assert overlay.engine.current is None or overlay.engine.current.note.kind != NotifyKind.SLOW_LEAD
