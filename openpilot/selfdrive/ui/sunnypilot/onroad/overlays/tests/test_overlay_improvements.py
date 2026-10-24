"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from unittest import mock

import pyray as rl

from openpilot.cereal import log, messaging
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays import detection, hard_brake_flash, pace_border as pb, stop_line
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.curve_hologram import CurveAhead, CurveHologram, curve_decel_needed
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.driver_flags import is_distracted
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_scope import RadarScope
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.tests import test_ar_overlays, test_pace_border
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.tests.test_ar_overlays import bell, car_msg, model_msg, overlay_env


def sync_clocks() -> None:
  """Both helpers feed the one SubMaster, whose frequency tracker needs time to only move forward."""
  latest = max(test_ar_overlays._clock[0], test_pace_border._clock[0])
  test_ar_overlays._clock[0] = test_pace_border._clock[0] = latest


def feed(**kw):
  sync_clocks()
  test_pace_border.feed(**kw)


def feed_msgs(**msgs):
  sync_clocks()
  test_ar_overlays.feed(**msgs)

RECT = rl.Rectangle(0, 0, 2100, 1020)


class TestPaceBorderCurves(OpenpilotTestCase):
  def setup_method(self):
    detection._feed = None

  def test_a_sharp_curve_ahead_lights_the_border(self):
    curve = CurveAhead(x=80.0, speed=15.0, right=True)
    assert curve_decel_needed(30.0, 15.0, 80.0) > 4.0
    with overlay_env(pace_border=True, started=True, started_frame=0), mock.patch.object(pb, "curve_ahead", return_value=curve):
      border = pb.PaceBorder()
      for _ in range(40):
        feed(v_ego=30.0)
        border.update()
      assert border.reason == "curve" and border.level > 0.9 and border.papi_shown and border.reds == 4

  def test_a_gentle_curve_does_nothing(self):
    with overlay_env(pace_border=True, started=True, started_frame=0), \
         mock.patch.object(pb, "curve_ahead", return_value=CurveAhead(x=80.0, speed=40.0, right=True)):
      border = pb.PaceBorder()
      for _ in range(40):
        feed(v_ego=30.0)
        border.update()
      assert border.level == 0.0 and not border.papi_shown


class TestOneEdgeGlow(OpenpilotTestCase):
  def test_the_hard_brake_warning_has_the_edge_to_itself(self):
    with overlay_env(pace_border=True, started=True, started_frame=0), mock.patch.object(pb, "draw_edge_glow") as glow, \
         mock.patch.object(pb, "draw_papi"):
      border = pb.PaceBorder()
      border._urgency.x = 1.0
      with mock.patch.object(hard_brake_flash, "_level", 1.0):
        border.render(RECT)
      assert glow.call_count == 0
      with mock.patch.object(hard_brake_flash, "_level", 0.0):
        border.render(RECT)
      assert glow.call_count == 1


def dm_state(distracted=False, pose=False, eye=False, phone=False, policy=log.DriverMonitoringState.MonitoringPolicy.vision):
  msg = messaging.new_message("driverMonitoringState")
  dm = msg.driverMonitoringState
  dm.activePolicy = policy
  vision = dm.visionPolicyState
  vision.isDistracted = distracted
  vision.distractedTypes.pose, vision.distractedTypes.eye, vision.distractedTypes.phone = pose, eye, phone
  return messaging.log_from_bytes(msg.to_bytes()).driverMonitoringState


class TestDistracted(OpenpilotTestCase):
  def test_follows_the_policy(self):
    assert is_distracted(dm_state(True, pose=True)) and is_distracted(dm_state(True, eye=True))
    assert not is_distracted(dm_state(False, pose=True))
    assert not is_distracted(dm_state(True, phone=True))   # the phone chip covers that
    assert not is_distracted(dm_state(True, pose=True, policy=log.DriverMonitoringState.MonitoringPolicy.wheeltouch))


class TestBrakePoint(OpenpilotTestCase):
  def setup_method(self):
    detection._feed = None

  def test_math(self):
    assert stop_line.brake_point(120.0, 15.0) == 120.0 - stop_line.BRAKE_MARGIN - 15.0 ** 2 / (2 * stop_line.BRAKE_DECEL)
    assert stop_line.brake_point(40.0, 25.0) < 0   # too late for gentle

  def run_line(self, a_ego: float):
    with overlay_env(stop_detection=True, started=True, started_frame=0):
      line = stop_line.StopLine()
      for _ in range(30):
        feed(v_ego=15.0, a_ego=a_ego, stop_at=70.0)  # the stop has to be inside the model's 10 s
        line.update()
      return line

  def test_shown_until_the_car_brakes(self):
    line = self.run_line(0.0)
    assert line._brake_alpha.x > 0.9 and 5.0 < line._brake_x < 15.0
    assert self.run_line(-1.5)._brake_alpha.x < 0.01


class TestCurveTooFast(OpenpilotTestCase):
  def run_curve(self, v_ego: float) -> CurveHologram:
    with overlay_env(curve_hologram=True, started=True, started_frame=0):
      holo = CurveHologram()
      for _ in range(10):
        feed_msgs(model=model_msg(bell(2.4, 60.0, 15.0)), car=car_msg(v_ego), sp=messaging.new_message('longitudinalPlanSP'))
        holo.update()
      return holo

  def test_flags_only_when_well_over_the_advisory(self):
    assert self.run_curve(35.0)._too_fast
    assert not self.run_curve(24.0)._too_fast


class TestScopeBlindSpots(OpenpilotTestCase):
  def test_reads_the_blind_spot_monitor(self):
    with overlay_env(radar_scope=True, started=True, started_frame=0):
      scope = RadarScope()
      for _ in range(20):
        cs = car_msg(25.0)
        cs.carState.rightBlindspot = True
        feed_msgs(car=cs)
        scope.update()
      assert scope._blindspots[1].x > 0.9 and scope._blindspots[0].x < 0.01
