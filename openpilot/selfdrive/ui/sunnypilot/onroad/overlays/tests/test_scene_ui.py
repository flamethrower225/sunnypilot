"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Camera Object Detection (scened) in the HUD: the traffic light and the camera's cars.
"""
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from openpilot.cereal import messaging
from openpilot.common.constants import CV
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays import detection, drive_notifications as dn, pace_border as pb
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.light_view import LightView, current_light
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_tracks import CAMERA_ID_BASE, RadarFeed, camera_tracks
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.tests.test_ar_overlays import overlay_env
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.tests.test_drive_notifications import HIGHWAY, engine_with, run
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.tests.test_overlay_improvements import feed
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.traffic_light import TrafficLightChip
from openpilot.selfdrive.ui.ui_state import ui_state

MPH = CV.MPH_TO_MS
RED, YELLOW, GREEN = 1, 2, 3


def vehicle(track_id=1, d_rel=30.0, y_rel=3.6, v_rel=-2.0, last_seen=0.1, lane=-1):
  return SimpleNamespace(trackId=track_id, dRel=d_rel, yRel=y_rel, vRel=v_rel, lastSeen=last_seen, lane=lane)


def lead(present=True, d_rel=40.0, y_rel=0.0):
  return SimpleNamespace(present=present, dRel=d_rel, yRel=y_rel, radar=True, radarTrackId=-1, vRel=0.0)


class TestCameraTracks(OpenpilotTestCase):
  def test_cars_the_radar_has_already_are_left_out(self):
    radar_state = SimpleNamespace(leadOne=lead(d_rel=40.0), leadTwo=lead(present=False))
    tracks = camera_tracks([vehicle(1, 30.0, 3.6), vehicle(2, 41.5, 0.4), vehicle(3, 20.0, -3.5, last_seen=3.0)], [], radar_state, 25.0)
    assert [t.track_id for t in tracks] == [CAMERA_ID_BASE + 1]
    assert tracks[0].camera and abs(tracks[0].v_abs - 23.0) < 1e-6

  def test_they_join_the_shared_feed(self):
    class SM:
      valid = {"radarTracks": False, "radarState": True, "sceneDetectionSP": True}
      alive = {"radarTracks": False, "radarState": True, "sceneDetectionSP": True}
      recv_frame = {"radarTracks": -1, "radarState": 1, "carState": 1, "sceneDetectionSP": 1, "modelV2": -1}
      data = {"radarTracks": SimpleNamespace(points=[]), "carState": SimpleNamespace(vEgo=25.0),
              "radarState": SimpleNamespace(leadOne=lead(present=False), leadTwo=lead(present=False)),
              "sceneDetectionSP": SimpleNamespace(vehicles=[vehicle(7, 15.0, -3.5)])}

      def __getitem__(self, k):
        return self.data[k]
    feed_ = RadarFeed(0.05)
    assert feed_.update(SM(), 1)
    assert [t.track_id for t in feed_.tracks] == [CAMERA_ID_BASE + 7] and feed_.tracks[0].camera


def scene_msg(state="red", confidence=0.9, distance=60.0, age=2.0, vehicles=()):
  m = messaging.new_message('sceneDetectionSP')
  m.valid = True
  tl = m.sceneDetectionSP.trafficLight
  tl.state, tl.confidence, tl.distance, tl.stateAge = state, confidence, distance, age
  vs = m.sceneDetectionSP.init('vehicles', len(vehicles))
  for v, (d, y, lane) in zip(vs, vehicles, strict=True):
    v.dRel, v.yRel, v.lane, v.lastSeen = d, y, lane, 0.1
  return m


class TestCurrentLight(OpenpilotTestCase):
  def test_needs_the_toggle_and_confidence(self):
    with overlay_env(scene_detection=True, started=True, started_frame=0):
      feed(v_ego=20.0)
      ui_state.sm.update_msgs(ui_state.sm.recv_time['carState'] + 0.01, [scene_msg("red")])
      view = current_light()
      assert view is not None and view.state == RED and abs(view.distance - 60.0) < 1e-3
      ui_state.sm.update_msgs(ui_state.sm.recv_time['carState'] + 0.02, [scene_msg("green", confidence=0.3)])
      assert current_light() is None
    with overlay_env(scene_detection=False, started=True, started_frame=0):
      assert current_light() is None

  def test_chip_follows_the_light(self):
    with overlay_env(scene_detection=True, started=True, started_frame=0):
      chip = TrafficLightChip()
      with mock.patch("openpilot.selfdrive.ui.sunnypilot.onroad.overlays.traffic_light.current_light",
                      return_value=LightView(YELLOW, 0.9, 30.0, 1.0)):
        for _ in range(20):
          chip.update()
      assert chip._alpha.x > 0.9 and chip._view.state == YELLOW
      with mock.patch("openpilot.selfdrive.ui.sunnypilot.onroad.overlays.traffic_light.current_light", return_value=None):
        for _ in range(40):
          chip.update()
      assert chip._alpha.x < 0.05


# ---- notifications ----

APPROACH = replace(HIGHWAY, lead_present=False, v_ego=40 * MPH, light_state=RED, light_distance=80.0, light_age=3.0)


class TestLightNotifications(OpenpilotTestCase):
  def test_red_light_while_not_braking_enough(self):
    shown, _ = run(engine_with(dn.NotifyKind.RED_LIGHT), APPROACH, 1.0)
    assert len(shown) == 1 and shown[0].icon == dn.Icon.SIGNAL_RED and "ft" in shown[0].detail
    braking = replace(APPROACH, a_ego=-3.5)   # 40 mph needs about 2.3 m/s^2 for the 70 m left
    shown, _ = run(engine_with(dn.NotifyKind.RED_LIGHT), braking, 1.0)
    assert shown == []

  def test_yellow_says_whether_there_is_room(self):
    yellow = replace(APPROACH, light_state=YELLOW, light_age=0.2)
    shown, _ = run(engine_with(dn.NotifyKind.YELLOW_LIGHT), yellow, 0.5)
    assert shown[0].text == "Room to stop"
    shown, _ = run(engine_with(dn.NotifyKind.YELLOW_LIGHT), replace(yellow, light_distance=25.0), 0.5)
    assert shown[0].text == "Too close to stop gently"
    shown, _ = run(engine_with(dn.NotifyKind.YELLOW_LIGHT), replace(yellow, light_age=5.0), 0.5)
    assert shown == []   # an old yellow is no news

  def test_green_after_red_while_stopped(self):
    engine = engine_with(dn.NotifyKind.LIGHT_GREEN)
    stopped = replace(APPROACH, v_ego=0.0, light_distance=15.0)
    shown, t = run(engine, stopped, 2.0)
    assert shown == []
    shown, t = run(engine, replace(stopped, light_state=GREEN, light_age=0.1), 1.0, start=t)
    assert len(shown) == 1 and shown[0].icon == dn.Icon.SIGNAL_GREEN
    # a green that was never red while we waited (we rolled up to it) says nothing
    shown, _ = run(engine_with(dn.NotifyKind.LIGHT_GREEN), replace(stopped, light_state=GREEN, light_age=0.1), 1.0)
    assert shown == []

  def test_a_camera_car_beside_us_holds_back_the_pass(self):
    shown, _ = run(engine_with(dn.NotifyKind.SLOW_LEAD), replace(HIGHWAY, left_busy=True), dn.SLOW_SUSTAIN + 1)
    assert shown[0].icon == dn.Icon.WAIT

  def test_snapshot_reads_the_cameras(self):
    with overlay_env(scene_detection=True, drive_notifications=True, started=True, started_frame=0, speed_limit_mode=0):
      feed(v_ego=25.0)
      ui_state.sm.update_msgs(ui_state.sm.recv_time['carState'] + 0.01,
                              [scene_msg("red", vehicles=[(5.0, 3.5, -1), (40.0, -3.6, 1), (60.0, 0.0, 0)])])
      snap = dn.read_snapshot(ui_state.sm, 100.0)
      assert snap.light_state == RED and snap.left_busy and not snap.right_busy and snap.left_lane and snap.right_lane


class TestPaceBorderLight(OpenpilotTestCase):
  def setup_method(self):
    detection._feed = None

  def test_a_red_light_needs_stopping_for(self):
    with overlay_env(pace_border=True, started=True, started_frame=0), \
         mock.patch.object(pb, "current_light", return_value=LightView(RED, 0.9, 50.0, 2.0)):
      border = pb.PaceBorder()
      for _ in range(40):
        feed(v_ego=20.0)
        border.update()
      assert border.reason == "light" and border.level > 0.5

  def test_a_yellow_too_close_to_stop_for_is_left_alone(self):
    with overlay_env(pace_border=True, started=True, started_frame=0), \
         mock.patch.object(pb, "current_light", return_value=LightView(YELLOW, 0.9, 20.0, 0.5)):
      border = pb.PaceBorder()
      for _ in range(40):
        feed(v_ego=20.0)
        border.update()
      assert border.reason != "light" and border.level == 0.0
