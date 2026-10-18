"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Camera Object Detection (scened) in the HUD: the traffic light, stop signs, people and the camera's cars.
"""
import math
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest

from openpilot.cereal import messaging
from openpilot.common.constants import CV
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays import detection, drive_notifications as dn, light_view, pace_border as pb
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.detection import StopPhase
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.light_view import LightView, SignView, current_light
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_tracks import CAMERA_ID_BASE, RadarFeed, camera_tracks
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.tests.test_ar_overlays import next_sm_time, overlay_env
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.tests.test_drive_notifications import HIGHWAY, engine_with, run
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.tests.test_overlay_improvements import feed
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.traffic_light import TrafficLightChip, wait_text
from openpilot.selfdrive.ui.ui_state import ui_state

MPH = CV.MPH_TO_MS
RED, YELLOW, GREEN = 1, 2, 3


def send_scene(msg) -> None:
  ui_state.sm.update_msgs(next_sm_time(0.01), [msg])


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
      send_scene(scene_msg("red"))
      view = current_light()
      assert view is not None and view.state == RED and abs(view.distance - 60.0) < 1e-3
      send_scene(scene_msg("green", confidence=0.3))
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
      send_scene(scene_msg("red", vehicles=[(5.0, 3.5, -1), (40.0, -3.6, 1), (60.0, 0.0, 0)]))
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


# ---- round two: stop signs, flashing and arrows, people, brake lights, merging ----

def scene_extras(sign=None, people=(), vehicles=(), flashing=False, arrow="none", state="red"):
  """sign: distance or None. people: (dRel, yRel, kind). vehicles: (dRel, yRel, lane, braking, merging)."""
  m = scene_msg(state)
  sd = m.sceneDetectionSP
  sd.trafficLight.flashing, sd.trafficLight.arrow = flashing, arrow
  if sign is not None:
    sd.stopSign.present, sd.stopSign.distance, sd.stopSign.confidence = True, sign, 0.9
  ps = sd.init('people', len(people))
  for p, (d, y, kind) in zip(ps, people, strict=True):
    p.dRel, p.yRel, p.kind, p.lastSeen, p.vRel = d, y, kind, 0.1, -10.0
  vs = sd.init('vehicles', len(vehicles))
  for v, (d, y, lane, braking, merging) in zip(vs, vehicles, strict=True):
    v.dRel, v.yRel, v.lane, v.lastSeen, v.braking, v.merging = d, y, lane, 0.1, braking, merging
  return m


def stop_feed(stop_x=None, car_ahead=False):
  stop = SimpleNamespace(phase=StopPhase.APPROACH if stop_x is not None else StopPhase.IDLE, stop_x=stop_x, car_ahead=car_ahead)
  return SimpleNamespace(stop=stop)


class TestFusedStop(OpenpilotTestCase):
  def test_the_models_stop_wins_near_the_light(self):
    with mock.patch.object(light_view, "get_detection_feed", return_value=stop_feed(40.0)):
      # a light on the far side of a crossing, 62 m out: the model's stop at 38.5 m is its stop line
      assert light_view.fused_stop_distance(62.0, light_view.LIGHT_STOP_WINDOW, 10.0) == pytest.approx(40.0 - 1.52)
      # a stop the model plans for something else, far from the light, is not it
      assert light_view.fused_stop_distance(120.0, light_view.LIGHT_STOP_WINDOW, 10.0) == 110.0
    with mock.patch.object(light_view, "get_detection_feed", return_value=stop_feed(40.0, car_ahead=True)):
      assert light_view.fused_stop_distance(62.0, light_view.LIGHT_STOP_WINDOW, 10.0) == 52.0   # stopping behind a car
    assert math.isnan(light_view.fused_stop_distance(math.nan, light_view.LIGHT_STOP_WINDOW, 10.0))

  def test_stop_at(self):
    assert LightView(RED, 0.9, 50.0, 1.0).stop_at == 40.0
    assert LightView(RED, 0.9, 50.0, 1.0, stop_distance=33.0).stop_at == 33.0
    assert LightView(RED, 0.9, 5.0, 1.0).stop_at == 0.0


class TestReaders(OpenpilotTestCase):
  def setup_method(self):
    detection._feed = None

  def test_sign_people_flashing_and_arrow(self):
    with overlay_env(scene_detection=True, started=True, started_frame=0):
      feed(v_ego=10.0)
      send_scene(scene_extras(sign=30.0, people=[(12.0, -1.0, "pedestrian")], flashing=True, arrow="left"))
      light = current_light()
      assert light.flashing and light.arrow == light_view.ARROW_LEFT
      sign = light_view.current_stop_sign()
      assert sign.distance == 30.0 and sign.stop_distance == 30.0
      (p,) = light_view.camera_people()
      assert p.dRel == 12.0


def snap_with(**kw):
  return replace(HIGHWAY, lead_present=False, v_ego=30 * MPH, **kw)


class TestNewRules(OpenpilotTestCase):
  def test_stop_sign_while_not_braking(self):
    shown, _ = run(engine_with(dn.NotifyKind.STOP_SIGN), snap_with(sign_stop_distance=45.0), 1.0)
    assert len(shown) == 1 and shown[0].icon == dn.Icon.STOP_SIGN
    shown, _ = run(engine_with(dn.NotifyKind.STOP_SIGN), snap_with(sign_stop_distance=45.0, a_ego=-3.0), 1.0)
    assert shown == []   # already stopping for it

  def test_flashing_red_and_yellow(self):
    shown, _ = run(engine_with(dn.NotifyKind.FLASHING_SIGNAL), snap_with(light_state=RED, light_flashing=True, light_distance=60.0), 0.5)
    assert shown[0].text == "Stop, then go when clear"
    shown, _ = run(engine_with(dn.NotifyKind.FLASHING_SIGNAL), snap_with(light_state=YELLOW, light_flashing=True, light_distance=60.0), 0.5)
    assert shown[0].text == "Proceed with caution"

  def test_a_flashing_red_is_not_a_red_light(self):
    flashing = snap_with(light_state=RED, light_flashing=True, light_distance=60.0, light_age=3.0)
    shown, _ = run(engine_with(dn.NotifyKind.RED_LIGHT, dn.NotifyKind.YELLOW_LIGHT), flashing, 1.0)
    assert shown == []

  def test_pedestrian_and_cyclist(self):
    shown, _ = run(engine_with(dn.NotifyKind.PEDESTRIAN), snap_with(person_distance=12.0, person_ttc=1.5), 0.3)
    assert shown[0].text == "Pedestrian ahead" and shown[0].icon == dn.Icon.PERSON
    shown, _ = run(engine_with(dn.NotifyKind.PEDESTRIAN), snap_with(person_distance=40.0, person_ttc=4.0, person_cyclist=True), 0.3)
    assert shown[0].text == "Cyclist ahead"
    shown, _ = run(engine_with(dn.NotifyKind.PEDESTRIAN), snap_with(person_distance=40.0, person_ttc=20.0), 0.3)
    assert shown == []

  def test_braking_ahead_and_merging(self):
    shown, _ = run(engine_with(dn.NotifyKind.BRAKING_AHEAD), snap_with(braking_ahead=70.0), 1.0)
    assert shown[0].text == "Brake lights ahead"
    shown, _ = run(engine_with(dn.NotifyKind.MERGING), snap_with(merging_side=1, merging_distance=8.0), 0.3)
    assert shown[0].text == "From the right" and shown[0].icon == dn.Icon.LEFT

  def test_a_pedestrian_outranks_a_curve(self):
    order = [r.kind for r in dn.default_rules()]
    assert order.index(dn.NotifyKind.PEDESTRIAN) < order.index(dn.NotifyKind.SHARP_CURVE)

  def test_snapshot_reads_people_braking_and_merging(self):
    with overlay_env(scene_detection=True, drive_notifications=True, started=True, started_frame=0, speed_limit_mode=0):
      feed(v_ego=20.0)
      msg = scene_extras(sign=40.0, people=[(20.0, 0.3, "cyclist"), (15.0, 8.0, "pedestrian")],
                         vehicles=[(70.0, 0.0, 0, True, False), (12.0, -3.4, 1, False, True)])
      send_scene(msg)
      snap = dn.read_snapshot(ui_state.sm, 100.0)
      assert snap.sign_stop_distance == 40.0
      assert snap.person_distance == 20.0 and snap.person_cyclist   # the one off the road is left out
      assert snap.braking_ahead == 70.0 and snap.merging_side == 1 and snap.merging_distance == 12.0


class TestPaceBorderSigns(OpenpilotTestCase):
  def setup_method(self):
    detection._feed = None

  def border_after(self, speeds, sign, light=None):
    with overlay_env(pace_border=True, started=True, started_frame=0), \
         mock.patch.object(pb, "current_light", return_value=light), mock.patch.object(pb, "current_stop_sign", return_value=sign):
      border = pb.PaceBorder()
      for v in speeds:
        feed(v_ego=v)
        border.update()
    return border

  def test_a_stop_sign_needs_stopping_for(self):
    border = self.border_after([15.0] * 40, SignView(30.0, 0.9, 30.0))
    assert border.reason == "sign" and border.level > 0.5

  def test_pulling_away_after_stopping_is_quiet(self):
    border = self.border_after([0.2] * 10 + [4.0] * 30, SignView(4.0, 0.9, 4.0))
    assert border.reason != "sign" and border.level < 0.05

  def test_a_flashing_red_is_a_stop_sign_and_a_flashing_yellow_nothing(self):
    border = self.border_after([15.0] * 40, None, LightView(RED, 0.9, 40.0, 3.0, flashing=True))
    assert border.reason == "sign"
    border = self.border_after([15.0] * 40, None, LightView(YELLOW, 0.9, 40.0, 3.0, flashing=True))
    assert border.reason == "" and border.level == 0.0


class TestChip(OpenpilotTestCase):
  def test_counts_the_wait_at_a_red(self):
    with overlay_env(scene_detection=True, started=True, started_frame=0):
      chip = TrafficLightChip()
      feed(v_ego=0.0)
      with mock.patch("openpilot.selfdrive.ui.sunnypilot.onroad.overlays.traffic_light.current_light",
                      return_value=LightView(RED, 0.9, 15.0, 30.0)), \
           mock.patch("openpilot.selfdrive.ui.sunnypilot.onroad.overlays.traffic_light.time.monotonic", side_effect=[100.0, 142.5]):
        chip.update()
        assert chip.waited == pytest.approx(42.5) and wait_text(42.5) == "0:42"
      feed(v_ego=10.0)
      with mock.patch("openpilot.selfdrive.ui.sunnypilot.onroad.overlays.traffic_light.current_light",
                      return_value=LightView(RED, 0.9, 15.0, 30.0)):
        chip.update()
      assert chip.waited is None

  def test_shows_a_stop_sign_when_there_is_no_light(self):
    with overlay_env(scene_detection=True, started=True, started_frame=0):
      chip = TrafficLightChip()
      with mock.patch("openpilot.selfdrive.ui.sunnypilot.onroad.overlays.traffic_light.current_light", return_value=None), \
           mock.patch("openpilot.selfdrive.ui.sunnypilot.onroad.overlays.traffic_light.current_stop_sign",
                      return_value=SignView(25.0, 0.9, 25.0)):
        for _ in range(20):
          chip.update()
      assert isinstance(chip._view, SignView) and chip._alpha.x > 0.9


class TestCameraFlagsOnTracks(OpenpilotTestCase):
  def test_braking_and_merging_carry_over(self):
    from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_tracks import make_track, with_camera_flags
    radar = [make_track(5, 40.0, 0.0, -1.0, 20.0), make_track(6, 20.0, 3.5, 0.0, 20.0)]
    cams = [SimpleNamespace(trackId=1, dRel=41.0, yRel=0.3, vRel=-1.0, lastSeen=0.1, lane=0, braking=True, merging=False)]
    flagged = with_camera_flags(radar, cams)
    assert flagged[0].braking and not flagged[0].merging and not flagged[1].braking
    tracks = camera_tracks([SimpleNamespace(trackId=2, dRel=12.0, yRel=-3.4, vRel=0.0, lastSeen=0.1, lane=1, braking=False, merging=True)],
                           radar, None, 20.0)
    assert tracks[0].merging and tracks[0].camera
