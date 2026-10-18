"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The second round of scened: turn arrows, flashing signals, stop signs, brake lights, merging cars, people, radar range calibration,
pass scheduling and the status the Developer panel shows.
"""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.scened.detector import Detection, ObjectClass
from openpilot.sunnypilot.selfdrive.scened.scened import PASSES, ROAD, Context, PassScheduler, PlanView, RangeCalibrator, \
  SceneDetection, StatusLog
from openpilot.sunnypilot.selfdrive.scened.tests.test_scened import H, ROAD_GEO, W, FakeDetector, nv12
from openpilot.sunnypilot.selfdrive.scened.traffic_lights import Arrow, LightColor, LightObservation, SignObservation, StopSignFilter, \
  TrafficLightFilter, arrow_weight, read_arrow, read_lamp, sign_distance, sign_relevant, stop_sign_face
from openpilot.sunnypilot.selfdrive.scened.vehicle_tracker import VehicleSighting, VehicleTrack, VehicleTracker, rear_lamp_energy, \
  update_brake_lights

HERE = Path(__file__).parent


# ---- turn arrows ----

def arrow_mask(direction: str, height: int = 11) -> np.ndarray:
  """A lit arrow as it glows on a lamp: a triangle head the lamp's height and a thin shaft behind it, pointing left, right or up."""
  length, c = height * 3 // 2, height // 2
  m = np.zeros((height, length), bool)
  for i in range(c + 1):              # the head, a column per step, narrowing to its tip at column 0
    m[c - i:c + i + 1, i] = True
  m[c - 1:c + 2, c:] = True           # the shaft
  if direction == "right":
    m = m[:, ::-1]
  elif direction == "up":
    m = m.T
  return m


class TestArrows:
  @pytest.mark.parametrize("direction, expected", [("left", Arrow.LEFT), ("right", Arrow.RIGHT), ("up", Arrow.STRAIGHT)])
  def test_shapes(self, direction, expected):
    assert read_arrow(arrow_mask(direction)) == expected

  def test_a_ball_and_a_speck_are_no_arrow(self):
    yy, xx = np.mgrid[:15, :15]
    assert read_arrow((yy - 7) ** 2 + (xx - 7) ** 2 <= 36) == Arrow.NONE
    assert read_arrow(arrow_mask("left", 4)) == Arrow.NONE
    assert read_arrow(((yy - 7) / 4.0) ** 2 + ((xx - 7) / 7.0) ** 2 <= 1.0) == Arrow.NONE   # a ball stretched by its halo

  def test_real_green_left_arrow(self):
    data = np.load(HERE / "lamp_crops.npz")
    arrows = {str(n): read_lamp(data[str(n)].astype(np.float32)).arrow for n in data["names"]}
    assert arrows.pop("crop2") == Arrow.LEFT
    assert set(arrows.values()) == {Arrow.NONE}   # every other crop is a ball (or unlit)

  def test_an_arrow_counts_only_for_its_movement(self):
    assert arrow_weight(Arrow.NONE, -1) == 1.0
    assert arrow_weight(Arrow.LEFT, -1) > 1.0 and arrow_weight(Arrow.LEFT, 0) == 0.0
    assert arrow_weight(Arrow.STRAIGHT, 0) > 1.0 and arrow_weight(Arrow.RIGHT, 0) == 0.0
    assert arrow_weight(Arrow.NONE, -1, ours_shown=True) < 0.5   # a green ball beside our red arrow is for the others

  def test_a_red_left_arrow_does_not_stop_us_going_straight(self):
    f = TrafficLightFilter()
    for i in range(6):
      reading = f.update(i * 0.2, [LightObservation(LightColor.RED, 0.8, 0.0, 0.08, 40.0, -3.0, Arrow.LEFT),
                                   LightObservation(LightColor.GREEN, 0.8, 0.0, 0.08, 40.0, 0.0)], turn=0)
    assert reading.state == LightColor.GREEN and reading.arrow == Arrow.NONE
    f = TrafficLightFilter()
    for i in range(6):
      reading = f.update(i * 0.2, [LightObservation(LightColor.RED, 0.8, 0.0, 0.08, 40.0, -3.0, Arrow.LEFT),
                                   LightObservation(LightColor.GREEN, 0.8, 0.0, 0.08, 40.0, 0.0)], turn=-1)
    assert reading.state == LightColor.RED and reading.arrow == Arrow.LEFT


# ---- flashing signals ----

def obs(color, distance=40.0):
  return LightObservation(color, 0.8, 0.0, math.radians(5), distance, 0.0)


class TestFlashing:
  def run(self, pattern, period=0.17, seconds=5.0):
    """pattern(t) is the lamp color at t, or None when the head is in view but dark."""
    f = TrafficLightFilter()
    t = 0.0
    while t < seconds:
      color = pattern(t)
      reading = f.update(t, [obs(color if color is not None else LightColor.UNKNOWN)])
      t += period
    return reading

  def test_a_flashing_red(self):
    reading = self.run(lambda t: LightColor.RED if (t % 1.0) < 0.5 else None)
    assert reading.flashing and reading.state == LightColor.RED and reading.confidence >= 0.7

  def test_a_flashing_yellow(self):
    reading = self.run(lambda t: LightColor.YELLOW if (t % 1.0) < 0.5 else None)
    assert reading.flashing and reading.state == LightColor.YELLOW

  def test_a_steady_red_is_not_flashing(self):
    reading = self.run(lambda t: LightColor.RED)
    assert not reading.flashing and reading.state == LightColor.RED

  def test_red_then_green_is_a_change_not_a_flash(self):
    reading = self.run(lambda t: LightColor.RED if t < 2.5 else LightColor.GREEN)
    assert not reading.flashing and reading.state == LightColor.GREEN

  def test_a_flashing_green_does_not_exist(self):
    assert not self.run(lambda t: LightColor.GREEN if (t % 1.0) < 0.5 else None).flashing


# ---- stop signs ----

class TestStopSigns:
  def test_real_faces(self):
    data = np.load(HERE / "sign_crops.npz")
    for i, label in enumerate(data["labels"]):
      assert stop_sign_face(data[f"sign{i}"]) == bool(label), i

  def test_synthetic_faces(self):
    assert stop_sign_face(np.full((30, 30, 3), (200, 25, 40), np.uint8))
    assert not stop_sign_face(np.full((30, 30, 3), (240, 120, 20), np.uint8))   # an orange work zone diamond
    assert not stop_sign_face(np.full((30, 30, 3), (120, 120, 120), np.uint8))  # a sign's back
    assert not stop_sign_face(np.zeros((0, 0, 3), np.uint8))

  def test_distance_and_relevance(self):
    assert sign_distance(2648.0 * 0.76 / 30.0, 2648.0) == pytest.approx(30.0)
    assert sign_relevant(SignObservation(0.8, 30.0, 6.0), 0.0)
    assert not sign_relevant(SignObservation(0.8, 30.0, 14.0), 0.0)   # a side street's
    assert not sign_relevant(SignObservation(0.8, 120.0, 0.0), 0.0)

  def test_filter_shows_then_passes(self):
    f = StopSignFilter()
    assert not f.update(0.0, [SignObservation(0.6, 40.0, 4.0)], 10.0).present
    reading = f.update(0.2, [SignObservation(0.6, 38.0, 4.0)], 10.0)
    assert reading.present and 37.0 < reading.distance < 40.0
    reading = f.update(0.7, [], 10.0)
    assert reading.present and reading.distance < 35.0   # it comes closer between looks
    for i in range(8):
      reading = f.update(1.0 + 0.4 * i, [], 10.0)
    assert not reading.present   # driven up to it and through it


# ---- brake lights ----

class TestBrakeLights:
  def test_on_and_off(self):
    t = VehicleTrack(1, 30.0, 0.0)
    for i in range(5):
      update_brake_lights(t, 0.01, i * 0.2)
    assert not t.braking
    for i in range(5, 8):
      update_brake_lights(t, 0.06, i * 0.2)
    assert t.braking
    for i in range(8, 11):
      update_brake_lights(t, 0.01, i * 0.2)
    assert not t.braking

  def test_tail_lights_at_night_are_not_brakes(self):
    t = VehicleTrack(1, 30.0, 0.0)
    for i in range(30):
      update_brake_lights(t, 0.05 + 0.003 * math.sin(i), i * 0.2)
    assert not t.braking

  def test_forgotten_when_not_seen(self):
    t = VehicleTrack(1, 30.0, 0.0)
    for i, lamp in enumerate((0.01, 0.01, 0.08, 0.08, 0.08)):
      update_brake_lights(t, lamp, i * 0.2)
    assert t.braking
    update_brake_lights(t, math.nan, 0.9)
    assert t.braking
    update_brake_lights(t, math.nan, 3.0)
    assert not t.braking

  def test_lamp_energy(self):
    back = np.full((40, 60, 3), 70, np.uint8)
    assert rear_lamp_energy(back) == 0.0
    back[10:18, 4:14] = back[10:18, 46:56] = (250, 30, 30)
    assert rear_lamp_energy(back) == pytest.approx(160 / 2400)
    assert math.isnan(rear_lamp_energy(np.zeros((2, 2, 3), np.uint8)))


# ---- merging ----

class TestMerging:
  def drive(self, y0, vy, steps=10):
    tr = VehicleTracker()
    for i in range(steps):
      t = i * 0.25
      tr.update(t, [VehicleSighting(20.0, y0 + vy * t, 0.9, 2, 0)])
      tr.update_merging(lambda x: (-1.8, 1.8))
    return tr.tracks[0]

  def test_drifting_in_from_the_right(self):
    assert self.drive(4.5, -0.8).merging

  def test_drifting_in_from_the_left(self):
    assert self.drive(-4.5, 0.8).merging

  def test_holding_its_lane_or_drifting_away(self):
    assert not self.drive(4.5, 0.0).merging
    assert not self.drive(4.5, 0.6).merging


# ---- range calibration ----

class TestRangeCalibration:
  def test_learns_the_radar_scale(self):
    cal = RangeCalibrator()
    assert cal.scale == 1.0
    for _ in range(80):
      cal.learn([VehicleSighting(36.0, 0.1, 0.9, 2, 0), VehicleSighting(20.0, 3.6, 0.9, 2, 0)], (40.0, 0.0))
    assert cal.scale == pytest.approx(40.0 / 36.0, rel=0.02)
    (s,) = cal.apply([VehicleSighting(18.0, 1.0, 0.9, 2, 0)])
    assert s.x == pytest.approx(20.0, rel=0.02)

  def test_ignores_what_it_cannot_match_and_stays_bounded(self):
    cal = RangeCalibrator()
    for _ in range(10):
      cal.learn([VehicleSighting(20.0, 3.6, 0.9, 2, 0)], (40.0, 0.0))   # nothing on the lead
      cal.learn([VehicleSighting(4.0, 0.0, 0.9, 2, 0)], None)
    assert cal.matches == 0 and cal.scale == 1.0
    for _ in range(500):
      cal.learn([VehicleSighting(30.0, 0.0, 0.9, 2, 0)], (38.0, 0.0))
    assert cal.scale <= 1.33


# ---- scheduling ----

class TestScheduler:
  def shares(self, sched, n=600):
    counts = dict.fromkeys((p.name for p in PASSES), 0)
    for _ in range(n):
      counts[sched.next().name] += 1
    return {k: v / n for k, v in counts.items()}

  def test_town_looks_for_lights_highway_for_cars(self):
    sched = PassScheduler()
    sched.set_context(12.0, False, False)
    town = self.shares(sched)
    assert town["lights_near"] + town["lights_far"] > 0.45 and sched.lights_share == pytest.approx(3.5 / 7.0)
    sched.set_context(30.0, False, False)
    highway = self.shares(sched)
    assert highway["lights_near"] + highway["lights_far"] < 0.2 and min(highway.values()) > 0.0   # nothing starves
    sched.set_context(30.0, True, False)
    assert sched.lights_share > 0.45   # a planned stop at speed is a light coming

  def test_revisit(self):
    sched = PassScheduler()
    sched.set_context(30.0, False, False)
    road = next(p for p in PASSES if p.name == "road")
    assert sched.revisit(road, 0.2) == pytest.approx(0.2 * 4.2 / 2.0)


# ---- the message and the status ----

class TestSceneMessage:
  def test_people_signs_and_flags_reach_the_message(self):
    rgb = np.full((H, W, 3), 90, np.uint8)
    walker = ROAD_GEO.project([18.0], [2.0])[0]
    sign_pt = ROAD_GEO.project([30.0], [5.0], [1.22 - 2.1])[0]
    sw = 2648.0 * 0.76 / 30.4
    sx0, sy0 = int(sign_pt[0] - sw / 2), int(sign_pt[1] - sw / 2)
    rgb[sy0:sy0 + int(sw), sx0:sx0 + int(sw)] = (200, 25, 40)
    dets = [Detection(walker[0] - 20, walker[1] - 110, walker[0] + 20, walker[1], 0.8, ObjectClass.PERSON),
            Detection(sx0, sy0, sx0 + sw, sy0 + sw, 0.7, ObjectClass.STOP_SIGN)]
    buf, sampler = nv12(rgb)
    scene = SceneDetection(FakeDetector(dets))
    t = 0.0
    road = next(p for p in PASSES if p.name == "road")
    for _ in range(4):
      scene.run_pass(road, buf, sampler, ROAD_GEO, PlanView(), Context(t))
      t += 0.2
    msg = scene.message(t, PlanView(), 1, "road", 0.01, 5.0, False, True, 0.4).sceneDetectionSP
    assert msg.stopSign.present and abs(msg.stopSign.distance - (30.0 - 1.52)) < 3.0
    (p,) = msg.people
    assert p.kind == "pedestrian" and abs(p.dRel - (18.0 - 1.52)) < 1.5 and abs(p.yRel + 2.0) < 0.4
    assert len(msg.vehicles) == 0 and msg.rangeScale == 1.0 and msg.lightsShare == pytest.approx(0.4)

  def test_status_snapshot(self):
    scene = SceneDetection(FakeDetector([]))
    log = StatusLog(started=0.0)
    for i, spec in enumerate(PASSES * 3):
      log.record(i * 0.2, spec, 0.05, i % 4 == 0, scene)
    scene.signs.reading = SimpleNamespace(present=True, distance=25.0)
    log.record(3.0, PASSES[0], 0.05, False, scene)
    snap = log.snapshot(60.0, scene, True, 0.2, 0.5)
    assert snap["uptime_s"] == 60 and snap["pass_rate"] == 5.0 and snap["lights_share_pct"] == 50
    assert snap["passes"]["road"] == {"count": 3, "avg_ms": 50.0} and snap["throttled_pct"] == pytest.approx(23.1, abs=0.1)
    assert snap["stop_signs"] == 1 and "stop sign at 25 m" in snap["events"][0]["event"]
    assert snap["range_scale"] == 1.0 and snap["calibration_matches"] == 0
    assert scene.calibration[ROAD].scale == 1.0
