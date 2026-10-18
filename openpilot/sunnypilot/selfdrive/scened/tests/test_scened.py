import math
from pathlib import Path

import numpy as np
import pytest

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.sunnypilot.selfdrive.scened import detector as det
from openpilot.sunnypilot.selfdrive.scened.detector import Detection, ObjectClass, decode, nms
from openpilot.sunnypilot.selfdrive.scened.frames import INPUT_SIZE, NV12Sampler, Region, clamp_region, rgb_to_nv12
from openpilot.sunnypilot.selfdrive.scened.geometry import camera_geometry
from openpilot.sunnypilot.selfdrive.scened.scened import PASSES, Context, PlanView, SceneDetection, Throttle, lights_region, MIN_PASS_HZ, \
  PASS_HZ
from openpilot.sunnypilot.selfdrive.scened.traffic_lights import LightColor, LightObservation, TrafficLightFilter, estimate_distance, \
  read_lamp, relevant
from openpilot.sunnypilot.selfdrive.scened.vehicle_tracker import VehicleSighting, VehicleTracker, ego_motion, lane_index, sightings_from

HERE = Path(__file__).parent
W, H = 1928, 1208
CAMS = DEVICE_CAMERAS[("tici", "ar0231")]
PITCH = math.radians(2.0)
ROAD_GEO = camera_geometry(CAMS.narrow_road.intrinsics, [0.0, PITCH, 0.0], 1.22)


def nv12(rgb):
  stride, y_height, uv_height, _ = get_nv12_info(rgb.shape[1], rgb.shape[0])
  return rgb_to_nv12(rgb, stride, y_height, uv_height), NV12Sampler(rgb.shape[1], rgb.shape[0], stride, stride * y_height)


# ---- frames ----

class TestFrames:
  def test_round_trip_keeps_colors(self):
    rgb = np.zeros((64, 96, 3), np.uint8)
    rgb[:, :32] = (220, 30, 40)
    rgb[:, 32:64] = (40, 200, 90)
    rgb[:, 64:] = (250, 210, 60)
    buf, sampler = nv12(rgb)
    back = sampler.sample(buf, np.arange(64), np.arange(96))
    assert np.abs(back - rgb).max() < 12

  def test_detector_input_is_letterboxed_bgr(self):
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[..., 0] = 200  # red
    buf, sampler = nv12(rgb)
    region = Region(0, 0, W, H)
    x = sampler.detector_input(buf, region)
    rows = int(H * region.scale)
    assert x.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert x[0, 2, 10, 10] > 150 and x[0, 0, 10, 10] < 40     # channel 2 is red: BGR
    assert np.all(x[0, :, rows + 2:, :] == 114.0)               # padding below the frame

  def test_regions_stay_inside_the_frame(self):
    r = clamp_region(-50, 900, 624, 624, W, H)
    assert r.x == 0 and r.y == H - 624
    assert clamp_region(10, 10, 4000, 4000, W, H) == Region(0, 0, W, H)

  def test_crop_is_capped(self):
    buf, sampler = nv12(np.full((H, W, 3), 128, np.uint8))
    assert max(sampler.crop_rgb(buf, 100, 100, 300, 260).shape[:2]) <= 48
    assert sampler.crop_rgb(buf, 10, 10, 11, 11).size == 0


# ---- detector output ----

class TestDecode:
  def raw_with(self, cells):
    raw = np.zeros((1, 3549, 85), np.float32)
    for idx, (dx, dy, w, h, cls, score) in cells.items():
      raw[0, idx, :4] = (dx, dy, math.log(w / 8), math.log(h / 8))
      raw[0, idx, 4] = 1.0
      raw[0, idx, 5 + cls] = score
    return raw

  def test_box_lands_in_source_pixels(self):
    # cell 0 of the stride-8 grid, centered at (0.5, 0.5) cells = (4, 4) px, 40 x 24 px
    raw = self.raw_with({0: (0.5, 0.5, 40.0, 24.0, ObjectClass.CAR, 0.9)})
    region = Region(100, 50, 832, 832)
    (d,) = decode(raw, region)
    r = region.scale
    assert d.cls == ObjectClass.CAR and abs(d.score - 0.9) < 1e-6
    assert abs(d.x0 - (100 + (4 - 20) / r)) < 1e-3 and abs(d.y1 - (50 + (4 + 12) / r)) < 1e-3

  def test_unused_classes_and_weak_boxes_are_dropped(self):
    raw = self.raw_with({0: (0.5, 0.5, 40, 24, 15, 0.95), 1: (0.5, 0.5, 40, 24, ObjectClass.CAR, 0.2),
                         2: (0.5, 0.5, 10, 30, ObjectClass.TRAFFIC_LIGHT, 0.28)})
    dets = decode(raw, Region(0, 0, INPUT_SIZE, INPUT_SIZE))
    assert [d.cls for d in dets] == [ObjectClass.TRAFFIC_LIGHT]   # a cat dropped, weak car dropped, light kept at its lower bar

  def test_nms(self):
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    assert nms(boxes, np.array([0.9, 0.8, 0.7])) == [0, 2]


# ---- geometry ----

class TestGeometry:
  def test_ground_round_trip(self):
    pts = ROAD_GEO.project([10.0, 30.0, 60.0], [0.0, -3.5, 4.0])
    back = ROAD_GEO.ground(pts[:, 0], pts[:, 1])
    assert np.allclose(back, [[10, 0], [30, -3.5], [60, 4.0]], atol=1e-6)

  def test_above_the_horizon_never_meets_the_road(self):
    horizon = ROAD_GEO.project([1e4], [0.0], [0.0])[0]
    assert np.isnan(ROAD_GEO.ground(horizon[0], horizon[1] - 20)[0, 0])
    bearing, elevation = ROAD_GEO.angles(horizon[0], horizon[1] - 100)
    assert abs(bearing[0]) < 1e-6 and elevation[0] > 0

  def test_wide_camera_uses_its_own_rotation(self):
    wide = camera_geometry(CAMS.wide_road.intrinsics, [0.0, 0.0, 0.0], 1.22, [0.0, 0.0, math.radians(3)])
    straight = camera_geometry(CAMS.wide_road.intrinsics, [0.0, 0.0, 0.0], 1.22, [0.0, 0.0, 0.0])
    assert abs(wide.project([20.0], [0.0])[0, 0] - straight.project([20.0], [0.0])[0, 0]) > 10


# ---- traffic lights ----

def head(lit: int, color, night: bool = False):
  """A vertical signal head, lamp `lit` (0 top .. 2 bottom) glowing in color."""
  img = np.full((60, 24, 3), 25, np.uint8)
  y = 10 + 20 * lit
  yy, xx = np.mgrid[:60, :24]
  lamp = (yy - y) ** 2 + (xx - 12) ** 2 <= (14 if night else 7) ** 2
  img[lamp] = color
  if night:
    img[(yy - y) ** 2 + (xx - 12) ** 2 <= 9] = (240, 250, 255)   # burnt out core
  return img


class TestLamp:
  @pytest.mark.parametrize("lit, color, expected", [
    (0, (230, 40, 35), LightColor.RED),
    (1, (240, 190, 40), LightColor.YELLOW),
    (2, (40, 220, 180), LightColor.GREEN),   # LED green reads blue-green
  ])
  def test_synthetic_heads(self, lit, color, expected):
    assert read_lamp(head(lit, color).astype(np.float32)).color == expected

  def test_night_flare_reads_its_halo(self):
    assert read_lamp(head(0, (230, 40, 60), night=True).astype(np.float32)).color == LightColor.RED

  def test_unlit_or_sky(self):
    assert read_lamp(np.full((60, 24, 3), 25, np.float32)).color == LightColor.UNKNOWN
    assert read_lamp(np.full((60, 24, 3), (150, 180, 215), np.float32)).color == LightColor.UNKNOWN

  def test_real_crops(self):
    data = np.load(HERE / "lamp_crops.npz")
    for name, label in zip(data["names"], data["labels"], strict=True):
      assert read_lamp(data[str(name)].astype(np.float32)).color == LightColor(int(label)), name


class TestLightFilter:
  def obs(self, color, conf=0.8, distance=40.0):
    return LightObservation(color, conf, 0.0, math.radians(5), distance, 0.0)

  def test_needs_agreement_and_ignores_one_odd_reading(self):
    f = TrafficLightFilter()
    assert f.update(0.0, [self.obs(LightColor.RED)]).state == LightColor.UNKNOWN
    assert f.update(0.2, [self.obs(LightColor.RED)]).state == LightColor.RED
    assert f.update(0.4, [self.obs(LightColor.GREEN)]).state == LightColor.RED
    for t in (0.6, 0.8, 1.0, 1.2):
      reading = f.update(t, [self.obs(LightColor.GREEN)])
    assert reading.state == LightColor.GREEN and reading.since > 0.4

  def test_goes_unknown_when_out_of_sight(self):
    f = TrafficLightFilter()
    for t in (0.0, 0.2, 0.4):
      f.update(t, [self.obs(LightColor.RED)])
    assert f.update(1.0, []).state == LightColor.RED
    assert f.update(2.5, []).state == LightColor.UNKNOWN and math.isnan(f.reading.distance)

  def test_relevance(self):
    above = LightObservation(LightColor.RED, 0.8, 0.0, math.radians(4), 50.0, 1.0)
    assert relevant(above, 0.0)
    assert not relevant(LightObservation(LightColor.RED, 0.8, 0.0, math.radians(-1), 50.0, 1.0), 0.0)   # a tail light
    assert not relevant(LightObservation(LightColor.RED, 0.8, 0.2, math.radians(4), 50.0, 10.0), 0.0)   # over another road
    assert relevant(LightObservation(LightColor.RED, 0.8, 0.2, math.radians(4), 50.0, 10.0), 8.0)       # unless our path bends there

  def test_distance_estimate(self):
    d = 50.0
    width = 2648.0 * 0.36 / d
    elevation = math.atan((5.2 - 1.22) / d)
    assert abs(estimate_distance(width, 2648.0, elevation, 1.22) - d) < 0.5


# ---- vehicles ----

class TestTracker:
  def test_confirms_and_estimates_closing_speed(self):
    tr = VehicleTracker()
    for i in range(12):
      t = i * 0.33
      tracks = tr.update(t, [VehicleSighting(40.0 - 5.0 * t, 0.2, 0.9, 2, 0)])
    (track,) = tracks
    assert track.hits == 12 and abs(track.vx + 5.0) < 0.6 and abs(track.x - (40 - 5 * 11 * 0.33)) < 1.0

  def test_one_sighting_is_not_a_car_yet_and_tracks_expire(self):
    tr = VehicleTracker()
    assert tr.update(0.0, [VehicleSighting(20.0, 3.5, 0.9, 2, 0)]) == []
    assert len(tr.update(0.3, [VehicleSighting(20.0, 3.5, 0.9, 2, 0)])) == 1
    tr.update(2.0, [])
    assert tr.tracks == []

  def test_two_cars_stay_apart_and_twins_merge(self):
    tr = VehicleTracker()
    for t in (0.0, 0.3):
      tracks = tr.update(t, [VehicleSighting(20.0, 0.0, 0.9, 2, 0), VehicleSighting(20.0, 3.6, 0.9, 2, 0),
                             VehicleSighting(20.3, 0.1, 0.6, 2, 0)])   # a second box on the first car
    assert sorted(round(t.y) for t in tracks) == [0, 4]

  def test_turning_swings_things_right(self):
    x, y = ego_motion(10.0, 0.0, 0.1)
    assert y > 0.9 and x < 10.0

  def test_lanes(self):
    lines = lambda x: (-1.7, 1.9)   # noqa: E731
    assert [lane_index(20.0, y, lines) for y in (0.0, -3.0, 3.5, -7.0, 9.0)] == [0, -1, 1, -2, 2]
    assert lane_index(20.0, 2.5, lambda x: None) == 1

  def test_sightings_are_placed_on_the_road(self):
    pt = ROAD_GEO.project([30.0], [3.6])[0]
    box = Detection(pt[0] - 60, pt[1] - 90, pt[0] + 60, pt[1], 0.9, ObjectClass.CAR)
    (s,) = sightings_from([box], ROAD_GEO, 0, 100.0, W, H)
    assert abs(s.x - 30.0) < 0.5 and abs(s.y - 3.6) < 0.2

  def test_width_steadies_a_miscalibrated_pitch(self):
    # the car is 40 m out and 1.8 m wide, but the calibration we use is off by half a degree
    true_geo = ROAD_GEO
    off_geo = camera_geometry(CAMS.narrow_road.intrinsics, [0.0, PITCH + math.radians(0.5), 0.0], 1.22)
    pt = true_geo.project([40.0], [0.0])[0]
    w = 2648.0 * 1.8 / 40.0
    box = Detection(pt[0] - w / 2, pt[1] - w * 0.8, pt[0] + w / 2, pt[1], 0.9, ObjectClass.CAR)
    (ground_only,) = sightings_from([box], off_geo, 0, 200.0, W, H)
    (blended,) = sightings_from([box], off_geo, 0, 200.0, W, H, width_focal=2648.0)
    assert abs(blended.x - 40.0) < abs(ground_only.x - 40.0)


  def test_a_car_hidden_behind_a_nearer_one_is_placed_by_its_width(self):
    near = ROAD_GEO.project([10.0], [0.0])[0]
    near_w = 2648.0 * 1.8 / 10.0
    near_box = Detection(near[0] - near_w / 2, near[1] - 0.8 * near_w, near[0] + near_w / 2, near[1], 0.9, ObjectClass.CAR)
    far = ROAD_GEO.project([40.0], [0.0])[0]
    far_w = 2648.0 * 1.8 / 40.0
    # its lower half is behind the near car, so its box stops short of the road
    far_box = Detection(far[0] - far_w / 2, far[1] - 0.8 * far_w, far[0] + far_w / 2, far[1] - 0.4 * far_w, 0.8, ObjectClass.CAR)
    twin = Detection(near_box.x0 + 3, near_box.y0, near_box.x1 + 3, near_box.y1 + 9, 0.6, ObjectClass.CAR)
    placed = sightings_from([near_box, far_box, twin], ROAD_GEO, 0, 200.0, W, H, width_focal=2648.0)
    assert len(placed) == 3
    assert abs(placed[1].x - 40.0) < 3.0                     # not the 80+ m its box bottom would say
    assert abs(placed[0].x - 10.0) < 1.0 and abs(placed[2].x - 10.0) < 1.5   # a twin box is no occluder


# ---- the process logic ----

class FakeDetector:
  def __init__(self, dets):
    self.dets = dets
    self.compiled = True

  def __call__(self, image, region):
    return [d for d in self.dets if region.x <= d.x0 and d.x1 <= region.x + region.w]


class TestSceneDetection:
  def test_throttle(self):
    th = Throttle()
    for _ in range(40):
      th.update(10.0, False)
    assert th.interval == pytest.approx(1 / MIN_PASS_HZ) and th.throttled
    for _ in range(200):
      th.update(0.0, False)
    assert th.interval == pytest.approx(1 / PASS_HZ) and not th.throttled
    assert th.update(0.0, False, pass_time=0.12) == pytest.approx(0.12 / 0.3)   # never more than 30 % busy

  def test_lights_region_looks_above_the_horizon(self):
    r = lights_region(ROAD_GEO, 624, W, H)
    horizon_v = ROAD_GEO.project([1e4], [0.0], [0.0])[0, 1]
    assert r.y < horizon_v < r.y + r.h and (horizon_v - r.y) / r.h > 0.75

  def test_a_green_light_and_a_car_end_up_in_the_message(self):
    rgb = np.full((H, W, 3), 90, np.uint8)
    # a signal head 45 m ahead, 5.2 m up, over our lane, lit green at the bottom
    top = ROAD_GEO.project([45.0], [0.0], [1.22 - 5.6])[0]
    bottom = ROAD_GEO.project([45.0], [0.0], [1.22 - 4.6])[0]
    hw = (bottom[1] - top[1]) * 0.18
    x0, y0, x1, y1 = int(top[0] - hw), int(top[1]), int(top[0] + hw), int(bottom[1])
    rgb[y0:y1, x0:x1] = 25
    rgb[y1 - (y1 - y0) // 3 + 2:y1 - 2, x0 + 2:x1 - 2] = (40, 220, 180)
    car = ROAD_GEO.project([25.0], [-3.6])[0]
    car_w = 2648.0 * 1.8 / math.hypot(25.0, 3.6)   # a 1.8 m wide car at that range
    dets = [Detection(x0, y0, x1, y1, 0.8, ObjectClass.TRAFFIC_LIGHT),
            Detection(car[0] - car_w / 2, car[1] - 0.8 * car_w, car[0] + car_w / 2, car[1], 0.9, ObjectClass.CAR)]
    buf, sampler = nv12(rgb)
    scene = SceneDetection(FakeDetector(dets))
    t = 0.0
    for spec in PASSES * 2:
      if spec.camera == 0:
        scene.run_pass(spec, buf, sampler, ROAD_GEO, PlanView(), Context(t))
      t += 0.17
    msg = scene.message(t, PlanView(), 7, "road", 0.01, 6.0, False, True).sceneDetectionSP
    assert msg.trafficLight.state == "green" and abs(msg.trafficLight.distance - (45 - 1.52)) < 6
    (v,) = msg.vehicles
    assert abs(v.dRel - (25 - 1.52)) < 1.0 and abs(v.yRel - 3.6) < 0.3 and v.lane == -1


@pytest.mark.skipif(not (det.PKL_PATH.exists() or Path(str(det.PKL_PATH) + ".chunkmanifest").exists()),
                    reason="the build compiles the detector")
def test_detector_on_a_real_frame():
  from PIL import Image
  rgb = np.asarray(Image.open(HERE / "road_scene.jpg").convert("RGB"))
  buf, sampler = nv12(rgb)
  detector = det.Detector()
  region = Region(0, 0, W, H)
  cars = [d for d in detector(sampler.detector_input(buf, region), region) if d.cls == ObjectClass.CAR]
  assert len(cars) >= 5
  # the car straight ahead, a white sedan around x 910-1245, y 465-730
  assert any(abs(d.x0 - 910) < 40 and abs(d.x1 - 1245) < 40 and abs(d.y1 - 728) < 30 for d in cars)
  lights_r = lights_region(ROAD_GEO, 832, W, H)
  lights = [d for d in detector(sampler.detector_input(buf, lights_r), lights_r) if d.cls == ObjectClass.TRAFFIC_LIGHT]
  greens = [read_lamp(sampler.crop_rgb(buf, d.x0, d.y0, d.x1, d.y1)).color for d in lights]
  assert greens.count(LightColor.GREEN) >= 2
