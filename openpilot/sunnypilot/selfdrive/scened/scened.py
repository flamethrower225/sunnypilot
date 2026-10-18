#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

scened: camera object detection, publishing sceneDetectionSP.

It takes turns looking at four regions, one per pass:
  lights_near  a 624 px square of the road camera above the horizon, straight ahead: signal heads out to ~80 m
  wide         the whole wide camera: cars beside and just ahead, and lights overhead when stopped at the line
  lights_far   an 832 px square, the same place: heads further out to the sides
  road         the whole road camera: cars ahead in every lane out to ~100 m
and folds each into the traffic light filter and the vehicle tracker. It runs well below the driving model's rate, at low CPU
priority, and slows down further whenever the driving model starts dropping frames or the device runs hot: the driving model always
comes first.
"""
import math
import os
import time
from dataclasses import dataclass

import numpy as np

from openpilot.cereal import custom, log, messaging
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.sunnypilot.selfdrive.scened.detector import VEHICLE_CLASSES, Detection, Detector, ObjectClass
from openpilot.sunnypilot.selfdrive.scened.frames import NV12Sampler, Region, clamp_region
from openpilot.sunnypilot.selfdrive.scened.geometry import RADAR_TO_CAMERA, CameraGeometry, camera_geometry
from openpilot.sunnypilot.selfdrive.scened.traffic_lights import LightColor, LightObservation, TrafficLightFilter, estimate_distance, \
  read_lamp, relevant
from openpilot.sunnypilot.selfdrive.scened.vehicle_tracker import VehicleTracker, lane_index, sightings_from

PASS_HZ = 6.0              # passes per second at most
MIN_PASS_HZ = 1.0
FRAME_DROP_LIMIT = 3.0     # % of driving model frames dropped before scened backs off
BACKOFF, RECOVER = 1.25, 0.97
DUTY_MAX = 0.3             # share of the time scened may spend in passes
ROAD, WIDE = 0, 1

LightState = custom.SceneDetectionSP.LightState
VehicleKind = custom.SceneDetectionSP.VehicleKind
KIND = {ObjectClass.CAR: VehicleKind.car, ObjectClass.MOTORCYCLE: VehicleKind.motorcycle, ObjectClass.BUS: VehicleKind.bus,
        ObjectClass.TRUCK: VehicleKind.truck}
LIGHT_STATE = {LightColor.UNKNOWN: LightState.unknown, LightColor.RED: LightState.red, LightColor.YELLOW: LightState.yellow,
               LightColor.GREEN: LightState.green}


@dataclass(frozen=True)
class PassSpec:
  name: str
  camera: int            # ROAD or WIDE
  lights_crop: int = 0   # side of the square above the horizon; 0 for the whole frame
  vehicles: bool = True
  max_distance: float = 100.0


PASSES = (
  PassSpec("lights_near", ROAD, lights_crop=624, vehicles=False),
  PassSpec("wide", WIDE, max_distance=25.0),   # a wide-camera pixel is half a degree: further out the road camera does better
  PassSpec("lights_far", ROAD, lights_crop=832, vehicles=False),
  PassSpec("road", ROAD, max_distance=100.0),
)


def lights_region(geometry: CameraGeometry, side: int, frame_w: int, frame_h: int) -> Region:
  """A square straight ahead whose bottom sits a little below the horizon, so it is almost all sky and signal heads."""
  vp = geometry.project([1000.0], [0.0], [0.0])[0]   # the horizon straight ahead
  u, v = (frame_w / 2, frame_h * 0.45) if not np.all(np.isfinite(vp)) else (float(vp[0]), float(vp[1]))
  return clamp_region(u - side / 2, v + 0.15 * side - side, side, side, frame_w, frame_h)


class PlanView:
  """What the passes need from the latest modelV2: the planned path's lateral position and our lane's lines, by distance."""

  def __init__(self, model=None):
    self.path_x = self.path_y = np.zeros(0)
    self.left = self.right = None
    if model is None:
      return
    xs, ys = np.array(model.position.x, dtype=float), np.array(model.position.y, dtype=float)
    if xs.size >= 2 and xs.size == ys.size:
      self.path_x, self.path_y = xs, ys
    lines = model.laneLines
    if len(lines) >= 4 and len(model.laneLineProbs) >= 4 and min(model.laneLineProbs[1], model.laneLineProbs[2]) > 0.3:
      lx, ly = np.array(lines[1].x, dtype=float), np.array(lines[1].y, dtype=float)
      rx, ry = np.array(lines[2].x, dtype=float), np.array(lines[2].y, dtype=float)
      if lx.size >= 2 and lx.size == ly.size and rx.size >= 2 and rx.size == ry.size:
        self.left, self.right = (lx, ly), (rx, ry)

  def path_at(self, x: float) -> float:
    return float(np.interp(x, self.path_x, self.path_y)) if self.path_x.size else 0.0

  def lane_lines_at(self, x: float):
    if self.left is None or self.right is None:
      return None
    return float(np.interp(x, *self.left)), float(np.interp(x, *self.right))


class SceneDetection:
  """Everything but the camera and message plumbing, so it can be driven with recorded frames."""

  def __init__(self, detector):
    self.detector = detector
    self.lights = TrafficLightFilter()
    self.tracker = VehicleTracker()
    self._samplers: dict[tuple, NV12Sampler] = {}

  def sampler(self, width: int, height: int, stride: int, uv_offset: int) -> NV12Sampler:
    key = (width, height, stride, uv_offset)
    if key not in self._samplers:
      self._samplers[key] = NV12Sampler(width, height, stride, uv_offset)
    return self._samplers[key]

  def run_pass(self, spec: PassSpec, buf: np.ndarray, sampler: NV12Sampler, geometry: CameraGeometry, plan: PlanView, now: float,
               yaw_rate: float, interval: float = 1.0 / PASS_HZ) -> list[Detection]:
    if spec.lights_crop:
      region = lights_region(geometry, spec.lights_crop, sampler.width, sampler.height)
    else:
      region = Region(0, 0, sampler.width, sampler.height)
    dets = self.detector(sampler.detector_input(buf, region), region)

    observations = []
    focal = float(geometry.intrinsics[0, 0])
    for d in dets:
      if d.cls != ObjectClass.TRAFFIC_LIGHT:
        continue
      lamp = read_lamp(sampler.crop_rgb(buf, d.x0, d.y0, d.x1, d.y1))
      (bearing,), (elevation,) = geometry.angles(0.5 * (d.x0 + d.x1), 0.5 * (d.y0 + d.y1))
      distance = estimate_distance(d.width, focal, float(elevation), geometry.height)
      obs = LightObservation(lamp.color, lamp.confidence * d.score ** 0.5, float(bearing), float(elevation), distance,
                             distance * math.sin(float(bearing)))
      if relevant(obs, plan.path_at(distance * math.cos(float(bearing)))):
        observations.append(obs)
    # a pass that couldn't have seen a light says nothing about the lights
    if spec.lights_crop or observations:
      self.lights.update(now, observations)

    if spec.vehicles:
      vehicles = [d for d in dets if d.cls in VEHICLE_CLASSES]
      width_focal = focal if spec.camera == ROAD else None   # the wide camera's fisheye stretches widths off center
      sightings = sightings_from(vehicles, geometry, spec.camera, spec.max_distance, sampler.width, sampler.height, width_focal)
      self.tracker.update(now, sightings, yaw_rate, revisit=interval * len(PASSES))
    return dets

  def message(self, now: float, plan: PlanView, frame_id: int, pass_name: str, execution_time: float, pass_rate: float, throttled: bool,
              compiled: bool):
    msg = messaging.new_message('sceneDetectionSP', valid=True)
    sd = msg.sceneDetectionSP
    sd.frameId, sd.passName, sd.executionTime = frame_id, pass_name, execution_time
    sd.passRate, sd.throttled, sd.compiled = pass_rate, throttled, compiled

    light = self.lights.reading
    tl = sd.trafficLight
    tl.state = LIGHT_STATE[light.state]
    tl.confidence = light.confidence
    tl.distance = light.distance - RADAR_TO_CAMERA if math.isfinite(light.distance) else math.nan
    tl.bearing = light.bearing
    tl.stateAge = now - light.since if light.state != LightColor.UNKNOWN else 0.0
    tl.headCount = min(light.count, 255)

    tracks = [t for t in self.tracker.tracks if t.confirmed]
    vehicles = sd.init('vehicles', len(tracks))
    for v, t in zip(vehicles, tracks, strict=True):
      v.trackId = t.id
      v.dRel, v.yRel, v.vRel = t.x - RADAR_TO_CAMERA, -t.y, t.vx
      v.lane = lane_index(t.x, t.y, plan.lane_lines_at)
      v.kind = KIND.get(t.cls, VehicleKind.car)
      v.confidence = t.score
      v.age, v.lastSeen = now - t.born, now - t.seen
      v.camera = t.camera
    return msg


class Throttle:
  """Pass interval that backs off while the driving model drops frames or the device is hot, and creeps back after. It also never
  lets a pass take more than DUTY_MAX of the time, whatever the detector costs on this device."""

  def __init__(self):
    self.interval = 1.0 / PASS_HZ

  def update(self, frame_drop_pct: float, hot: bool, pass_time: float = 0.0) -> float:
    if hot or frame_drop_pct > FRAME_DROP_LIMIT:
      self.interval = min(self.interval * BACKOFF, 1.0 / MIN_PASS_HZ)
    else:
      self.interval = max(self.interval * RECOVER, 1.0 / PASS_HZ)
    self.interval = max(self.interval, min(pass_time / DUTY_MAX, 1.0 / MIN_PASS_HZ))
    return self.interval

  @property
  def throttled(self) -> bool:
    return self.interval > 1.05 / PASS_HZ


def geometry_for(sm, camera: int) -> CameraGeometry:
  ds = sm['deviceState']
  sensor = str(sm['narrowRoadCameraState'].sensor) if sm.recv_frame['narrowRoadCameraState'] > 0 else "ar0231"
  cams = DEVICE_CAMERAS.get((str(ds.deviceType), sensor), DEVICE_CAMERAS[("tici", "ar0231")])
  calib = sm['extrinsicsCalibration']
  height = calib.height[0] if len(calib.height) else 0.0
  if camera == WIDE:
    return camera_geometry(cams.wide_road.intrinsics, list(calib.rpyCalib), height, list(calib.wideFromDeviceEuler) or [0.0, 0.0, 0.0])
  return camera_geometry(cams.narrow_road.intrinsics, list(calib.rpyCalib), height)


def main():
  from msgq.visionipc import VisionIpcClient
  os.nice(10)  # the CPU is shared with everything that keeps the car on the road

  sm = messaging.SubMaster(['modelV2', 'extrinsicsCalibration', 'carState', 'deviceState', 'narrowRoadCameraState'])
  pm = messaging.PubMaster(['sceneDetectionSP'])
  clients = {ROAD: VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_NARROW_ROAD, True),
             WIDE: VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, True)}
  while not all(c.connect(False) for c in clients.values()):
    time.sleep(0.1)

  detector = Detector()
  cloudlog.warning(f"scened: detector ready, compiled={detector.compiled}")
  scene = SceneDetection(detector)
  throttle = Throttle()
  rk = Ratekeeper(PASS_HZ, print_delay_threshold=None)
  step = 0
  pass_time = 0.0
  while True:
    sm.update(0)
    hot = sm['deviceState'].thermalStatus >= log.DeviceState.ThermalStatus.overheated
    interval = throttle.update(sm['modelV2'].frameDropPerc if sm.recv_frame['modelV2'] > 0 else 0.0, hot, pass_time)

    spec = PASSES[step % len(PASSES)]
    step += 1
    client = clients[spec.camera]
    buf = client.recv()
    if buf is None:
      continue
    t0 = time.monotonic()
    plan = PlanView(sm['modelV2'] if sm.recv_frame['modelV2'] > 0 else None)
    sampler = scene.sampler(buf.width, buf.height, buf.stride, buf.uv_offset)
    yaw_rate = float(sm['carState'].yawRate) if sm.recv_frame['carState'] > 0 else 0.0
    scene.run_pass(spec, np.frombuffer(buf.data, dtype=np.uint8), sampler, geometry_for(sm, spec.camera), plan, t0, yaw_rate, interval)
    pass_time = time.monotonic() - t0
    pm.send('sceneDetectionSP', scene.message(time.monotonic(), plan, clients[ROAD].frame_id, spec.name, pass_time,
                                              1.0 / interval, throttle.throttled, detector.compiled))

    rk.keep_time()
    if interval > 1.0 / PASS_HZ:
      time.sleep(interval - 1.0 / PASS_HZ)


if __name__ == "__main__":
  main()
