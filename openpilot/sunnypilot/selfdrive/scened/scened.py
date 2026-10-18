#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

scened: camera object detection, publishing sceneDetectionSP.

It looks at one of four regions per pass:
  lights_near  a 624 px square of the road camera above the horizon, straight ahead: signal heads and stop signs out to ~80 m
  wide         the whole wide camera: cars and people beside and just ahead, and lights overhead when stopped at the line
  lights_far   an 832 px square, the same place: heads further out to the sides
  road         the whole road camera: cars, people and stop signs ahead in every lane out to ~100 m
and folds each into the traffic light and stop sign filters and the vehicle and people trackers. Which region comes next depends on
where we are (PassScheduler): in town the lights get most of the looks, on an open highway the cars do.

Distances from the cameras lean on the calibration; whenever the radar has a lead the cameras also see, the ratio of the two corrects
the cameras' distances (RangeCalibrator). Every STATUS_EVERY the process writes how it's doing to the SceneDetectionStatus param for the
Developer panel.

It runs well below the driving model's rate, at low CPU priority, and slows down further whenever the driving model starts dropping
frames or the device runs hot: the driving model always comes first.
"""
import datetime
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from openpilot.cereal import custom, log, messaging
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.sunnypilot.selfdrive.scened.detector import PEOPLE_CLASSES, VEHICLE_CLASSES, Detection, Detector, ObjectClass
from openpilot.sunnypilot.selfdrive.scened.frames import NV12Sampler, Region, clamp_region
from openpilot.sunnypilot.selfdrive.scened.geometry import RADAR_TO_CAMERA, CameraGeometry, camera_geometry
from openpilot.sunnypilot.selfdrive.scened.traffic_lights import Arrow, LightColor, LightObservation, SignObservation, StopSignFilter, \
  TrafficLightFilter, estimate_distance, read_lamp, relevant, sign_distance, sign_relevant, stop_sign_face
from openpilot.sunnypilot.selfdrive.scened.vehicle_tracker import VehicleSighting, VehicleTracker, lane_index, rear_lamp_energy, \
  sightings_from

PASS_HZ = 5.0              # passes per second at most
MIN_PASS_HZ = 1.0
FRAME_DROP_LIMIT = 3.0     # % of driving model frames dropped before scened backs off
BACKOFF, RECOVER = 1.25, 0.97
DUTY_MAX = 0.2             # share of the time scened may spend in passes

# the GPU is shared with the driving model and the driver monitoring model, and work on it can't be preempted: a pass that lands
# on top of them makes them late, and a late driverMonitoringState is a communication issue that disengages. scened watches how
# long each of them takes and gets out of the way when they slow down.
GUARD_SLACK = (1.25, 0.004)   # a run longer than its usual time * 1.25 + 4 ms was slowed down by something
GUARD_RISE = 0.01             # the usual time follows a faster run at once and a slower one only this share per run
GUARD_WINDOW = 2.0            # s of runs to judge from
GUARD_SLOW_SHARE = 0.15       # share of slowed runs in the window that makes scened back off
DM_BUDGET = 0.038             # s, a driver monitoring run longer than this leaves no margin at 20 Hz, whatever its usual time
GUARD_PAUSE_AFTER = 3.0       # s of backing off before scened stops altogether ...
GUARD_PAUSE = 20.0            # ... for this long
ROAD, WIDE = 0, 1
TOWN_SPEED = 20.0          # m/s, below this lights and signs are likely
LIGHTS_RECENT = 20.0       # s, a light seen this recently means more may come

LightState = custom.SceneDetectionSP.LightState
VehicleKind = custom.SceneDetectionSP.VehicleKind
KIND = {ObjectClass.CAR: VehicleKind.car, ObjectClass.MOTORCYCLE: VehicleKind.motorcycle, ObjectClass.BUS: VehicleKind.bus,
        ObjectClass.TRUCK: VehicleKind.truck, ObjectClass.PERSON: VehicleKind.pedestrian, ObjectClass.BICYCLE: VehicleKind.cyclist}
ArrowKind = custom.SceneDetectionSP.Arrow
ARROW = {Arrow.NONE: ArrowKind.none, Arrow.LEFT: ArrowKind.left, Arrow.RIGHT: ArrowKind.right, Arrow.STRAIGHT: ArrowKind.straight,
         Arrow.OTHER: ArrowKind.other}

# people: tighter gates than cars, they're small and slow
PEOPLE_GATE_X, PEOPLE_GATE_Y, PEOPLE_COAST = (1.5, 0.06), (0.9, 0.02), 1.0
PEOPLE_MAX_DISTANCE = 50.0

# range calibration from the radar lead
CAL_LEAD_RANGE = (8.0, 80.0)     # m, radar lead distances to learn from
CAL_GATE = (0.3, 1.6)            # a sighting within 30 % of the lead's distance and 1.6 m of its lateral position is the lead
CAL_RATE = 0.04                  # per match, in log space
CAL_LIMITS = (0.75, 1.33)
CAL_MIN_MATCHES = 5              # matches before the correction is used

# turn direction at a light, from the planned path
TURN_LOOKAHEAD = 25.0            # m past the light
TURN_OFFSET = 4.0                # m of sideways path there that means a turn

MESSAGE_PREDICT_MAX = 1.0       # s a message brings what it reports forward at most
STATUS_EVERY = 30.0              # s between status writes
STATUS_EVENTS = 20
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
  """What the passes need from the latest modelV2: the planned path's lateral position, our lane's lines, and whether it plans to stop."""

  def __init__(self, model=None):
    self.path_x = self.path_y = np.zeros(0)
    self.left = self.right = None
    self.stop_planned = False
    if model is None:
      return
    xs, ys = np.array(model.position.x, dtype=float), np.array(model.position.y, dtype=float)
    if xs.size >= 2 and xs.size == ys.size:
      self.path_x, self.path_y = xs, ys
    vs = np.array(model.velocity.x, dtype=float)
    self.stop_planned = bool(vs.size and vs.min() < 0.5 and vs[0] > 1.0)
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

  def turn_at(self, x: float) -> int:
    """-1 when the path turns left past x, 1 right, 0 straight on: which arrows at a light there are ours."""
    if self.path_x.size < 2 or x + TURN_LOOKAHEAD > self.path_x[-1]:
      return 0
    offset = self.path_at(x + TURN_LOOKAHEAD) - self.path_at(x)
    return -1 if offset < -TURN_OFFSET else 1 if offset > TURN_OFFSET else 0


@dataclass
class Context:
  """What a pass needs to know about us besides the picture."""
  now: float
  yaw_rate: float = 0.0
  v_ego: float = 0.0
  revisit: float = 4.0 / PASS_HZ          # s until the same camera looks at the same place again
  lead: tuple[float, float] | None = None  # the radar lead (distance from the camera, m right) when the radar itself has it


class RangeCalibrator:
  """Learns how far off a camera's distances are from the radar: whenever the radar has a lead and a sighting sits on it, the ratio
  of their distances moves the correction a little (in log space, so too far and too near count alike). Pitch errors in the
  calibration show up as exactly this kind of scale error."""

  def __init__(self):
    self.log_scale = 0.0
    self.matches = 0

  @property
  def scale(self) -> float:
    return math.exp(self.log_scale) if self.matches >= CAL_MIN_MATCHES else 1.0

  def learn(self, sightings: list[VehicleSighting], lead: tuple[float, float] | None) -> None:
    if lead is None or not CAL_LEAD_RANGE[0] <= lead[0] <= CAL_LEAD_RANGE[1]:
      return
    lead_x, lead_y = lead
    scale = math.exp(self.log_scale)
    near = [s for s in sightings if abs(s.x * scale - lead_x) < CAL_GATE[0] * lead_x and abs(s.y * scale - lead_y) < CAL_GATE[1]]
    if not near:
      return
    s = min(near, key=lambda s: abs(s.x * scale - lead_x))
    target = math.log(lead_x / s.x)
    self.log_scale += CAL_RATE * (target - self.log_scale)
    self.log_scale = float(np.clip(self.log_scale, math.log(CAL_LIMITS[0]), math.log(CAL_LIMITS[1])))
    self.matches += 1

  def apply(self, sightings: list[VehicleSighting]) -> list[VehicleSighting]:
    k = self.scale
    if k == 1.0:
      return sightings
    for s in sightings:
      s.x, s.y = s.x * k, s.y * k
    return sightings


class SceneDetection:
  """Everything but the camera and message plumbing, so it can be driven with recorded frames."""

  def __init__(self, detector):
    self.detector = detector
    self.lights = TrafficLightFilter()
    self.signs = StopSignFilter()
    self.tracker = VehicleTracker()
    self.people = VehicleTracker(PEOPLE_GATE_X, PEOPLE_GATE_Y, PEOPLE_COAST)
    self.calibration = {ROAD: RangeCalibrator(), WIDE: RangeCalibrator()}
    self._samplers: dict[tuple, NV12Sampler] = {}
    self._v_ego = 0.0

  def sampler(self, width: int, height: int, stride: int, uv_offset: int) -> NV12Sampler:
    key = (width, height, stride, uv_offset)
    if key not in self._samplers:
      self._samplers[key] = NV12Sampler(width, height, stride, uv_offset)
    return self._samplers[key]

  def run_pass(self, spec: PassSpec, buf: np.ndarray, sampler: NV12Sampler, geometry: CameraGeometry, plan: PlanView,
               ctx: Context) -> list[Detection]:
    if spec.lights_crop:
      region = lights_region(geometry, spec.lights_crop, sampler.width, sampler.height)
    else:
      region = Region(0, 0, sampler.width, sampler.height)
    dets = self.detector(sampler.detector_input(buf, region), region)
    focal = float(geometry.intrinsics[0, 0])
    road = spec.camera == ROAD

    # signal heads, read for which lamp is lit and which way an arrow points
    observations, turn = [], 0
    for d in dets:
      if d.cls != ObjectClass.TRAFFIC_LIGHT:
        continue
      lamp = read_lamp(sampler.crop_rgb(buf, d.x0, d.y0, d.x1, d.y1))
      (bearing,), (elevation,) = geometry.angles(0.5 * (d.x0 + d.x1), 0.5 * (d.y0 + d.y1))
      distance = estimate_distance(d.width, focal, float(elevation), geometry.height)
      along = distance * math.cos(float(bearing))
      obs = LightObservation(lamp.color, lamp.confidence * d.score ** 0.5, float(bearing), float(elevation), distance,
                             distance * math.sin(float(bearing)), lamp.arrow)
      if relevant(obs, plan.path_at(along)):
        observations.append(obs)
        turn = plan.turn_at(along)
    # a pass that couldn't have seen a light says nothing about the lights
    self._v_ego = ctx.v_ego
    if spec.lights_crop or observations:
      self.lights.update(ctx.now, observations, turn, ctx.v_ego)

    # stop signs, from the road camera (the wide camera's fisheye bends their size)
    if road:
      signs = []
      for d in dets:
        if d.cls != ObjectClass.STOP_SIGN:
          continue
        if not stop_sign_face(sampler.crop_rgb(buf, d.x0, d.y0, d.x1, d.y1)):
          continue
        (bearing,), _ = geometry.angles(0.5 * (d.x0 + d.x1), 0.5 * (d.y0 + d.y1))
        distance = sign_distance(d.width, focal)
        sign_obs = SignObservation(d.score, distance, distance * math.sin(float(bearing)))
        if sign_relevant(sign_obs, plan.path_at(distance * math.cos(float(bearing)))):
          signs.append(sign_obs)
      self.signs.update(ctx.now, signs, ctx.v_ego)

    if spec.vehicles:
      cal = self.calibration[spec.camera]
      vehicles = [d for d in dets if d.cls in VEHICLE_CLASSES]
      width_focal = focal if road else None   # the wide camera's fisheye stretches widths off center
      lamp_of = (lambda d: rear_lamp_energy(sampler.crop_rgb(buf, d.x0, d.y0 + 0.1 * d.height, d.x1, d.y1 - 0.1 * d.height, max_side=64))) \
        if road else None
      sightings = sightings_from(vehicles, geometry, spec.camera, spec.max_distance, sampler.width, sampler.height, width_focal, lamp_of)
      cal.learn(sightings, ctx.lead)
      self.tracker.update(ctx.now, cal.apply(sightings), ctx.yaw_rate, revisit=ctx.revisit)
      self.tracker.update_merging(plan.lane_lines_at)

      people = [d for d in dets if d.cls in PEOPLE_CLASSES]
      walkers = sightings_from(people, geometry, spec.camera, min(spec.max_distance, PEOPLE_MAX_DISTANCE), sampler.width, sampler.height)
      self.people.update(ctx.now, cal.apply(walkers), ctx.yaw_rate, revisit=ctx.revisit)
    return dets

  def message(self, now: float, plan: PlanView, frame_id: int, pass_name: str, execution_time: float, pass_rate: float, throttled: bool,
              compiled: bool, lights_share: float = 0.0):
    msg = messaging.new_message('sceneDetectionSP', valid=True)
    sd = msg.sceneDetectionSP
    sd.frameId, sd.passName, sd.executionTime = frame_id, pass_name, execution_time
    sd.passRate, sd.throttled, sd.compiled = pass_rate, throttled, compiled
    sd.rangeScale, sd.rangeScaleWide = self.calibration[ROAD].scale, self.calibration[WIDE].scale
    sd.lightsShare = lights_share

    light = self.lights.reading
    tl = sd.trafficLight
    tl.state = LIGHT_STATE[light.state]
    tl.confidence = light.confidence
    # distances as of now, not as of the pass that last saw them: the UI counts down from here, so they don't jump at each look
    tl.distance = self._closer(light.distance, self.lights.updated_at, now) - RADAR_TO_CAMERA
    tl.bearing = light.bearing
    tl.stateAge = now - light.since if light.state != LightColor.UNKNOWN else 0.0
    tl.headCount = min(light.count, 255)
    tl.flashing = light.flashing
    tl.arrow = ARROW[light.arrow]

    sign = self.signs.reading
    sd.stopSign.present = sign.present
    sd.stopSign.distance = self._closer(sign.distance, self.signs.updated_at, now) - RADAR_TO_CAMERA if sign.present else math.nan
    sd.stopSign.confidence = sign.confidence

    for field_name, tracker in (('vehicles', self.tracker), ('people', self.people)):
      tracks = [t for t in tracker.tracks if t.confirmed]
      out = sd.init(field_name, len(tracks))
      ahead = min(max(now - tracker.updated_at, 0.0), MESSAGE_PREDICT_MAX) if tracker.updated_at is not None else 0.0
      for v, t in zip(out, tracks, strict=True):
        v.trackId = t.id
        v.dRel, v.yRel, v.vRel = t.x + t.vx * ahead - RADAR_TO_CAMERA, -(t.y + t.vy * ahead), t.vx
        v.lateralSpeed = -t.vy
        v.lane = lane_index(t.x, t.y, plan.lane_lines_at)
        v.kind = KIND.get(t.cls, VehicleKind.car)
        v.confidence = t.score
        v.age, v.lastSeen = now - t.born, now - t.seen
        v.camera = t.camera
        v.braking, v.merging = t.braking, t.merging
    return msg

  def _closer(self, distance: float, since: float | None, now: float) -> float:
    """A distance read at since, brought forward to now at our speed."""
    if not math.isfinite(distance) or since is None:
      return distance
    return max(distance - self._v_ego * min(max(now - since, 0.0), MESSAGE_PREDICT_MAX), 0.0)


class PassScheduler:
  """Picks the next region by weight, deficit round robin: each pass earns its weight every step and the richest goes next, so the
  shares follow the weights exactly and nothing starves. Lights and signs get most of the looks around town, when the model plans a
  stop or a light was seen lately; on an open highway the cars do."""

  def __init__(self, passes=PASSES):
    self.passes = passes
    self.credit = dict.fromkeys((p.name for p in passes), 0.0)
    self.weights = dict.fromkeys((p.name for p in passes), 1.0)

  def set_context(self, v_ego: float, stop_planned: bool, lights_recent: bool) -> None:
    town = v_ego < TOWN_SPEED or stop_planned or lights_recent
    stopped = v_ego < 1.0
    for p in self.passes:
      if p.lights_crop:
        w = (2.0 if p.lights_crop < 700 else 1.5) if town else (0.4 if p.lights_crop < 700 else 0.3)
      elif p.camera == WIDE:
        w = 0.7 if stopped else 1.5
      else:
        w = 1.0 if stopped else 2.0
      self.weights[p.name] = w

  def next(self) -> PassSpec:
    for p in self.passes:
      self.credit[p.name] += self.weights[p.name]
    best = max(self.passes, key=lambda p: self.credit[p.name])
    self.credit[best.name] -= sum(self.weights.values())
    return best

  @property
  def lights_share(self) -> float:
    total = sum(self.weights.values())
    return sum(self.weights[p.name] for p in self.passes if p.lights_crop) / total

  def revisit(self, spec: PassSpec, interval: float) -> float:
    """s until a pass of the same camera comes round again at these weights."""
    total = sum(self.weights.values())
    same = sum(self.weights[p.name] for p in self.passes if p.camera == spec.camera and p.vehicles == spec.vehicles)
    return interval * total / max(same, 1e-3)


@dataclass
class StatusLog:
  """What the Developer panel shows: how fast scened is running, how often it was held back, what it last saw."""
  started: float
  passes: dict = field(default_factory=dict)        # name -> [count, total s]
  throttled_passes: int = 0
  total_passes: int = 0
  events: deque = field(default_factory=lambda: deque(maxlen=STATUS_EVENTS))
  max_vehicles: int = 0
  max_people: int = 0
  stop_signs: int = 0
  last_write: float = 0.0
  guard: 'ModelGuard | None' = None   # the GPU guard, for the yields and the models' usual times
  _light: tuple = (LightColor.UNKNOWN, False, Arrow.NONE)
  _sign: bool = False

  def record(self, now: float, spec: PassSpec, seconds: float, throttled: bool, scene: SceneDetection) -> None:
    count, total = self.passes.get(spec.name, (0, 0.0))
    self.passes[spec.name] = (count + 1, total + seconds)
    self.total_passes += 1
    self.throttled_passes += int(throttled)
    self.max_vehicles = max(self.max_vehicles, sum(t.confirmed for t in scene.tracker.tracks))
    self.max_people = max(self.max_people, sum(t.confirmed for t in scene.people.tracks))
    light = scene.lights.reading
    key = (light.state, light.flashing, light.arrow)
    if key != self._light:
      self._light = key
      self._event(f"light {('flashing ' if light.flashing else '') + light.state.name.lower()}"
                  + (f" ({light.arrow.name.lower()} arrow)" if light.arrow != Arrow.NONE else "")
                  + (f" at {light.distance:.0f} m" if math.isfinite(light.distance) else ""))
    if scene.signs.reading.present != self._sign:
      self._sign = scene.signs.reading.present
      if self._sign:
        self.stop_signs += 1
        self._event(f"stop sign at {scene.signs.reading.distance:.0f} m")

  def _event(self, text: str) -> None:
    self.events.appendleft({"time": datetime.datetime.now().strftime("%H:%M:%S"), "event": text})

  def snapshot(self, now: float, scene: SceneDetection, compiled: bool, interval: float, lights_share: float) -> dict:
    return {
      "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      "uptime_s": round(now - self.started),
      "compiled": compiled,
      "pass_rate": round(1.0 / interval, 2),
      "throttled_pct": round(100.0 * self.throttled_passes / max(self.total_passes, 1), 1),
      "lights_share_pct": round(100.0 * lights_share),
      "passes": {name: {"count": c, "avg_ms": round(1000.0 * t / max(c, 1), 1)} for name, (c, t) in self.passes.items()},
      "range_scale": round(scene.calibration[ROAD].scale, 3),
      "range_scale_wide": round(scene.calibration[WIDE].scale, 3),
      "calibration_matches": scene.calibration[ROAD].matches,
      "max_vehicles": self.max_vehicles,
      "max_people": self.max_people,
      "stop_signs": self.stop_signs,
      "events": list(self.events),
      "model_ms": round(1000.0 * self.guard.usual.get('modelV2', 0.0), 1) if self.guard else 0.0,
      "dm_ms": round(1000.0 * self.guard.usual.get('driverStateV2', 0.0), 1) if self.guard else 0.0,
      "yields": self.guard.pauses if self.guard else 0,
    }


class ModelGuard:
  """Watches the driving model's and the driver monitoring model's run times for the slowdowns scened's GPU work causes. Each
  model's usual time is the floor of its recent runs (scened is idle for most of them); a run well over it was held up. slow is
  true while too many recent runs were; after GUARD_PAUSE_AFTER of that, paused stays true for GUARD_PAUSE. It keeps watching while
  paused, so if the models are still slow when the pause ends (scened was not the cause, but there is no room for it either) it
  pauses again at once, and comes back only once they have been quick for a while."""

  def __init__(self):
    self.usual: dict[str, float] = {}
    self._runs: deque[tuple[float, str, bool]] = deque()
    self._slow_since: float | None = None
    self.paused_until = -math.inf
    self.pauses = 0
    self.slow = False

  def sample(self, now: float, model: str, run_time: float, budget: float = math.inf) -> None:
    if not math.isfinite(run_time) or run_time <= 0.0:
      return
    usual = self.usual.get(model, run_time)
    held_up = run_time > usual * GUARD_SLACK[0] + GUARD_SLACK[1] or run_time > budget
    self.usual[model] = run_time if run_time < usual else usual + GUARD_RISE * (run_time - usual)
    self._runs.append((now, model, held_up))

  def update(self, now: float) -> None:
    while self._runs and self._runs[0][0] < now - GUARD_WINDOW:
      self._runs.popleft()
    runs: dict[str, list[bool]] = {}
    for _, model, held_up in self._runs:
      runs.setdefault(model, []).append(held_up)
    self.slow = any(sum(held) / len(held) > GUARD_SLOW_SHARE for held in runs.values())   # either model, judged on its own runs
    if not self.slow:
      self._slow_since = None
    elif self._slow_since is None:
      self._slow_since = now
    elif now - self._slow_since > GUARD_PAUSE_AFTER and not self.paused(now):
      self.paused_until, self._slow_since = now + GUARD_PAUSE, None
      self.pauses += 1
      self._runs.clear()   # judge afresh once back

  def paused(self, now: float) -> bool:
    return now < self.paused_until


class Throttle:
  """Pass interval that backs off while the driving model drops frames, the models sharing the GPU are slowed down or the device is
  hot, and creeps back after. It also never lets a pass take more than DUTY_MAX of the time, whatever the detector costs on this
  device."""

  def __init__(self):
    self.interval = 1.0 / PASS_HZ

  def update(self, frame_drop_pct: float, hot: bool, pass_time: float = 0.0, models_slow: bool = False) -> float:
    if hot or frame_drop_pct > FRAME_DROP_LIMIT or models_slow:
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

  params = Params()
  # paced by the driving model: each pass starts just after modeld has published, in the gap before its next frame
  sm = messaging.SubMaster(['modelV2', 'driverStateV2', 'extrinsicsCalibration', 'carState', 'deviceState', 'narrowRoadCameraState',
                            'radarState'], poll='modelV2')
  pm = messaging.PubMaster(['sceneDetectionSP'])
  clients = {ROAD: VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_NARROW_ROAD, True),
             WIDE: VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, True)}
  while not all(c.connect(False) for c in clients.values()):
    time.sleep(0.1)

  detector = Detector()
  cloudlog.warning(f"scened: detector ready, compiled={detector.compiled}")
  scene = SceneDetection(detector)
  throttle = Throttle()
  guard = ModelGuard()
  scheduler = PassScheduler()
  status = StatusLog(time.monotonic(), guard=guard)
  pass_time, last_pass = 0.0, -math.inf
  while True:
    sm.update(200)
    now = time.monotonic()
    if sm.updated['modelV2']:
      guard.sample(now, 'modelV2', sm['modelV2'].modelExecutionTime)
    if sm.updated['driverStateV2']:
      guard.sample(now, 'driverStateV2', sm['driverStateV2'].modelExecutionTime, DM_BUDGET)
    guard.update(now)
    hot = sm['deviceState'].thermalStatus >= log.DeviceState.ThermalStatus.overheated
    interval = throttle.update(sm['modelV2'].frameDropPerc if sm.recv_frame['modelV2'] > 0 else 0.0, hot, pass_time, guard.slow)
    if guard.paused(now) or not sm.updated['modelV2'] or now - last_pass < interval:
      continue

    plan = PlanView(sm['modelV2'] if sm.recv_frame['modelV2'] > 0 else None)
    cs = sm['carState']
    v_ego = float(cs.vEgo) if sm.recv_frame['carState'] > 0 else 0.0
    lights_recent = scene.lights.reading.state != LightColor.UNKNOWN or now - scene.lights.last_seen < LIGHTS_RECENT or scene.signs.reading.present
    scheduler.set_context(v_ego, plan.stop_planned, lights_recent)
    spec = scheduler.next()
    buf = clients[spec.camera].recv()
    if buf is None:
      continue
    t0 = last_pass = time.monotonic()
    lead = sm['radarState'].leadOne if sm.recv_frame['radarState'] > 0 else None
    radar_lead = (lead.dRel + RADAR_TO_CAMERA, -lead.yRel) if lead is not None and lead.present and lead.radar else None
    ctx = Context(t0, float(cs.yawRate) if sm.recv_frame['carState'] > 0 else 0.0, v_ego, scheduler.revisit(spec, interval), radar_lead)
    sampler = scene.sampler(buf.width, buf.height, buf.stride, buf.uv_offset)
    scene.run_pass(spec, np.frombuffer(buf.data, dtype=np.uint8), sampler, geometry_for(sm, spec.camera), plan, ctx)
    pass_time = time.monotonic() - t0
    pm.send('sceneDetectionSP', scene.message(time.monotonic(), plan, clients[ROAD].frame_id, spec.name, pass_time,
                                              1.0 / interval, throttle.throttled, detector.compiled, scheduler.lights_share))

    status.record(t0, spec, pass_time, throttle.throttled, scene)
    if t0 - status.last_write > STATUS_EVERY:
      status.last_write = t0
      params.put("SceneDetectionStatus", status.snapshot(t0, scene, detector.compiled, interval, scheduler.lights_share))


if __name__ == "__main__":
  main()
