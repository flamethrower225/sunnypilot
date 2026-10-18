"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Cars around us from the cameras. Every detected vehicle is put on the road where its wheels touch it (the bottom middle of its box
through the camera geometry), then tracked: each track is predicted forward with its own relative speed and with our own turning,
matched to the nearest new sighting it could plausibly be, and corrected toward it (an alpha-beta filter, position and relative
velocity). A track shows after a second sighting and is dropped once nothing has seen it for a while, longer beside us, where only
the wide camera reaches and so it is looked at less often.

Positions are in the calibrated frame, from the camera: x forward, y right, in m. The published tracks use the radar convention
instead (distance from the front bumper, lateral left positive), so the UI can treat them like radar tracks.
"""
import math
from dataclasses import dataclass

import numpy as np

from openpilot.sunnypilot.selfdrive.scened.detector import Detection
from openpilot.sunnypilot.selfdrive.scened.geometry import CameraGeometry
from openpilot.sunnypilot.selfdrive.scened.traffic_lights import rgb_to_hsv

ALPHA = 0.55              # how far a track moves toward its new sighting
BETA = 0.2                # and how much of the leftover goes into its speed
GATE_X = (2.5, 0.14)      # m: a sighting matches a track within max(2.5 m, 14 % of the distance) along the road
GATE_Y = (1.3, 0.035)     # m: and max(1.3 m, 3.5 %) across it
CONFIRM_HITS = 2
COAST = 1.2               # s a track lives on without a sighting
COAST_BESIDE = 2.4        # s when it's beside us (x under BESIDE_X), seen only by the wide camera
BESIDE_X = 12.0           # m
REVISITS_TO_COAST = 2.2   # a track survives at least this many of its camera's revisits
MAX_TRACKS = 24
MAX_SPEED_REL = 40.0      # m/s, a speed estimate beyond this is a bad match, not a car
LANE_WIDTH = 3.6          # m, used when the model has no lane lines
MIN_BOX_PX = 10.0         # boxes smaller than this are too far to place
VEHICLE_WIDTH = {2: 1.8, 3: 0.8, 5: 2.55, 7: 2.4}   # m, by COCO class: car, motorcycle, bus, truck
REAR_ASPECT = (0.7, 1.8)  # width / height of a box that shows a vehicle's back (or front), not its side
WIDTH_BLEND_FROM = 12.0   # m, nearer than this the ground point alone is better
MERGE = 0.45              # two tracks closer than this share of the gate are one car seen twice
OCCLUSION_OVERLAP = 0.3   # share of a box's width a nearer box has to cover for its bottom to be hidden
OCCLUSION_MARGIN = 8.0    # px, the nearer box has to reach at least this much lower
TWIN_IOU = 0.6            # a box overlapping this much is the same car boxed twice, not one in front

# brake lights
BRAKE_RATIO = 1.8         # the back glows this many times its brakes-off level
BRAKE_JUMP = 0.012        # and at least this much more of it
BRAKE_MIN = 0.02          # and at least this share of it glows at all
BRAKE_OFF_RISE = 0.05     # how fast the brakes-off level follows a brighter back while not braking
BRAKE_VOTES = 2           # looks in a row
BRAKE_STALE = 1.5         # s without a look at its back before braking is forgotten
BRAKE_LAMP_RANGE = 60.0   # m, further out the lamps are a few pixels
LAMP_HUE = (18.0, 335.0)  # degrees: red is below the first or above the second
LAMP_SATURATION = 0.5
LAMP_VALUE = 0.5
LAMP_ABOVE_MEDIAN = 0.15

# merging
MERGE_RANGE = (-5.0, 40.0)   # m from the camera
MERGE_MIN_SPEED = 0.35       # m/s toward our lane
MERGE_MAX_GAP = 2.5          # m from our lane line
MERGE_TIME = 3.0             # s to reach it
MERGE_VOTES = 2


@dataclass
class VehicleSighting:
  x: float          # m forward of the camera
  y: float          # m right
  score: float
  cls: int
  camera: int       # 0 road, 1 wide
  lamp: float = math.nan   # how much of its back glows brake-light red (rear_lamp_energy), when it's seen from behind


@dataclass
class VehicleTrack:
  id: int
  x: float
  y: float
  vx: float = 0.0     # m/s relative to us
  vy: float = 0.0
  hits: int = 1
  score: float = 0.0
  cls: int = 2
  camera: int = 0
  born: float = 0.0
  seen: float = 0.0   # monotonic s of the last sighting
  lamp_fast: float = math.nan   # recent rear lamp energy
  lamp_off: float = math.nan    # its level with the brakes off: follows drops at once, rises slowly
  lamp_seen: float = 0.0
  brake_votes: int = 0
  braking: bool = False
  merge_votes: int = 0
  merging: bool = False

  @property
  def confirmed(self) -> bool:
    return self.hits >= CONFIRM_HITS


def iou(a: Detection, b: Detection) -> float:
  w = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
  h = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
  inter = w * h
  return inter / max(a.width * a.height + b.width * b.height - inter, 1e-6)


def occluded(d: Detection, others: list[Detection]) -> bool:
  """Whether the bottom of the box is hidden behind another vehicle: one that reaches lower in the frame (so it's nearer) and
  covers the box's bottom edge."""
  for o in others:
    if o is d or o.y1 <= d.y1 + OCCLUSION_MARGIN or not (o.y0 < d.y1 < o.y1) or iou(o, d) > TWIN_IOU:
      continue
    overlap = min(d.x1, o.x1) - max(d.x0, o.x0)
    if overlap > OCCLUSION_OVERLAP * d.width:
      return True
  return False


def width_range(d: Detection, focal: float, max_aspect: float = REAR_ASPECT[1]) -> float | None:
  """Distance (m) along the ray from how wide a vehicle seen from behind looks against how wide such vehicles are. None for a box
  that shows the side of the car too (then its width isn't the car's)."""
  aspect = d.width / max(d.height, 1.0)
  if not REAR_ASPECT[0] <= aspect <= max_aspect:
    return None
  return focal * VEHICLE_WIDTH.get(int(d.cls), 1.8) / max(d.width, 1.0)


def rear_lamp_energy(rgb: np.ndarray) -> float:
  """Share of a car's back (RGB 0-255 crop of its box) glowing lamp red: saturated red pixels clearly brighter than the rest of
  the back. Brake lights raise it a lot; tail lights, red paint and reflections only a little and steadily, which the off level in
  update_brake_lights absorbs."""
  if rgb.size == 0 or min(rgb.shape[:2]) < 4:
    return math.nan
  h, s, v = rgb_to_hsv(rgb)
  red = ((h < LAMP_HUE[0]) | (h > LAMP_HUE[1])) & (s > LAMP_SATURATION) & (v > max(LAMP_VALUE, float(np.median(v)) + LAMP_ABOVE_MEDIAN))
  return float(red.mean())


def sightings_from(dets: list[Detection], geometry: CameraGeometry, camera: int, max_distance: float,
                   frame_w: int, frame_h: int, width_focal: float | None = None, lamp_of=None) -> list[VehicleSighting]:
  """Places detected vehicles on the road where the bottom of the box meets it. Further out that leans on the calibration's pitch
  (a tenth of a degree is meters at 50 m), so when width_focal is given (a camera without much distortion) the distance is
  averaged with the one the car's width gives. A box cut off at the side is placed from its visible end; one too small or that
  never meets the road is skipped. lamp_of(detection) reads the brake lights of a car seen from behind, when given."""
  out = []
  for d in dets:
    if d.width < MIN_BOX_PX or d.height < MIN_BOX_PX:
      continue
    cut = d.x0 <= 1 or d.x1 >= frame_w - 1
    if occluded(d, dets):
      # its wheels are hidden behind a nearer car, so the bottom of its box isn't on the road: only its width can place it
      r = width_range(d, width_focal, max_aspect=math.inf) if width_focal and not cut else None   # its height is cut short too
      if r is None:
        continue
      (ray,) = geometry.rays(0.5 * (d.x0 + d.x1), 0.5 * (d.y0 + d.y1))
      x, y = ray[0] * r, ray[1] * r
      if 1.0 < x <= max_distance:
        out.append(VehicleSighting(float(x), float(y), d.score, int(d.cls), camera))
      continue
    u, v = 0.5 * (d.x0 + d.x1), min(d.y1, frame_h - 1)
    if d.x0 <= 1 and d.x1 < frame_w * 0.5:   # cut off on the left: the visible part's middle is not the car's middle
      u = d.x1 - 0.5 * min(d.width, d.height * 1.6)
    elif d.x1 >= frame_w - 1 and d.x0 > frame_w * 0.5:
      u = d.x0 + 0.5 * min(d.width, d.height * 1.6)
    x, y = geometry.ground(u, v)[0]
    if not math.isfinite(x) or x <= 1.0:
      continue
    ground = math.hypot(x, y)
    by_width = width_range(d, width_focal) if width_focal and not cut else None
    if by_width is not None and ground > WIDTH_BLEND_FROM:
      scale = 0.5 * (ground + by_width) / ground
      x, y = x * scale, y * scale
    if x > max_distance:
      continue
    lamp = math.nan
    if lamp_of is not None and not cut and x <= BRAKE_LAMP_RANGE and REAR_ASPECT[0] <= d.width / max(d.height, 1.0) <= REAR_ASPECT[1]:
      lamp = lamp_of(d)
    out.append(VehicleSighting(float(x), float(y), d.score, int(d.cls), camera, lamp))
  return out


def ego_motion(x: float, y: float, dtheta: float) -> tuple[float, float]:
  """A point (or vector) in our frame after we turn by dtheta (rad, left positive, like carState.yawRate). y is right positive, so
  turning left swings everything ahead to the right."""
  c, s = math.cos(dtheta), math.sin(dtheta)
  return x * c - y * s, x * s + y * c


def update_brake_lights(t: VehicleTrack, lamp: float, now: float) -> None:
  """Brake lights: the back of the car glowing clearly more red than it does with the brakes off. The off level adapts (tail lights
  at night are always on) but never while braking, and it drops at once when the lamps dim, so a car first seen braking at a light
  shows as braking once it lets go and brakes again."""
  if not math.isfinite(lamp):
    if now - t.lamp_seen > BRAKE_STALE:
      t.braking, t.brake_votes = False, 0
    return
  t.lamp_seen = now
  t.lamp_fast = lamp if not math.isfinite(t.lamp_fast) else 0.5 * t.lamp_fast + 0.5 * lamp
  if not math.isfinite(t.lamp_off):
    t.lamp_off = lamp
  elif lamp < t.lamp_off:
    t.lamp_off = lamp
  elif not t.braking:
    t.lamp_off += BRAKE_OFF_RISE * (lamp - t.lamp_off)
  lit = t.lamp_fast > max(BRAKE_RATIO * t.lamp_off, t.lamp_off + BRAKE_JUMP, BRAKE_MIN)
  t.brake_votes = min(t.brake_votes + 1, BRAKE_VOTES) if lit else max(t.brake_votes - 1, 0)
  if t.brake_votes >= BRAKE_VOTES:
    t.braking = True
  elif t.brake_votes == 0:
    t.braking = False


class VehicleTracker:
  """gate_x and gate_y are the matching gates (m, and share of the distance); people get tighter ones than cars."""

  def __init__(self, gate_x: tuple[float, float] = GATE_X, gate_y: tuple[float, float] = GATE_Y, coast: float = COAST):
    self.tracks: list[VehicleTrack] = []
    self._next_id = 1
    self._t: float | None = None
    self.gate_x, self.gate_y, self.coast = gate_x, gate_y, coast

  @property
  def updated_at(self) -> float | None:
    """Monotonic s the tracks' positions are as of."""
    return self._t

  def predict(self, now: float, yaw_rate: float) -> None:
    dt = 0.0 if self._t is None else max(now - self._t, 0.0)
    self._t = now
    if dt <= 0 or not self.tracks:
      return
    for t in self.tracks:
      t.x += t.vx * dt
      t.y += t.vy * dt
      t.x, t.y = ego_motion(t.x, t.y, yaw_rate * dt)
      t.vx, t.vy = ego_motion(t.vx, t.vy, yaw_rate * dt)

  def update(self, now: float, sightings: list[VehicleSighting], yaw_rate: float = 0.0, revisit: float = 0.0) -> list[VehicleTrack]:
    """Takes one camera pass's sightings. Returns the confirmed tracks. revisit is how long until a camera looks at the same place
    again; a track outlives at least a couple of those, so a slowed-down scened doesn't drop cars between looks."""
    last = self._t
    self.predict(now, yaw_rate)
    dt = 0.0 if last is None else max(now - last, 1e-3)

    # greedy matching, closest pairs first: cheap and good enough at a few dozen objects
    pairs = []
    for i, t in enumerate(self.tracks):
      gx, gy = max(self.gate_x[0], self.gate_x[1] * t.x), max(self.gate_y[0], self.gate_y[1] * t.x)
      for j, s in enumerate(sightings):
        cost = math.hypot((s.x - t.x) / gx, (s.y - t.y) / gy)
        if cost < 1.0:
          pairs.append((cost, i, j))
    used_t, used_s = set(), set()
    for _, i, j in sorted(pairs):
      if i in used_t or j in used_s:
        continue
      used_t.add(i)
      used_s.add(j)
      self._correct(self.tracks[i], sightings[j], now, dt)

    for j, s in enumerate(sightings):
      if j not in used_s and len(self.tracks) < MAX_TRACKS:
        track = VehicleTrack(self._next_id, s.x, s.y, score=s.score, cls=s.cls, camera=s.camera, born=now, seen=now)
        update_brake_lights(track, s.lamp, now)
        self.tracks.append(track)
        self._next_id += 1

    floor = REVISITS_TO_COAST * revisit
    self.tracks = [t for t in self.tracks if now - t.seen <= max(COAST_BESIDE if t.x < BESIDE_X else self.coast, floor)]
    self._merge()
    return [t for t in self.tracks if t.confirmed]

  def _merge(self) -> None:
    """Two overlapping boxes for one car (or the same car from both cameras) end up as neighboring tracks; keep the older one."""
    self.tracks.sort(key=lambda t: t.born)
    kept: list[VehicleTrack] = []
    for t in self.tracks:
      twin = next((k for k in kept if math.hypot((t.x - k.x) / max(self.gate_x[0], self.gate_x[1] * k.x),
                                                 (t.y - k.y) / max(self.gate_y[0], self.gate_y[1] * k.x)) < MERGE), None)
      if twin is None:
        kept.append(t)
      else:
        twin.hits = max(twin.hits, t.hits)
        twin.seen = max(twin.seen, t.seen)
    self.tracks = kept

  def update_merging(self, lane_ys_at) -> None:
    """A car in the next lane drifting toward ours fast enough to be over the line within MERGE_TIME is merging, once that has held
    for MERGE_VOTES looks in a row."""
    for t in self.tracks:
      lines = lane_ys_at(t.x)
      left, right = lines if lines is not None else (-LANE_WIDTH / 2, LANE_WIDTH / 2)
      if t.y < left:
        gap, toward = left - t.y, t.vy        # left of our lane, moving right is toward us
      elif t.y > right:
        gap, toward = t.y - right, -t.vy
      else:
        gap, toward = 0.0, 0.0
      drifting = (t.confirmed and MERGE_RANGE[0] <= t.x <= MERGE_RANGE[1] and toward > MERGE_MIN_SPEED and gap < MERGE_MAX_GAP
                  and gap / toward < MERGE_TIME)
      t.merge_votes = min(t.merge_votes + 1, MERGE_VOTES) if drifting else max(t.merge_votes - 1, 0)
      t.merging = t.merge_votes >= MERGE_VOTES or (t.merging and t.merge_votes > 0)

  @staticmethod
  def _correct(t: VehicleTrack, s: VehicleSighting, now: float, dt: float) -> None:
    rx, ry = s.x - t.x, s.y - t.y
    t.x += ALPHA * rx
    t.y += ALPHA * ry
    since = max(now - t.seen, dt)
    t.vx = float(np.clip(t.vx + BETA * rx / since, -MAX_SPEED_REL, MAX_SPEED_REL))
    t.vy = float(np.clip(t.vy + BETA * ry / since, -MAX_SPEED_REL / 4, MAX_SPEED_REL / 4))
    t.hits += 1
    t.seen = now
    t.score = 0.7 * t.score + 0.3 * s.score
    t.cls, t.camera = s.cls, s.camera
    update_brake_lights(t, s.lamp, now)

  def reset(self) -> None:
    self.tracks.clear()
    self._t = None


def lane_index(x: float, y: float, lane_ys_at, lane_width: float = LANE_WIDTH) -> int:
  """Which lane a point is in, counted from ours: 0 ours, -1 the one on the left, 1 on the right, and so on. lane_ys_at(x) gives
  the model's left and right lane lines of our lane at x (m right), or None when it has none, then the lane is assumed centered."""
  lines = lane_ys_at(x)
  if lines is None:
    left, right = -lane_width / 2, lane_width / 2
  else:
    left, right = lines
    lane_width = float(np.clip(right - left, 2.5, 4.8))
  if left <= y <= right:
    return 0
  if y < left:
    return -1 - int((left - y) // lane_width)
  return 1 + int((y - right) // lane_width)
