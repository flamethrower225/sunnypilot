"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Radar track parsing, classification and threat ranking shared by the radar overlays. No drawing in here.
"""
import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum, StrEnum

import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone, CRITICAL_LIMITS, CLOSE_LIMITS, CAUTION_TTC, \
  MIN_CLOSING_SPEED, SNAP_DISTANCE
from openpilot.selfdrive.ui.sunnypilot.model_plan import model_plan

LANE_HALF_WIDTH = 1.8          # m, |yRel| below this is in our lane
ADJACENT_LANE_LIMIT = 5.4      # m, the neighboring lanes end here
STATIONARY_SPEED = 1.0         # m/s, |absolute speed| below this is a sign, guardrail or parked car
MAX_POINTS = 48                # radarTracks never carries more than this many useful points
BOARD_TTC_MAX = 20.0           # s, closing slower than this isn't worth a row on the threat board
NO_TRACK = -1
VISION_LEAD_IDS = (-2, -3)     # leadOne and leadTwo when no radar track is behind them, so each still has an id of its own
RADAR_TO_CAMERA = 1.52         # m, the model's distances are from the camera, radar distances from the front bumper


class TrackClass(IntEnum):
  MOVING = 0       # same direction as us
  STATIONARY = 1   # not moving over the ground
  ONCOMING = 2     # moving toward us over the ground


class Lane(StrEnum):
  LEFT = "L"
  CENTER = "C"
  RIGHT = "R"


# colors a track is drawn in when it isn't a threat (threats use the severity colors of lead_gap_badge.common.ZONE_RGB)
CLASS_RGB = {
  TrackClass.MOVING: (86, 214, 255),
  TrackClass.STATIONARY: (154, 163, 160),
  TrackClass.ONCOMING: (232, 134, 255),
}


@dataclass(frozen=True, slots=True)
class RadarTrack:
  track_id: int
  d_rel: float      # m from the front bumper
  y_rel: float      # m, left positive
  v_rel: float      # m/s, negative when closing
  v_abs: float      # m/s over the ground, negative when moving toward us
  cls: TrackClass
  lane: Lane
  ttc: float        # s, inf unless closing
  lateral_estimated: bool = False  # the radar gave no lateral position, y_rel was filled in (see parse_tracks)

  @property
  def y_right(self) -> float:
    """Lateral position in the car frame the road projector uses (right positive)."""
    return -self.y_rel

  @property
  def closing(self) -> bool:
    return self.v_rel < -MIN_CLOSING_SPEED

  @property
  def closing_speed(self) -> float:
    return max(-self.v_rel, 0.0)


def classify_motion(v_abs: float) -> TrackClass:
  if abs(v_abs) < STATIONARY_SPEED:
    return TrackClass.STATIONARY
  return TrackClass.ONCOMING if v_abs < 0 else TrackClass.MOVING


def lane_of(y_rel: float) -> Lane:
  if y_rel > LANE_HALF_WIDTH:
    return Lane.LEFT
  if y_rel < -LANE_HALF_WIDTH:
    return Lane.RIGHT
  return Lane.CENTER


def time_to_collision(d_rel: float, v_rel: float) -> float:
  closing = -v_rel
  return d_rel / closing if closing > MIN_CLOSING_SPEED else math.inf


def make_track(track_id: int, d_rel: float, y_rel: float, v_rel: float, v_ego: float, lateral_estimated: bool = False) -> RadarTrack:
  v_abs = v_rel + v_ego
  return RadarTrack(int(track_id), d_rel, y_rel, v_rel, v_abs, classify_motion(v_abs), lane_of(y_rel), time_to_collision(d_rel, v_rel),
                    lateral_estimated)


LateralFallback = Callable[[int, float], float]


def parse_tracks(points, v_ego: float, lateral_fallback: LateralFallback | None = None) -> list[RadarTrack]:
  """Turns radarTracks.points into RadarTracks, skipping garbage (NaN distance or speed, nothing ahead of the bumper).
  Radars that only report the ACC target (Hyundai, Kia and Genesis on camera SCC, for one) give no lateral position. Those
  points are kept: lateral_fallback(track_id, d_rel) fills it in, or without one the point sits in our lane, which is where
  an ACC target is."""
  tracks = []
  for p in itertools.islice(points, MAX_POINTS):
    d_rel, y_rel, v_rel = p.dRel, p.yRel, p.vRel
    if not (math.isfinite(d_rel) and math.isfinite(v_rel)) or d_rel <= 0.0:
      continue
    estimated = not math.isfinite(y_rel)
    if estimated:
      y_rel = lateral_fallback(p.trackId, d_rel) if lateral_fallback is not None else 0.0
      y_rel = y_rel if math.isfinite(y_rel) else 0.0
    tracks.append(make_track(p.trackId, d_rel, y_rel, v_rel, v_ego, estimated))
  return tracks


# ---- threats ----

def ttc_zone(ttc: float) -> GapZone:
  """Severity of a time to collision, shared by the scope, the brackets and the threat board."""
  if ttc < CRITICAL_LIMITS[2]:
    return GapZone.CRITICAL
  if ttc < CLOSE_LIMITS[2]:
    return GapZone.CLOSE
  if ttc < CAUTION_TTC:
    return GapZone.CAUTION
  return GapZone.GOOD if math.isfinite(ttc) else GapZone.OPEN


def in_threat_corridor(track: RadarTrack) -> bool:
  """Whether a closing track can matter to us: anything in our lane, same-direction traffic in the neighboring lanes.
  Stationary tracks are left out (overhead signs and guardrails would cry wolf all day), as is oncoming traffic across the line."""
  if track.cls == TrackClass.STATIONARY:
    return False
  if track.lane == Lane.CENTER:
    return True
  return track.cls == TrackClass.MOVING and abs(track.y_rel) <= ADJACENT_LANE_LIMIT


def is_threat(track: RadarTrack, ttc_max: float = CAUTION_TTC) -> bool:
  return track.ttc <= ttc_max and in_threat_corridor(track)


def display_zone(track: RadarTrack) -> GapZone | None:
  """Severity zone of a closing threat, or None when the track is drawn in its class color."""
  return ttc_zone(track.ttc) if is_threat(track) else None


def threat_key(track: RadarTrack) -> tuple[int, float, float]:
  """Sort key: closing tracks by time to collision first, then everything else by distance."""
  if math.isfinite(track.ttc):
    return 0, track.ttc, track.d_rel
  return 1, 0.0, track.d_rel


def rank_threats(tracks: list[RadarTrack]) -> list[RadarTrack]:
  return sorted(tracks, key=threat_key)


def select_threats(tracks: list[RadarTrack], lead_ids: tuple[int, ...] = (NO_TRACK, NO_TRACK), limit: int = 4) -> list[RadarTrack]:
  """The most threatening tracks for the board: closing within BOARD_TTC_MAX and in the corridor. A radar-matched lead is
  included whenever it's closing, even if it's stationary (a stopped car ahead)."""
  candidates = [t for t in tracks if t.ttc <= BOARD_TTC_MAX and (in_threat_corridor(t) or t.track_id in lead_ids)]
  return rank_threats(candidates)[:limit]


def board_rows(tracks: list[RadarTrack], leads: list[RadarTrack], limit: int = 3) -> list[RadarTrack]:
  """Threat board rows: whatever is closing on us, most urgent first, then the leads even when they aren't closing. A radar
  that reports only the car it follows, or no tracks at all, still gives the board a row for that car."""
  known = {t.track_id for t in tracks}
  pool = tracks + [t for t in leads if t.track_id not in known]
  rows = select_threats(pool, tuple(t.track_id for t in leads), limit)
  taken = {t.track_id for t in rows}
  return rows + [t for t in leads if t.track_id not in taken][:limit - len(rows)]


# ---- leads ----

def lead_track_ids(radar_state) -> tuple[int, int]:
  """radarTrackId of leadOne and leadTwo when they're radar-matched, NO_TRACK otherwise."""
  ids = []
  for lead in (radar_state.leadOne, radar_state.leadTwo):
    ids.append(int(lead.radarTrackId) if lead.present and lead.radar and lead.radarTrackId >= 0 else NO_TRACK)
  return ids[0], ids[1]


def match_leads(tracks: list[RadarTrack], lead_ids: tuple[int, int]) -> tuple[RadarTrack | None, RadarTrack | None]:
  by_id = {t.track_id: t for t in tracks}
  return by_id.get(lead_ids[0]) if lead_ids[0] != NO_TRACK else None, by_id.get(lead_ids[1]) if lead_ids[1] != NO_TRACK else None


# ---- smoothing and freshness ----

class TrackSmoother:
  """Low-passes each track's position and speed by id, so dots glide and numbers don't jitter. A track that
  jumps (new object reusing an id) snaps instead of sliding across the screen."""

  def __init__(self, dt: float, rc: float = 0.12, rc_speed: float = 0.25):
    self._dt = dt
    self._rc = rc
    self._rc_speed = rc_speed
    self._filters: dict[int, tuple[FirstOrderFilter, FirstOrderFilter, FirstOrderFilter]] = {}

  def update(self, tracks: list[RadarTrack], v_ego: float) -> list[RadarTrack]:
    filters = {}
    smoothed = []
    for t in tracks:
      f = self._filters.get(t.track_id)
      if f is None or abs(t.d_rel - f[0].x) > max(SNAP_DISTANCE, 0.25 * t.d_rel):
        f = (FirstOrderFilter(t.d_rel, self._rc, self._dt), FirstOrderFilter(t.y_rel, self._rc, self._dt),
             FirstOrderFilter(t.v_rel, self._rc_speed, self._dt))
      else:
        f[0].update(t.d_rel)
        f[1].update(t.y_rel)
        f[2].update(t.v_rel)
      filters[t.track_id] = f
      smoothed.append(make_track(t.track_id, f[0].x, f[1].x, f[2].x, v_ego, t.lateral_estimated))
    self._filters = filters
    return smoothed


def service_fresh(sm, service: str, started_frame: int) -> bool:
  """Received since the drive started, flagged valid, and still arriving."""
  return bool(sm.valid[service] and sm.recv_frame[service] >= started_frame and sm.alive[service])


class RadarFeed:
  """Everything the radar overlays read each frame: smoothed tracks, whether there is anything to show, and the leads.
  It works down from full radar: a radar with many tracks, one that reports only the ACC target, and a car with no radar
  tracks at all, where the leads radard picked (from the model when there's no radar) are all there is."""

  def __init__(self, dt: float):
    self._smoother = TrackSmoother(dt)
    self.available = False       # radar tracks or radard's leads are arriving
    self.tracks: list[RadarTrack] = []
    self.v_ego = 0.0
    self.lead_ids = (NO_TRACK, NO_TRACK)
    self.lead_one: RadarTrack | None = None   # the radar track leadOne is matched to
    self.lead_two: RadarTrack | None = None
    self.vision_leads: tuple[RadarTrack | None, RadarTrack | None] = (None, None)  # leads with no radar track behind them

  @property
  def leads(self) -> list[RadarTrack]:
    """leadOne then leadTwo, from their radar track when they have one."""
    return [t for t in (self.lead_one or self.vision_leads[0], self.lead_two or self.vision_leads[1]) if t is not None]

  def lead_rank(self, track_id: int) -> int:
    """1 for leadOne, 2 for leadTwo, 0 for anything else."""
    for rank, ids in enumerate(zip(self.lead_ids, VISION_LEAD_IDS, strict=True), start=1):
      if track_id != NO_TRACK and track_id in ids:
        return rank
    return 0

  def update(self, sm, started_frame: int) -> bool:
    self.v_ego = float(sm['carState'].vEgo) if sm.recv_frame['carState'] >= started_frame else 0.0

    tracks_ok = service_fresh(sm, 'radarTracks', started_frame)
    state_ok = service_fresh(sm, 'radarState', started_frame)
    self.available = tracks_ok or state_ok
    radar_state = sm['radarState'] if state_ok else None
    self.lead_ids = lead_track_ids(radar_state) if radar_state is not None else (NO_TRACK, NO_TRACK)

    raw = parse_tracks(sm['radarTracks'].points, self.v_ego, self._lateral_fallback(sm, started_frame, radar_state)) if tracks_ok else []
    self.tracks = self._smoother.update(raw, self.v_ego)
    self.lead_one, self.lead_two = match_leads(self.tracks, self.lead_ids)

    vision: list[RadarTrack | None] = [None, None]
    if radar_state is not None:
      for i, (lead, track) in enumerate(zip((radar_state.leadOne, radar_state.leadTwo), (self.lead_one, self.lead_two), strict=True)):
        if lead.present and track is None and lead.dRel > 0 and math.isfinite(lead.dRel) and math.isfinite(lead.yRel):
          vision[i] = make_track(VISION_LEAD_IDS[i], lead.dRel, lead.yRel, lead.vRel, self.v_ego)
    self.vision_leads = (vision[0], vision[1])
    return self.available

  def _lateral_fallback(self, sm, started_frame: int, radar_state) -> LateralFallback:
    """Lateral position for a radar point that has none. radard gives the lead it matched to the point the model's lateral
    position, so that point takes the lead's; any other sits on the planned path at its distance."""
    by_id = {}
    if radar_state is not None:
      for lead, track_id in zip((radar_state.leadOne, radar_state.leadTwo), self.lead_ids, strict=True):
        if track_id != NO_TRACK and math.isfinite(lead.yRel):
          by_id[track_id] = float(lead.yRel)
    path: tuple[np.ndarray, np.ndarray] | None = None

    def lateral(track_id: int, d_rel: float) -> float:
      nonlocal path
      if track_id in by_id:
        return by_id[track_id]
      if path is None:  # only read the plan when a point actually needs it
        path = planned_path(sm, started_frame)
      if path[0].size:
        return -float(np.interp(d_rel + RADAR_TO_CAMERA, path[0], path[1]))  # the model's y is right positive
      return 0.0
    return lateral


_shared: RadarFeed | None = None
_shared_frame = -1


def shared_radar_feed(sm, started_frame: int, dt: float) -> RadarFeed:
  """The one RadarFeed the scope, the board and the brackets all read, advanced once per SubMaster frame. Each used to keep its own,
  which parsed and smoothed every track three times a frame."""
  global _shared, _shared_frame
  if _shared is None:
    _shared = RadarFeed(dt)
  if sm.frame != _shared_frame:
    _shared_frame = sm.frame
    _shared.update(sm, started_frame)
  return _shared


def planned_path(sm, started_frame: int) -> tuple[np.ndarray, np.ndarray]:
  """x and y (m, right positive) of the model's planned path, empty when there is no plan this drive."""
  if sm.recv_frame.get('modelV2', -1) < started_frame:
    return np.empty(0), np.empty(0)
  plan = model_plan(sm, started_frame)
  if plan.size:
    return plan.x, plan.y
  position = sm['modelV2'].position  # a plan without speeds still has a path
  xs, ys = list(position.x), list(position.y)  # capnp lists don't slice
  n = min(len(xs), len(ys))
  return np.asarray(xs[:n], dtype=float), np.asarray(ys[:n], dtype=float)
