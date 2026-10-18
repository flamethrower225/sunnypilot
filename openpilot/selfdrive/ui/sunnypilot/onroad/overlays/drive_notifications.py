"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Drive notifications: short advisories in a card at the top right, for things worth knowing that no alert covers.

  Slow lead        stuck behind a lead well under the speed limit (or the set speed when there is no limit): pass on the left,
                   or the right when that is the only lane, unless the blind spot monitor sees a car there
  Stopped traffic  a stopped car ahead while still at speed, before the gap gets short
  Sharp curve      the bend ahead calls for a lot less speed than now and the car isn't slowing for it
  Speed limit      the limit changed
  Over the limit   well over the limit for a while (only when Speed Limit doesn't already warn)
  Break            two hours on the road
  Red light        a red light ahead the car isn't braking enough for     } with Camera Object Detection on
  Yellow light     it just turned yellow: room to stop or not             }
  Light is green   stopped at a red that turned green                     }
  Flashing signal  flashing red (stop, then go) or yellow (caution)       }
  Stop sign        a stop sign ahead the car isn't braking enough for     }
  Pedestrian       someone walking or cycling in our path                 }
  Braking ahead    brake lights past the car we follow                    }
  Merging          a car in the next lane drifting into ours              }

One card at a time, the most important first. Each kind has a cooldown so nothing nags, and nothing shows while you are already
changing lanes. The logic below works on a DriveSnapshot and has no drawing in it; DriveNotifications reads one from the
SubMaster every frame and draws the card. Every notification also goes into the detection log.
"""
import datetime
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pyray as rl

from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.onroad.hud_renderer import UI_CONFIG
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, INK, TEXT_MIN, WHITE, ZONE_RGB, draw_text, \
  draw_text_centered, draw_triangle, rgba, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import FT_PER_M, SNAP_DISTANCE, GapZone
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.curve_hologram import curve_ahead, curve_decel_needed
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.detection import DetectionEvent, EventKind, get_detection_feed
from openpilot.selfdrive.ui.sunnypilot.model_plan import model_plan
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.light_view import camera_people, current_light, current_stop_sign, draw_signal_head, \
  draw_stop_sign
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_tracks import RADAR_TO_CAMERA, service_fresh
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.common import Mode as SpeedLimitMode
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wrap_text import wrap_text

LaneChangeState = log.LaneChangeState
MPH = CV.MPH_TO_MS

# slow lead
PASS_MIN_SPEED = 25 * MPH        # below this it's town traffic, not a pass
SLOW_MARGIN = 5 * MPH            # the lead has to be this far under the limit (or 10 % of it, whichever is more)
SLOW_MARGIN_FRACTION = 0.1
SLOW_SUSTAIN = 8.0               # s stuck behind it before suggesting a pass
FOLLOW_TIME = 3.5                # s, a lead within this time gap (or 60 m) is the one holding us up
FOLLOW_MIN_DISTANCE = 60.0       # m

# adjacent lanes, from the model's lane lines a little ahead
LANE_PROBE_X = 20.0              # m ahead where the lanes are measured
LANE_PROB = 0.45                 # the outer line has to be at least this likely
LANE_WIDTH = (2.6, 4.8)          # m, a gap between lines this wide is a lane
EDGE_ROOM = 2.4                  # m, and the road edge has to leave this much room beyond our lane line

# stopped traffic
STOPPED_LEAD_SPEED = 1.5         # m/s
STOPPED_MIN_EGO = 30 * MPH
STOPPED_RANGE = 160.0            # m
STOPPED_HORIZON = 9.0            # s at our speed

# sharp curve
CURVE_RANGE = (25.0, 200.0)      # m to the tightest point
CURVE_EXCESS = 6 * MPH           # faster than the bend's speed by this much
CURVE_DECEL_MARGIN = 0.6         # m/s^2 more braking needed than the car already has
CURVE_SUSTAIN = 0.4              # s

# speed limit
LIMIT_SETTLE = 1.5               # s a new limit has to hold before it counts
OVER_MARGIN = 5 * MPH            # or 10 %, whichever is more
OVER_SUSTAIN = 5.0               # s

BREAK_AFTER = 2 * 3600.0         # s on the road

# traffic lights (Camera Object Detection)
LIGHT_RED, LIGHT_YELLOW, LIGHT_GREEN = 1, 2, 3
KIND_CYCLIST = 5                 # SceneDetectionSP.VehicleKind.cyclist
LIGHT_SETBACK = 10.0             # m, the stop line is about this far before the light
LIGHT_RANGE = (8.0, 150.0)       # m
LIGHT_MIN_SPEED = 15 * MPH
LIGHT_DECEL_MARGIN = 0.6         # m/s^2 more braking needed than the car has before it warns
LIGHT_SUSTAIN = 0.3              # s
YELLOW_FRESH = 1.5               # s, a yellow older than this is old news
YELLOW_COMFORT_DECEL = 3.0       # m/s^2
GREEN_FRESH = 2.0                # s
GREEN_STOPPED_SPEED = 1.0        # m/s
CAMERA_BESIDE = (-8.0, 25.0)     # m from the bumper: a camera car this close in the next lane blocks a lane change
SIGN_RANGE = (5.0, 100.0)        # m
SIGN_MIN_SPEED = 15 * MPH
FLASH_MIN_SPEED = 3.0            # m/s
PERSON_MIN_SPEED = 2.0           # m/s
PERSON_NEAR = 15.0               # m, this close in our path always counts
PERSON_TTC = 6.0                 # s
PERSON_RANGE = 45.0              # m
PERSON_PATH = 2.2                # m either side of the planned path
BRAKING_MIN_SPEED = 8.0          # m/s
BRAKING_SUSTAIN = 0.3            # s
BRAKING_PAST_LEAD = 5.0          # m: a braking car this far past the lead is one the lead hides
BRAKING_RANGE = 100.0            # m
MERGING_MIN_SPEED = 5.0          # m/s
MERGING_RANGE = (-5.0, 40.0)     # m from the bumper

CARD_TTL_MIN = 3.0               # s a card stays at least, even when its reason is gone


class NotifyKind(IntEnum):
  # in order of priority, most important first
  STOPPED_TRAFFIC = 0
  PEDESTRIAN = 1
  MERGING = 2
  RED_LIGHT = 3
  STOP_SIGN = 4
  BRAKING_AHEAD = 5
  SHARP_CURVE = 6
  FLASHING_SIGNAL = 7
  YELLOW_LIGHT = 8
  LIGHT_GREEN = 9
  SLOW_LEAD = 10
  OVER_LIMIT = 11
  SPEED_LIMIT = 12
  BREAK = 13


class Icon(IntEnum):
  WARNING = 0
  LEFT = 1
  RIGHT = 2
  DOWN = 3
  UP = 4
  CURVE_LEFT = 5
  CURVE_RIGHT = 6
  CLOCK = 7
  WAIT = 8
  SIGNAL_RED = 9
  SIGNAL_YELLOW = 10
  SIGNAL_GREEN = 11
  STOP_SIGN = 12
  PERSON = 13


@dataclass(frozen=True)
class Notification:
  kind: NotifyKind
  title: str
  text: str
  detail: str
  rgb: tuple
  icon: Icon
  key: object = None               # what this one is about (a lead, a limit), so a cooldown doesn't hold back a different one
  distance: float | None = None    # m, for the detection log


@dataclass(frozen=True)
class DriveSnapshot:
  now: float                       # monotonic s
  is_metric: bool = False
  v_ego: float = 0.0
  a_ego: float = 0.0
  speed_limit: float = 0.0         # m/s, 0 when unknown
  speed_limit_warns: bool = False  # Speed Limit is in warning or assist mode and already tells the driver
  cruise_speed: float = 0.0        # m/s, 0 when cruise isn't set
  lead_present: bool = False
  lead_d: float = 0.0              # m
  lead_v: float = 0.0              # m/s
  left_lane: bool = False          # a lane to the left of ours
  right_lane: bool = False
  left_blindspot: bool = False
  right_blindspot: bool = False
  changing_lanes: bool = False     # blinker on or a lane change under way
  apex_x: float | None = None      # m to the tightest point of the bend ahead
  apex_speed: float = math.inf     # m/s the bend calls for
  curve_right: bool = True
  drive_time: float = 0.0          # s since the drive started
  # from Camera Object Detection (scened), when it's on
  light_state: int = 0             # 0 unknown, 1 red, 2 yellow, 3 green (SceneDetectionSP.LightState)
  light_distance: float = math.nan # m from the front bumper
  light_age: float = 0.0           # s the state has held
  left_busy: bool = False          # the cameras see a car in the lane beside us, just behind to a little ahead
  right_busy: bool = False
  light_stop_distance: float = math.nan   # m to where we'd stop for the light (light_view.fused_stop_distance)
  light_flashing: bool = False
  sign_stop_distance: float = math.nan    # m to where we'd stop for a stop sign, NaN when there's none
  person_distance: float = math.nan       # m to the nearest pedestrian or cyclist in our path
  person_ttc: float = math.inf            # s until we reach them at the speeds now
  person_cyclist: bool = False
  braking_ahead: float = math.nan         # m to the nearest car in our lane past the lead whose brake lights are on
  merging_side: int = 0                   # a car merging into our lane: -1 from the left, 1 from the right
  merging_distance: float = math.nan


# ---- formatting ----

def speed_text(v: float, is_metric: bool) -> str:
  return f"{round(v * (CV.MS_TO_KPH if is_metric else CV.MS_TO_MPH))} {tr('km/h') if is_metric else tr('mph')}"


def distance_text(d: float, is_metric: bool) -> str:
  if is_metric:
    return f"{round(d / 10) * 10:.0f} m" if d >= 100 else f"{d:.0f} m"
  feet = d * FT_PER_M
  return f"{round(feet / 50) * 50:.0f} ft" if feet >= 500 else f"{feet:.0f} ft"


def duration_text(seconds: float) -> str:
  hours, minutes = int(seconds // 3600), int(seconds % 3600 // 60)
  return tr("{} h {} min").format(hours, minutes) if hours else tr("{} min").format(minutes)


def lane_widths_ok(inner: float, outer: float) -> bool:
  return LANE_WIDTH[0] <= abs(outer - inner) <= LANE_WIDTH[1]


def adjacent_lanes(lane_ys, lane_probs, edge_ys) -> tuple[bool, bool]:
  """Whether there is a lane left and right of ours. lane_ys are the four lane lines' lateral positions (m, right positive) at
  LANE_PROBE_X, far left to far right, edge_ys the two road edges. A lane needs its outer line seen, a lane's width between the
  lines, and the road edge (when the model has one) far enough out that the outer line isn't the shoulder line."""
  if len(lane_ys) < 4 or len(lane_probs) < 4:
    return False, False
  left = lane_probs[0] > LANE_PROB and lane_widths_ok(lane_ys[1], lane_ys[0])
  right = lane_probs[3] > LANE_PROB and lane_widths_ok(lane_ys[2], lane_ys[3])
  if len(edge_ys) >= 2:
    left = left and lane_ys[1] - edge_ys[0] > EDGE_ROOM
    right = right and edge_ys[1] - lane_ys[2] > EDGE_ROOM
  return bool(left), bool(right)


# ---- rules ----

class Rule:
  kind: NotifyKind
  ttl = 6.0          # s the card stays up
  cooldown = 30.0    # s before this kind can show again (for the same key)
  sticky = False     # stays up while the reason holds, up to ttl, and goes once it's gone

  def check(self, s: DriveSnapshot) -> Notification | None:
    raise NotImplementedError

  def shown(self, n: Notification) -> None:
    """Called when the engine puts a notification of this rule on screen."""


class SlowLeadRule(Rule):
  kind = NotifyKind.SLOW_LEAD
  ttl = 8.0
  cooldown = 90.0

  def __init__(self):
    self._since: float | None = None
    self._lead_d = 0.0
    self._lead_id = 0   # bumped whenever a different car becomes the lead

  def check(self, s: DriveSnapshot) -> Notification | None:
    if s.lead_present and abs(s.lead_d - self._lead_d) > max(SNAP_DISTANCE, 0.25 * s.lead_d):
      self._lead_id += 1
    self._lead_d = s.lead_d if s.lead_present else 0.0

    reference = s.speed_limit if s.speed_limit > 0 else s.cruise_speed
    margin = max(SLOW_MARGIN, SLOW_MARGIN_FRACTION * reference)
    following = s.lead_present and s.lead_d < max(FOLLOW_MIN_DISTANCE, FOLLOW_TIME * s.v_ego)
    stuck = (reference > 0 and following and s.v_ego > PASS_MIN_SPEED and s.lead_v < reference - margin and
             s.v_ego < reference - margin / 2)
    if not stuck or s.changing_lanes:
      self._since = None
      return None
    self._since = s.now if self._since is None else self._since
    if s.now - self._since < SLOW_SUSTAIN:
      return None

    under = speed_text(reference - s.lead_v, s.is_metric)
    detail = tr("{} under limit").format(under) if s.speed_limit > 0 else tr("{} under set speed").format(under)
    rgb = ZONE_RGB[GapZone.OPEN]
    if s.left_lane and not (s.left_blindspot or s.left_busy):
      return Notification(self.kind, tr("SLOW LEAD"), tr("Pass on the left"), detail, rgb, Icon.LEFT, self._lead_id, s.lead_d)
    if s.left_lane:
      return Notification(self.kind, tr("SLOW LEAD"), tr("Left lane busy"), detail, ZONE_RGB[GapZone.CAUTION], Icon.WAIT, self._lead_id,
                          s.lead_d)
    if s.right_lane and not (s.right_blindspot or s.right_busy):
      return Notification(self.kind, tr("SLOW LEAD"), tr("Right lane open"), detail, rgb, Icon.RIGHT, self._lead_id, s.lead_d)
    return None  # nowhere to go, so nothing to suggest


class StoppedTrafficRule(Rule):
  kind = NotifyKind.STOPPED_TRAFFIC
  ttl = 8.0
  cooldown = 30.0
  sticky = True

  def check(self, s: DriveSnapshot) -> Notification | None:
    if not (s.lead_present and s.lead_v < STOPPED_LEAD_SPEED and s.v_ego > STOPPED_MIN_EGO):
      return None
    if s.lead_d > min(STOPPED_RANGE, STOPPED_HORIZON * s.v_ego):
      return None
    return Notification(self.kind, tr("STOPPED TRAFFIC"), tr("Stopped car ahead"), distance_text(s.lead_d, s.is_metric),
                        ZONE_RGB[GapZone.CRITICAL], Icon.WARNING, None, s.lead_d)


class SharpCurveRule(Rule):
  kind = NotifyKind.SHARP_CURVE
  ttl = 8.0
  cooldown = 20.0
  sticky = True

  def __init__(self):
    self._since: float | None = None

  def check(self, s: DriveSnapshot) -> Notification | None:
    apex_x = s.apex_x if s.apex_x is not None else math.inf
    hot = False
    if CURVE_RANGE[0] <= apex_x <= CURVE_RANGE[1] and math.isfinite(s.apex_speed):
      need = curve_decel_needed(s.v_ego, s.apex_speed, apex_x)
      hot = s.v_ego > s.apex_speed + CURVE_EXCESS and need > max(-s.a_ego, 0.0) + CURVE_DECEL_MARGIN
    if not hot:
      self._since = None
      return None
    self._since = s.now if self._since is None else self._since
    if s.now - self._since < CURVE_SUSTAIN:
      return None
    return Notification(self.kind, tr("SHARP CURVE"), tr("Slow to {}").format(speed_text(s.apex_speed, s.is_metric)),
                        tr("in {}").format(distance_text(apex_x, s.is_metric)), ZONE_RGB[GapZone.CLOSE],
                        Icon.CURVE_RIGHT if s.curve_right else Icon.CURVE_LEFT, None, apex_x)


def display_limit(v: float, is_metric: bool) -> int:
  return round(v * (CV.MS_TO_KPH if is_metric else CV.MS_TO_MPH))


class SpeedLimitRule(Rule):
  """Fires once when the limit changes and the new one has held for a moment, and keeps the news until it has been shown."""
  kind = NotifyKind.SPEED_LIMIT
  ttl = 5.0
  cooldown = 0.0

  def __init__(self):
    self._shown_limit = 0
    self._candidate = 0
    self._candidate_since = 0.0
    self._pending: Notification | None = None

  def check(self, s: DriveSnapshot) -> Notification | None:
    limit = display_limit(s.speed_limit, s.is_metric) if s.speed_limit > 0 else 0
    if limit != self._candidate:
      self._candidate, self._candidate_since = limit, s.now
    elif limit and limit != self._shown_limit and s.now - self._candidate_since >= LIMIT_SETTLE:
      icon = Icon.DOWN if self._shown_limit and limit < self._shown_limit else Icon.UP if self._shown_limit else Icon.WARNING
      unit = tr("km/h") if s.is_metric else tr("mph")
      was = tr("was {}").format(self._shown_limit) if self._shown_limit else tr("posted limit")
      self._pending = Notification(self.kind, tr("SPEED LIMIT"), f"{limit} {unit}", was, WHITE, icon, limit)
      self._shown_limit = limit
    if self._pending is not None and s.now - self._candidate_since > LIMIT_SETTLE + 15.0:
      self._pending = None  # stale news by now
    return self._pending

  def shown(self, n: Notification) -> None:
    self._pending = None


class OverLimitRule(Rule):
  kind = NotifyKind.OVER_LIMIT
  ttl = 6.0
  cooldown = 120.0
  sticky = True

  def __init__(self):
    self._since: float | None = None

  def check(self, s: DriveSnapshot) -> Notification | None:
    over = s.v_ego - s.speed_limit
    if s.speed_limit <= 0 or s.speed_limit_warns or over <= max(OVER_MARGIN, 0.1 * s.speed_limit):
      self._since = None
      return None
    self._since = s.now if self._since is None else self._since
    if s.now - self._since < OVER_SUSTAIN:
      return None
    return Notification(self.kind, tr("SPEED"), tr("{} over the limit").format(speed_text(over, s.is_metric)),
                        tr("Limit {}").format(speed_text(s.speed_limit, s.is_metric)), ZONE_RGB[GapZone.CAUTION], Icon.DOWN)


class BreakRule(Rule):
  kind = NotifyKind.BREAK
  ttl = 10.0
  cooldown = 0.0

  def __init__(self):
    self._next = BREAK_AFTER

  def check(self, s: DriveSnapshot) -> Notification | None:
    if s.drive_time < self._next - BREAK_AFTER:  # a new drive started
      self._next = BREAK_AFTER
    if s.drive_time < self._next:
      return None
    return Notification(self.kind, tr("BREAK"), tr("Time for a break?"), tr("{} on the road").format(duration_text(s.drive_time)),
                        ZONE_RGB[GapZone.GOOD], Icon.CLOCK, self._next)

  def shown(self, n: Notification) -> None:
    self._next += BREAK_AFTER


def light_stop_decel(v_ego: float, stop_distance: float) -> float:
  """m/s^2 to stop in stop_distance (to the stop line, see light_view.fused_stop_distance)."""
  return v_ego ** 2 / (2 * max(stop_distance, 1.0))


def stop_distance_of(s: DriveSnapshot) -> float:
  return s.light_stop_distance if math.isfinite(s.light_stop_distance) else max(s.light_distance - LIGHT_SETBACK, 0.0)


class RedLightRule(Rule):
  """Coming up on a red light faster than the braking so far will stop for."""
  kind = NotifyKind.RED_LIGHT
  ttl = 8.0
  cooldown = 15.0
  sticky = True

  def __init__(self):
    self._since: float | None = None

  def check(self, s: DriveSnapshot) -> Notification | None:
    hot = (s.light_state == LIGHT_RED and not s.light_flashing and math.isfinite(s.light_distance)
           and LIGHT_RANGE[0] <= s.light_distance <= LIGHT_RANGE[1] and s.v_ego > LIGHT_MIN_SPEED
           and light_stop_decel(s.v_ego, stop_distance_of(s)) > max(-s.a_ego, 0.0) + LIGHT_DECEL_MARGIN)
    if not hot:
      self._since = None
      return None
    self._since = s.now if self._since is None else self._since
    if s.now - self._since < LIGHT_SUSTAIN:
      return None
    return Notification(self.kind, tr("RED LIGHT"), tr("Red light ahead"),
                        tr("Stop in {}").format(distance_text(stop_distance_of(s), s.is_metric)),
                        ZONE_RGB[GapZone.CRITICAL], Icon.SIGNAL_RED, None, s.light_distance)


class YellowLightRule(Rule):
  """The light just turned yellow: whether there's room to stop at comfortable braking. It doesn't tell anyone to go."""
  kind = NotifyKind.YELLOW_LIGHT
  ttl = 4.0
  cooldown = 15.0

  def check(self, s: DriveSnapshot) -> Notification | None:
    if not (s.light_state == LIGHT_YELLOW and not s.light_flashing and s.light_age < YELLOW_FRESH and math.isfinite(s.light_distance)
            and LIGHT_RANGE[0] <= s.light_distance <= LIGHT_RANGE[1] and s.v_ego > LIGHT_MIN_SPEED):
      return None
    need = light_stop_decel(s.v_ego, stop_distance_of(s))
    text = tr("Room to stop") if need <= YELLOW_COMFORT_DECEL else tr("Too close to stop gently")
    return Notification(self.kind, tr("YELLOW LIGHT"), text, distance_text(s.light_distance, s.is_metric), ZONE_RGB[GapZone.CAUTION],
                        Icon.SIGNAL_YELLOW, None, s.light_distance)


class LightGreenRule(Rule):
  """Stopped at a red, and it turned green."""
  kind = NotifyKind.LIGHT_GREEN
  ttl = 5.0
  cooldown = 10.0

  def __init__(self):
    self._red_while_stopped = False

  def check(self, s: DriveSnapshot) -> Notification | None:
    stopped = s.v_ego < GREEN_STOPPED_SPEED
    if not stopped:
      self._red_while_stopped = False
      return None
    if s.light_state == LIGHT_RED:
      self._red_while_stopped = True
    if self._red_while_stopped and s.light_state == LIGHT_GREEN and s.light_age < GREEN_FRESH:
      return Notification(self.kind, tr("GREEN LIGHT"), tr("Light is green"), tr("Check the intersection"), ZONE_RGB[GapZone.GOOD],
                          Icon.SIGNAL_GREEN)
    return None

  def shown(self, n: Notification) -> None:
    self._red_while_stopped = False


class StopSignRule(Rule):
  """Coming up on a stop sign faster than the braking so far will stop for."""
  kind = NotifyKind.STOP_SIGN
  ttl = 8.0
  cooldown = 15.0
  sticky = True

  def __init__(self):
    self._since: float | None = None

  def check(self, s: DriveSnapshot) -> Notification | None:
    d = s.sign_stop_distance
    hot = (math.isfinite(d) and SIGN_RANGE[0] <= d <= SIGN_RANGE[1] and s.v_ego > SIGN_MIN_SPEED
           and light_stop_decel(s.v_ego, d) > max(-s.a_ego, 0.0) + LIGHT_DECEL_MARGIN)
    if not hot:
      self._since = None
      return None
    self._since = s.now if self._since is None else self._since
    if s.now - self._since < LIGHT_SUSTAIN:
      return None
    return Notification(self.kind, tr("STOP SIGN"), tr("Stop sign ahead"), tr("Stop in {}").format(distance_text(d, s.is_metric)),
                        ZONE_RGB[GapZone.CRITICAL], Icon.STOP_SIGN, None, d)


class FlashingSignalRule(Rule):
  """A flashing signal ahead, said once per approach: flashing red is a stop sign, flashing yellow is caution."""
  kind = NotifyKind.FLASHING_SIGNAL
  ttl = 6.0
  cooldown = 30.0

  def check(self, s: DriveSnapshot) -> Notification | None:
    if not (s.light_flashing and s.light_state in (LIGHT_RED, LIGHT_YELLOW) and s.v_ego > FLASH_MIN_SPEED
            and math.isfinite(s.light_distance) and s.light_distance <= LIGHT_RANGE[1]):
      return None
    if s.light_state == LIGHT_RED:
      return Notification(self.kind, tr("FLASHING RED"), tr("Stop, then go when clear"), distance_text(s.light_distance, s.is_metric),
                          ZONE_RGB[GapZone.CRITICAL], Icon.SIGNAL_RED, LIGHT_RED, s.light_distance)
    return Notification(self.kind, tr("FLASHING YELLOW"), tr("Proceed with caution"), distance_text(s.light_distance, s.is_metric),
                        ZONE_RGB[GapZone.CAUTION], Icon.SIGNAL_YELLOW, LIGHT_YELLOW, s.light_distance)


class PedestrianRule(Rule):
  """Someone walking or cycling in our path, close or soon to be."""
  kind = NotifyKind.PEDESTRIAN
  ttl = 6.0
  cooldown = 8.0
  sticky = True

  def check(self, s: DriveSnapshot) -> Notification | None:
    d = s.person_distance
    if not (math.isfinite(d) and s.v_ego > PERSON_MIN_SPEED and (d <= PERSON_NEAR or s.person_ttc <= PERSON_TTC)):
      return None
    text = tr("Cyclist ahead") if s.person_cyclist else tr("Pedestrian ahead")
    return Notification(self.kind, tr("CYCLIST") if s.person_cyclist else tr("PEDESTRIAN"), text, distance_text(d, s.is_metric),
                        ZONE_RGB[GapZone.CRITICAL], Icon.PERSON, None, d)


class BrakingAheadRule(Rule):
  """Brake lights past the car we follow: the cars ahead of it are slowing before it does."""
  kind = NotifyKind.BRAKING_AHEAD
  ttl = 4.0
  cooldown = 12.0

  def __init__(self):
    self._since: float | None = None

  def check(self, s: DriveSnapshot) -> Notification | None:
    if not (math.isfinite(s.braking_ahead) and s.v_ego > BRAKING_MIN_SPEED):
      self._since = None
      return None
    self._since = s.now if self._since is None else self._since
    if s.now - self._since < BRAKING_SUSTAIN:
      return None
    return Notification(self.kind, tr("TRAFFIC"), tr("Brake lights ahead"), distance_text(s.braking_ahead, s.is_metric),
                        ZONE_RGB[GapZone.CLOSE], Icon.DOWN, None, s.braking_ahead)


class MergingRule(Rule):
  """A car in the next lane drifting into ours."""
  kind = NotifyKind.MERGING
  ttl = 6.0
  cooldown = 8.0
  sticky = True

  def check(self, s: DriveSnapshot) -> Notification | None:
    if s.merging_side == 0 or s.v_ego < MERGING_MIN_SPEED:
      return None
    text = tr("From the left") if s.merging_side < 0 else tr("From the right")
    detail = distance_text(max(s.merging_distance, 0.0), s.is_metric) if math.isfinite(s.merging_distance) else ""
    # the arrow shows the way it's coming: from the left it moves right
    return Notification(self.kind, tr("CAR MERGING"), text, detail, ZONE_RGB[GapZone.CLOSE], Icon.RIGHT if s.merging_side < 0 else Icon.LEFT,
                        s.merging_side, s.merging_distance)


def default_rules() -> list[Rule]:
  rules: list[Rule] = [StoppedTrafficRule(), PedestrianRule(), MergingRule(), RedLightRule(), StopSignRule(), BrakingAheadRule(),
                       SharpCurveRule(), FlashingSignalRule(), YellowLightRule(), LightGreenRule(), SlowLeadRule(), OverLimitRule(),
                       SpeedLimitRule(), BreakRule()]
  return sorted(rules, key=lambda r: r.kind)


@dataclass
class _Shown:
  rule: Rule
  note: Notification
  since: float
  last_seen: float


@dataclass
class NotificationEngine:
  """Picks the one notification on screen. A more important one takes over at once; otherwise a card stays for its rule's time
  (a sticky one for as long as its reason holds, at least CARD_TTL_MIN), then its rule cools down."""
  rules: list[Rule] = field(default_factory=default_rules)
  current: _Shown | None = None
  _cooldown: dict = field(default_factory=dict)   # kind -> (until, key)

  def _cooling(self, rule: Rule, note: Notification, now: float) -> bool:
    until, key = self._cooldown.get(rule.kind, (-math.inf, None))
    return now < until and key == note.key

  def _end(self, now: float) -> None:
    if self.current is not None:
      self._cooldown[self.current.rule.kind] = (now + self.current.rule.cooldown, self.current.note.key)
      self.current = None

  def update(self, s: DriveSnapshot) -> Notification | None:
    """Returns a notification the moment it goes on screen, None otherwise. What's on screen is self.current."""
    candidates = [(rule, note) for rule in self.rules if (note := rule.check(s)) is not None]

    cur = self.current
    if cur is not None:
      same = next((note for rule, note in candidates if rule is cur.rule), None)
      if same is not None:
        cur.note, cur.last_seen = same, s.now  # keep the numbers fresh
      age = s.now - cur.since
      gone = cur.rule.sticky and same is None and age >= CARD_TTL_MIN
      if age >= cur.rule.ttl or gone:
        self._end(s.now)

    for rule, note in candidates:  # most important first
      if self.current is not None and rule.kind >= self.current.rule.kind:
        break  # only something more important takes over
      if self._cooling(rule, note, s.now):
        continue
      self._end(s.now)
      self.current = _Shown(rule, note, s.now, s.now)
      rule.shown(note)
      return note
    return None

  def reset(self) -> None:
    self.current = None
    self._cooldown.clear()


# ---- reading the car ----

_lanes_model = None
_lanes = (False, False)


def model_lanes(sm) -> tuple[bool, bool]:
  """adjacent_lanes from the latest modelV2, worked out once per model frame."""
  global _lanes_model, _lanes
  model = sm['modelV2']
  if model is not _lanes_model:
    _lanes_model = model
    lines, edges = model.laneLines, model.roadEdges

    def y_at(line) -> float:
      xs, ys = list(line.x), list(line.y)
      return float(np.interp(LANE_PROBE_X, xs, ys)) if len(xs) >= 2 and len(xs) == len(ys) else math.nan

    lane_ys = [y_at(line) for line in lines]
    edge_ys = [y_at(edge) for edge in edges] if len(edges) >= 2 else []
    if any(math.isnan(y) for y in edge_ys):
      edge_ys = []
    _lanes = (False, False) if len(lane_ys) < 4 or any(math.isnan(y) for y in lane_ys) else \
      adjacent_lanes(lane_ys, list(model.laneLineProbs), edge_ys)
  return _lanes


def read_snapshot(sm, now: float) -> DriveSnapshot:
  started = ui_state.started_frame
  cs = sm['carState']
  v_ego = float(cs.vEgo) if sm.recv_frame['carState'] >= started else 0.0

  lead = sm['radarState'].leadOne
  lead_ok = bool(sm.valid['radarState'] and sm.recv_frame['radarState'] >= started and lead.present)

  resolver = sm['longitudinalPlanSP'].speedLimit.resolver
  speed_limit = float(resolver.speedLimit) if resolver.speedLimitValid and resolver.speedLimit > 0 else 0.0
  cruise = float(cs.cruiseState.speed) if cs.cruiseState.enabled and 0 < cs.cruiseState.speed < 70 else 0.0

  left_lane = right_lane = False
  if sm.recv_frame['modelV2'] >= started:
    left_lane, right_lane = model_lanes(sm)
  curve = curve_ahead(sm, started)

  light = current_light(sm)
  sign = current_stop_sign(sm)
  left_busy = right_busy = False
  braking_ahead, merging_side, merging_distance = math.nan, 0, math.nan
  if ui_state.scene_detection and service_fresh(sm, 'sceneDetectionSP', started):
    for v in sm['sceneDetectionSP'].vehicles:
      if v.lastSeen > 1.0:
        continue
      if v.lane == -1:
        left_lane = True   # a car driving there is the best sign there's a lane
        left_busy |= CAMERA_BESIDE[0] <= v.dRel <= CAMERA_BESIDE[1]
      elif v.lane == 1:
        right_lane = True
        right_busy |= CAMERA_BESIDE[0] <= v.dRel <= CAMERA_BESIDE[1]
      if v.braking and v.lane == 0 and v.dRel <= BRAKING_RANGE and (not lead_ok or v.dRel > lead.dRel + BRAKING_PAST_LEAD):
        braking_ahead = v.dRel if not math.isfinite(braking_ahead) else min(braking_ahead, v.dRel)
      if v.merging and MERGING_RANGE[0] <= v.dRel <= MERGING_RANGE[1] and (not math.isfinite(merging_distance) or v.dRel < merging_distance):
        merging_side, merging_distance = (-1 if v.yRel > 0 else 1), float(v.dRel)
  person_distance, person_ttc, person_cyclist = person_in_path(sm, started, v_ego)

  lane_change = sm['modelV2'].meta.laneChangeState if sm.recv_frame['modelV2'] >= started else LaneChangeState.off
  return DriveSnapshot(
    now=now, is_metric=ui_state.is_metric, v_ego=v_ego, a_ego=float(cs.aEgo),
    speed_limit=speed_limit, speed_limit_warns=ui_state.speed_limit_mode in (SpeedLimitMode.warning, SpeedLimitMode.assist),
    cruise_speed=cruise, lead_present=lead_ok, lead_d=float(lead.dRel), lead_v=float(lead.vLeadK),
    left_lane=left_lane, right_lane=right_lane, left_blindspot=bool(cs.leftBlindspot), right_blindspot=bool(cs.rightBlindspot),
    changing_lanes=bool(cs.leftBlinker or cs.rightBlinker or lane_change != LaneChangeState.off),
    apex_x=curve.x if curve else None, apex_speed=curve.speed if curve else math.inf, curve_right=curve.right if curve else True,
    drive_time=now - ui_state.started_time if ui_state.started_time else 0.0,
    light_state=light.state if light else 0, light_distance=light.distance if light else math.nan, light_age=light.age if light else 0.0,
    left_busy=left_busy, right_busy=right_busy,
    light_stop_distance=light.stop_distance if light else math.nan, light_flashing=light.flashing if light else False,
    sign_stop_distance=sign.stop_distance if sign else math.nan,
    person_distance=person_distance, person_ttc=person_ttc, person_cyclist=person_cyclist,
    braking_ahead=braking_ahead, merging_side=merging_side, merging_distance=merging_distance,
  )


def person_in_path(sm, started: int, v_ego: float) -> tuple[float, float, bool]:
  """(distance, time to reach, cyclist) of the nearest pedestrian or cyclist within PERSON_PATH of the planned path."""
  best = (math.nan, math.inf, False)
  people = camera_people(sm)
  if not people:
    return best
  plan = model_plan(sm, started)
  for p in people:
    if not 0.0 < p.dRel <= PERSON_RANGE:
      continue
    path_y = float(np.interp(p.dRel + RADAR_TO_CAMERA, plan.x, plan.y)) if plan.size else 0.0
    if abs(-p.yRel - path_y) > PERSON_PATH:
      continue
    closing = max(-p.vRel, v_ego * 0.5)
    ttc = p.dRel / closing if closing > 0.5 else math.inf
    if not math.isfinite(best[0]) or p.dRel < best[0]:
      best = (float(p.dRel), ttc, _raw_kind(p) == KIND_CYCLIST)
  return best


def _raw_kind(p) -> int:
  return int(p.kind.raw if hasattr(p.kind, "raw") else p.kind)


LOG_KINDS = {
  NotifyKind.STOPPED_TRAFFIC: EventKind.STOPPED_TRAFFIC,
  NotifyKind.PEDESTRIAN: EventKind.PEDESTRIAN,
  NotifyKind.MERGING: EventKind.MERGING,
  NotifyKind.STOP_SIGN: EventKind.STOP_SIGN,
  NotifyKind.BRAKING_AHEAD: EventKind.BRAKING_AHEAD,
  NotifyKind.FLASHING_SIGNAL: EventKind.FLASHING_SIGNAL,
  NotifyKind.RED_LIGHT: EventKind.RED_LIGHT,
  NotifyKind.YELLOW_LIGHT: EventKind.YELLOW_LIGHT,
  NotifyKind.LIGHT_GREEN: EventKind.LIGHT_GREEN,
  NotifyKind.SHARP_CURVE: EventKind.SHARP_CURVE,
  NotifyKind.SLOW_LEAD: EventKind.SLOW_LEAD,
  NotifyKind.OVER_LIMIT: EventKind.OVER_LIMIT,
  NotifyKind.SPEED_LIMIT: EventKind.SPEED_LIMIT,
  NotifyKind.BREAK: EventKind.BREAK,
}


def log_notification(note: Notification, now_mono: float) -> None:
  feed = get_detection_feed()
  feed.log.appendleft(DetectionEvent(LOG_KINDS[note.kind], datetime.datetime.now().timestamp(), now_mono, distance=note.distance, note=note.text))
  feed.last_event_mono = now_mono


# ---- drawing ----

TOP = 40.0
SCC_CLEARANCE = 360.0     # from the center: the SCC-V and SCC-M chips end 340 px right of it (smart_cruise_control.py)
BUTTON_GAP = 20.0
MIN_WIDTH, MAX_WIDTH = 300.0, 440.0
PAD = 18.0
ICON_R = 34.0
TITLE_SIZE = TEXT_MIN
TEXT_SIZE = 48
DETAIL_SIZE = TEXT_MIN
LINE_GAP = 4.0
MAX_TEXT_LINES = 2
ACCENT_W = 8.0
SLIDE = 40.0              # px the card slides in from


def card_column(rect: rl.Rectangle) -> tuple[float, float]:
  """x and width of the card: right of the SCC chips, left of the experimental mode button, above the threat board."""
  x = rect.x + rect.width / 2 + SCC_CLEARANCE
  right = rect.x + rect.width - UI_CONFIG.border_size - UI_CONFIG.button_size - BUTTON_GAP
  return x, max(MIN_WIDTH, min(MAX_WIDTH, right - x))


class DriveNotifications:
  def __init__(self):
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi = gui_app.font(FontWeight.SEMI_BOLD)
    self.engine = NotificationEngine()
    self._alpha = FirstOrderFilter(0.0, 0.15, 1 / gui_app.target_fps)
    self._note: Notification | None = None   # kept while the card fades out

  def update(self) -> None:
    if not ui_state.drive_notifications or not ui_state.started:
      self.engine.reset()
      self._alpha.x = 0.0
      self._note = None
      return
    now = time.monotonic()
    new = self.engine.update(read_snapshot(ui_state.sm, now))
    if new is not None:
      log_notification(new, now)
    if self.engine.current is not None:
      self._note = self.engine.current.note
    self._alpha.update(1.0 if self.engine.current is not None else 0.0)

  def render(self, rect: rl.Rectangle) -> None:
    alpha = self._alpha.x
    note = self._note
    if not ui_state.drive_notifications or alpha < 0.01 or note is None:
      return
    x, w = card_column(rect)
    x += (1.0 - alpha) * SLIDE
    y = rect.y + TOP
    text_x = x + ACCENT_W + PAD + 2 * ICON_R + 16
    room = x + w - PAD - text_x

    text_size = TEXT_SIZE
    while text_size > TEXT_MIN and measure_text_cached(self._font_bold, note.text, text_size).x > room:
      text_size -= 2
    lines = [note.text]
    if measure_text_cached(self._font_bold, note.text, text_size).x > room:   # too long even at the smallest size: two lines
      lines = wrap_text(self._font_bold, note.text, text_size, int(room))[:MAX_TEXT_LINES]
    detail = note.detail
    while detail and measure_text_cached(self._font_semi, detail, DETAIL_SIZE).x > room:
      detail = detail.rsplit(" ", 1)[0] if " " in detail else ""
    text_h = len(lines) * text_size + (len(lines) - 1) * LINE_GAP
    h = PAD + TITLE_SIZE + LINE_GAP + text_h + (LINE_GAP + DETAIL_SIZE if detail else 0) + PAD

    card = rl.Rectangle(x, y, w, h)
    r = roundness(22, w, h)
    rl.draw_rectangle_rounded(card, r, 12, rgba(BLACK, 0.72 * alpha))
    rl.draw_rectangle_rounded(card, r, 12, rgba(note.rgb, 0.1 * alpha))
    rl.draw_rectangle_rounded_lines_ex(card, r, 12, 3, rgba(note.rgb, 0.8 * alpha))
    rl.draw_rectangle_rounded(rl.Rectangle(x + 6, y + 14, ACCENT_W, h - 28), 1.0, 6, rgba(note.rgb, alpha))

    # time left on the card, a thin bar along the bottom (sticky cards stay full while their reason holds)
    cur = self.engine.current
    if cur is not None and cur.note is note and not cur.rule.sticky:
      left = max(0.0, 1.0 - (time.monotonic() - cur.since) / cur.rule.ttl)
      bar_w = (w - 2 * PAD - ACCENT_W) * left
      rl.draw_rectangle_rounded(rl.Rectangle(x + ACCENT_W + PAD, y + h - 9, bar_w, 4), 1.0, 4, rgba(note.rgb, 0.6 * alpha))

    icon = rl.Vector2(x + ACCENT_W + PAD + ICON_R, y + h / 2)
    self._draw_icon(icon, note, alpha)
    ty = y + PAD
    draw_text(self._font_bold, note.title, TITLE_SIZE, text_x, ty, rgba(note.rgb, alpha), spacing=1.5)
    ty += TITLE_SIZE + LINE_GAP
    for line in lines:
      draw_text(self._font_bold, line, text_size, text_x, ty, rgba(WHITE, alpha))
      ty += text_size + LINE_GAP
    if detail:
      draw_text(self._font_semi, detail, DETAIL_SIZE, text_x, ty, rgba(WHITE, 0.75 * alpha))

  def _draw_icon(self, c: rl.Vector2, note: Notification, alpha: float) -> None:
    rl.draw_circle_v(c, ICON_R, rgba(note.rgb, alpha))
    ink = rgba(INK, alpha)
    cx, cy = c.x, c.y
    icon = note.icon
    if icon in (Icon.LEFT, Icon.RIGHT):
      d = -1 if icon == Icon.LEFT else 1
      draw_triangle((cx + d * 20, cy), (cx + d * 2, cy - 17), (cx + d * 2, cy + 17), ink)
      rl.draw_rectangle_rec(rl.Rectangle(cx - 16 if d > 0 else cx - 2, cy - 6, 18, 12), ink)
    elif icon in (Icon.DOWN, Icon.UP):
      d = 1 if icon == Icon.DOWN else -1
      draw_triangle((cx, cy + d * 20), (cx - 17, cy + d * 2), (cx + 17, cy + d * 2), ink)
      rl.draw_rectangle_rec(rl.Rectangle(cx - 6, cy - 16 if d > 0 else cy - 2, 12, 18), ink)
    elif icon in (Icon.CURVE_LEFT, Icon.CURVE_RIGHT):
      d = -1 if icon == Icon.CURVE_LEFT else 1
      rl.draw_ring(rl.Vector2(cx - d * 16, cy + 16), 18, 28, -90 if d > 0 else 180, 0 if d > 0 else 270, 16, ink)
      draw_triangle((cx + d * 4, cy + 22), (cx + d * 18, cy + 4), (cx + d * 32 - d * 4, cy + 22), ink)
    elif icon == Icon.CLOCK:
      rl.draw_ring(c, 18, 24, 0, 360, 32, ink)
      rl.draw_line_ex(c, rl.Vector2(cx, cy - 13), 5, ink)
      rl.draw_line_ex(c, rl.Vector2(cx + 10, cy), 5, ink)
    elif icon == Icon.STOP_SIGN:
      draw_stop_sign(cx, cy, 26, self._font_bold, alpha)
    elif icon == Icon.PERSON:
      rl.draw_circle_v(rl.Vector2(cx, cy - 14), 7, ink)
      rl.draw_rectangle_rounded(rl.Rectangle(cx - 7, cy - 5, 14, 18), 0.5, 6, ink)
      rl.draw_line_ex(rl.Vector2(cx - 3, cy + 11), rl.Vector2(cx - 8, cy + 23), 5, ink)
      rl.draw_line_ex(rl.Vector2(cx + 3, cy + 11), rl.Vector2(cx + 8, cy + 23), 5, ink)
    elif icon in (Icon.SIGNAL_RED, Icon.SIGNAL_YELLOW, Icon.SIGNAL_GREEN):
      rl.draw_circle_v(c, ICON_R, rgba(BLACK, alpha))   # a signal head reads best on black
      draw_signal_head(cx, cy, 0.72, {Icon.SIGNAL_RED: 1, Icon.SIGNAL_YELLOW: 2, Icon.SIGNAL_GREEN: 3}[icon], alpha)
    elif icon == Icon.WAIT:
      rl.draw_rectangle_rounded(rl.Rectangle(cx - 13, cy - 15, 9, 30), 0.5, 4, ink)
      rl.draw_rectangle_rounded(rl.Rectangle(cx + 4, cy - 15, 9, 30), 0.5, 4, ink)
    else:
      draw_text_centered(self._font_bold, "!", 52, cx, cy, ink)
