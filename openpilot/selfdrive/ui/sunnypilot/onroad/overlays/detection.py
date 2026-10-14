"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

What the driving model and planner detect, as timestamped events.

The driving model has no traffic light or stop sign classifier. What it does have is its plan: when it intends to stop,
its predicted speed reaches zero inside the 10 s horizon. With no car ahead at that point, the stop is almost always for
a red light, a stop sign or a crossing. StopDetector turns that into a stop with a distance, a cause and the moment it
was first seen. EventDetector adds the other things the system picks up: leads, cut-ins, curves and hard braking.
"""
import math
from collections import deque
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

V_STOP = 0.5               # m/s, planned speed that counts as stopped
MIN_APPROACH_SPEED = 2.0   # m/s, below this we're creeping or stopped, not approaching a stop
STOPPED_SPEED = 0.3        # m/s, same as drive_helpers.should_stop
MAX_STOP_RANGE = 250.0     # m
CONFIRM_FRAMES = 3         # model frames the plan must agree before a stop counts (0.15 s at 20 Hz)
RELEASE_TIME = 0.6         # s without a planned stop before an approach is called off
GO_X = 30.0                # m, planned travel that means the model wants to go (sunnypilot's green light threshold)
GO_CONFIRM = 0.3           # s
LEAD_MARGIN = 8.0          # m, a lead this close to the stop point is what we're stopping for
SHOW_GO = 2.5              # s the GO state stays up
SHOW_CLEARED = 2.0         # s the cleared state stays up
STOP_FILTER_RC = 0.15      # s

LEAD_CONFIRM = 0.3         # s a new lead must stay before it's logged
LEAD_LOST_TIME = 1.0       # s
CUT_IN_JUMP = 5.0          # m, and 25% of the distance
CURVE_ON, CURVE_OFF = 1.6, 1.2  # m/s^2 predicted lateral acceleration
CURVE_RANGE = 120.0        # m

LOG_SIZE = 6


class StopPhase(IntEnum):
  IDLE = 0
  APPROACH = 1
  STOPPED = 2
  GO = 3
  CLEARED = 4


class EventKind(IntEnum):
  STOP = 0
  CLEARED = 1
  GO = 2
  GREEN_LIGHT = 3
  LEAD = 4
  LEAD_LOST = 5
  CUT_IN = 6
  LEAD_DEPART = 7
  CURVE = 8
  HARD_BRAKE = 9


@dataclass(frozen=True)
class DetectionEvent:
  kind: EventKind
  wall_time: float                 # time.time() of the model frame where it first showed up
  mono_time: float
  distance: float | None = None    # m
  value: float | None = None       # kind specific: lateral accel for CURVE
  car_ahead: bool | None = None    # STOP and GO: whether a car is the reason
  radar: bool | None = None        # LEAD and CUT_IN: radar matched


def predicted_stop_distance(position_x, velocity_x, v_stop: float = V_STOP) -> float | None:
  """Distance (m) along the model's plan where its predicted speed first drops below v_stop, or None."""
  n = min(len(position_x), len(velocity_x))
  if n and velocity_x[0] < v_stop:
    return max(float(position_x[0]), 0.0)
  for i in range(1, n):
    v0, v1 = velocity_x[i - 1], velocity_x[i]
    if v1 < v_stop:
      frac = 1.0 if v0 <= v_stop else (v0 - v_stop) / max(v0 - v1, 1e-6)
      x = position_x[i - 1] + frac * (position_x[i] - position_x[i - 1])
      return max(float(x), 0.0)
  return None


def stop_is_for_car(stop_x: float | None, lead_present: bool, lead_d: float) -> bool:
  if not lead_present:
    return False
  if stop_x is None:
    return lead_d < LEAD_MARGIN * 2
  return lead_d <= stop_x + LEAD_MARGIN


@dataclass
class _Run:
  wall: float
  mono: float
  stop_x: float
  frames: int = 1


class StopDetector:
  def __init__(self):
    self.phase = StopPhase.IDLE
    self.stop_x: float | None = None       # smoothed distance to the planned stop
    self.car_ahead = False
    self.onset: DetectionEvent | None = None
    self.phase_since = 0.0                 # mono time the current phase began
    self._run: _Run | None = None
    self._miss = 0.0
    self._go = 0.0

  def update(self, now_wall: float, now_mono: float, dt: float, position_x, velocity_x, v_ego: float,
             lead_present: bool, lead_d: float) -> list[DetectionEvent]:
    """Call once per model frame."""
    events: list[DetectionEvent] = []
    raw = predicted_stop_distance(position_x, velocity_x)
    if raw is not None and raw > MAX_STOP_RANGE:
      raw = None
    endpoint = float(position_x[-1]) if len(position_x) else 0.0
    moving = v_ego > MIN_APPROACH_SPEED
    car = stop_is_for_car(raw, lead_present, lead_d)

    def enter(phase: StopPhase) -> None:
      self.phase = phase
      self.phase_since = now_mono

    # a planned stop that holds for a few frames starts an approach, from any phase except an approach or a stop
    if self.phase in (StopPhase.IDLE, StopPhase.GO, StopPhase.CLEARED):
      if moving and raw is not None:
        if self._run is None:
          self._run = _Run(now_wall, now_mono, raw)
        else:
          self._run.frames += 1
        if self._run.frames >= CONFIRM_FRAMES:
          self.onset = DetectionEvent(EventKind.STOP, self._run.wall, self._run.mono, distance=self._run.stop_x, car_ahead=car)
          events.append(self.onset)
          self.stop_x, self.car_ahead, self._miss, self._run = raw, car, 0.0, None
          enter(StopPhase.APPROACH)
          return events
      else:
        self._run = None

    if self.phase == StopPhase.APPROACH:
      if raw is not None:
        alpha = dt / (STOP_FILTER_RC + dt)
        self.stop_x = raw if self.stop_x is None else self.stop_x + alpha * (raw - self.stop_x)
        self.car_ahead = car
        self._miss = 0.0
      else:
        self._miss += dt
      if v_ego < STOPPED_SPEED:
        self._go = 0.0
        enter(StopPhase.STOPPED)
      elif self._miss >= RELEASE_TIME:
        events.append(DetectionEvent(EventKind.CLEARED, now_wall, now_mono, car_ahead=self.car_ahead))
        enter(StopPhase.CLEARED)

    elif self.phase == StopPhase.STOPPED:
      self.car_ahead = lead_present and lead_d < LEAD_MARGIN * 2
      if endpoint > GO_X:
        self._go += dt
        if self._go >= GO_CONFIRM - 1e-6:
          events.append(DetectionEvent(EventKind.GO, now_wall, now_mono, car_ahead=self.car_ahead))
          enter(StopPhase.GO)
      else:
        self._go = 0.0
      if self.phase == StopPhase.STOPPED and moving:
        enter(StopPhase.IDLE)

    elif self.phase == StopPhase.IDLE:
      # stopped without having seen the approach, e.g. the UI started at a light
      if v_ego < STOPPED_SPEED and endpoint < GO_X:
        self._go = 0.0
        self.onset = None
        enter(StopPhase.STOPPED)

    elif self.phase == StopPhase.GO and now_mono - self.phase_since > SHOW_GO:
      enter(StopPhase.IDLE)
    elif self.phase == StopPhase.CLEARED and now_mono - self.phase_since > SHOW_CLEARED:
      enter(StopPhase.IDLE)

    return events


class EventDetector:
  """Everything else worth logging: leads, cut-ins, sunnypilot's e2e alerts, curves and predicted hard braking."""

  def __init__(self):
    self._lead_on = False
    self._lead_since: float | None = None
    self._lead_gone: float | None = None
    self._lead_d = 0.0
    self._curve_on = False
    self._hard_brake = False
    self._green = False
    self._depart = False

  def update(self, now_wall: float, now_mono: float, lead_present: bool, lead_d: float, lead_radar: bool,
             lat_accels, position_x, hard_brake: bool, green_light: bool, lead_depart: bool) -> list[DetectionEvent]:
    events: list[DetectionEvent] = []

    def emit(kind: EventKind, **kw) -> None:
      events.append(DetectionEvent(kind, now_wall, now_mono, **kw))

    # leads
    if lead_present:
      self._lead_gone = None
      if not self._lead_on:
        self._lead_since = now_mono if self._lead_since is None else self._lead_since
        if now_mono - self._lead_since >= LEAD_CONFIRM - 1e-6:
          self._lead_on = True
          emit(EventKind.LEAD, distance=lead_d, radar=lead_radar)
      elif self._lead_d - lead_d > max(CUT_IN_JUMP, 0.25 * self._lead_d):
        emit(EventKind.CUT_IN, distance=lead_d, radar=lead_radar)
      self._lead_d = lead_d
    else:
      self._lead_since = None
      if self._lead_on:
        self._lead_gone = now_mono if self._lead_gone is None else self._lead_gone
        if now_mono - self._lead_gone >= LEAD_LOST_TIME - 1e-6:
          self._lead_on = False
          self._lead_gone = None
          emit(EventKind.LEAD_LOST)

    # curves, from the model's predicted lateral acceleration
    n = min(len(lat_accels), len(position_x))
    if n:
      lat = np.asarray(lat_accels[:n], dtype=float)
      xs = np.asarray(position_x[:n], dtype=float)
      ahead = lat[xs <= CURVE_RANGE]
      peak = float(ahead.max()) if ahead.size else 0.0
      if not self._curve_on and peak > CURVE_ON:
        self._curve_on = True
        i = int(np.argmax(np.where(xs <= CURVE_RANGE, lat, -math.inf)))
        emit(EventKind.CURVE, distance=float(xs[i]), value=peak)
      elif self._curve_on and peak < CURVE_OFF:
        self._curve_on = False

    for attr, now, kind in (("_hard_brake", hard_brake, EventKind.HARD_BRAKE), ("_green", green_light, EventKind.GREEN_LIGHT),
                            ("_depart", lead_depart, EventKind.LEAD_DEPART)):
      if now and not getattr(self, attr):
        emit(kind)
      setattr(self, attr, now)

    return events


class DetectionFeed:
  """One per UI process, shared by the stop card, the stop line and the log. Advances once per model frame."""

  def __init__(self):
    self.stop = StopDetector()
    self.events = EventDetector()
    self.log: deque[DetectionEvent] = deque(maxlen=LOG_SIZE)
    self.last_event_mono = -math.inf
    self._last_model_frame = -1
    self._last_mono: float | None = None

  def update(self, sm, started_frame: int, now_wall: float, now_mono: float) -> None:
    model_frame = sm.recv_frame['modelV2']
    if model_frame < started_frame or model_frame == self._last_model_frame:
      return
    self._last_model_frame = model_frame
    dt = 0.05 if self._last_mono is None else float(np.clip(now_mono - self._last_mono, 0.01, 0.5))
    self._last_mono = now_mono

    model = sm['modelV2']
    radar_ok = sm.valid['radarState'] and sm.recv_frame['radarState'] >= started_frame
    lead = sm['radarState'].leadOne
    lead_present = bool(radar_ok and lead.present)
    v_ego = sm['carState'].vEgo
    position_x, velocity_x = list(model.position.x), list(model.velocity.x)
    yaw, vx = list(model.orientationRate.z), list(model.velocity.x)
    n = min(len(yaw), len(vx))
    lat = [abs(yaw[i] * vx[i]) for i in range(n)]
    e2e = sm['longitudinalPlanSP'].e2eAlerts

    new = self.stop.update(now_wall, now_mono, dt, position_x, velocity_x, v_ego, lead_present, lead.dRel)
    new += self.events.update(now_wall, now_mono, lead_present, lead.dRel, lead.radar, lat, position_x,
                              model.meta.hardBrakePredicted, e2e.greenLightAlert, e2e.leadDepartAlert)
    for event in new:
      self.log.appendleft(event)
      self.last_event_mono = now_mono


_feed: DetectionFeed | None = None


def get_detection_feed() -> DetectionFeed:
  global _feed
  if _feed is None:
    _feed = DetectionFeed()
  return _feed
