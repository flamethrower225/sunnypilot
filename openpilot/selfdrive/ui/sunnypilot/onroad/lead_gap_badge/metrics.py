"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
from dataclasses import dataclass, replace
from enum import IntEnum

from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter

# Mirrors get_T_FOLLOW() and STOP_DISTANCE in selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py.
# Not imported from there because that module pulls in the acados solver.
T_FOLLOW = {
  log.LongitudinalPersonality.aggressive: 1.25,
  log.LongitudinalPersonality.standard: 1.45,
  log.LongitudinalPersonality.relaxed: 1.75,
}
STOP_DISTANCE = 6.0  # m

LOW_SPEED = 3.0           # m/s, below this a time gap is meaningless, so the badge shows distance
MAX_GAP_DISPLAY = 9.9     # s
TTC_DISPLAY = 6.0         # s, time to collision is shown while closing and below this
MIN_CLOSING_SPEED = 0.3   # m/s
OPEN_GAP = 3.0            # s, beyond this the lead isn't limiting
SNAP_DISTANCE = 5.0       # m, a jump this large (and 25% of dRel) is a new lead, so skip the smoothing
FT_PER_M = 3.28084

# zone thresholds: (gap s, low speed distance m, time to collision s)
CRITICAL_LIMITS = (0.6, 3.0, 2.5)
CLOSE_LIMITS = (1.0, 4.5, 4.0)
CAUTION_TTC = 6.0
CAUTION_T_FOLLOW_FACTOR = 0.9


class LeadGapBadgeStyle(IntEnum):
  OFF = 0
  DIAL = 1
  STRIP = 2
  LOCK = 3
  BLOCK = 4
  PAINT = 5


class GapZone(IntEnum):
  # ordered by severity, NONE means no lead
  NONE = 0
  OPEN = 1
  GOOD = 2
  CAUTION = 3
  CLOSE = 4
  CRITICAL = 5


def get_t_follow(personality) -> float:
  personality = int(getattr(personality, "raw", personality))
  return T_FOLLOW.get(personality, T_FOLLOW[log.LongitudinalPersonality.standard])


def get_target_gap(t_follow: float, v_ego: float) -> float:
  """Steady-state time gap the longitudinal MPC settles at when the lead holds speed."""
  return t_follow + STOP_DISTANCE / max(v_ego, 1.0)


def classify_zone(gap: float, d_rel: float, ttc: float, t_follow: float, low_speed: bool) -> GapZone:
  crit_gap, crit_dist, crit_ttc = CRITICAL_LIMITS
  close_gap, close_dist, close_ttc = CLOSE_LIMITS

  if ttc < crit_ttc or (d_rel < crit_dist if low_speed else gap < crit_gap):
    return GapZone.CRITICAL
  if ttc < close_ttc or (d_rel < close_dist if low_speed else gap < close_gap):
    return GapZone.CLOSE
  if ttc < CAUTION_TTC or (d_rel < CAUTION_T_FOLLOW_FACTOR * STOP_DISTANCE if low_speed else gap < CAUTION_T_FOLLOW_FACTOR * t_follow):
    return GapZone.CAUTION
  if not low_speed and gap > OPEN_GAP:
    return GapZone.OPEN
  return GapZone.GOOD


class ZoneHysteresis:
  """Escalation to CRITICAL and lead appearing/disappearing apply at once. Other changes have to hold
  for a moment so the color doesn't flicker on a threshold."""

  def __init__(self, dt: float, escalate_time: float = 0.2, relax_time: float = 0.6):
    self.zone = GapZone.NONE
    self._dt = dt
    self._escalate_time = escalate_time
    self._relax_time = relax_time
    self._candidate = GapZone.NONE
    self._timer = 0.0

  def update(self, raw: GapZone) -> GapZone:
    if raw == self.zone:
      self._candidate = raw
      self._timer = 0.0
      return self.zone

    if raw == GapZone.CRITICAL or GapZone.NONE in (raw, self.zone):
      self.zone = raw
      self._candidate = raw
      self._timer = 0.0
      return self.zone

    if raw != self._candidate:
      self._candidate = raw
      self._timer = 0.0

    self._timer += self._dt
    hold = self._escalate_time if raw > self.zone else self._relax_time
    if self._timer >= hold - 1e-6:
      self.zone = raw
      self._timer = 0.0
    return self.zone


@dataclass(frozen=True)
class LeadGapReading:
  present: bool = False
  zone: GapZone = GapZone.NONE
  d_rel: float = 0.0       # m
  v_ego: float = 0.0       # m/s
  v_lead: float = 0.0      # m/s
  v_rel: float = 0.0       # m/s, negative when closing
  gap: float = math.inf    # s
  ttc: float = math.inf    # s
  t_follow: float = T_FOLLOW[log.LongitudinalPersonality.standard]
  target_gap: float = 0.0  # s
  low_speed: bool = False

  @property
  def show_ttc(self) -> bool:
    return self.present and self.ttc < TTC_DISPLAY


class LeadGapMetrics:
  """Smooths radarState.leadOne for display. Zones are classified on the raw values (through hysteresis)
  so they never lag a cut-in, while the numbers on screen are filtered so they don't jitter."""

  def __init__(self, dt: float, rc: float = 0.25):
    self._filters = {name: FirstOrderFilter(0.0, rc, dt, initialized=False) for name in ("d_rel", "v_ego", "v_lead", "v_rel")}
    self._zone = ZoneHysteresis(dt)
    self._present = False
    self.reading = LeadGapReading()

  def update(self, present: bool, d_rel: float, v_rel: float, v_lead: float, v_ego: float, personality) -> LeadGapReading:
    if not present:
      self._present = False
      # keep the last numbers so the badge can fade out on them
      self.reading = replace(self.reading, present=False, zone=self._zone.update(GapZone.NONE))
      return self.reading

    d_filter = self._filters["d_rel"]
    new_lead = not self._present or (d_filter.initialized and abs(d_rel - d_filter.x) > max(SNAP_DISTANCE, 0.25 * d_rel))
    if new_lead:
      for f in self._filters.values():
        f.initialized = False
    self._present = True

    t_follow = get_t_follow(personality)
    raw_gap, raw_ttc = self._gap_and_ttc(d_rel, v_ego, v_rel)
    zone = self._zone.update(classify_zone(raw_gap, d_rel, raw_ttc, t_follow, v_ego < LOW_SPEED))

    d = max(d_filter.update(d_rel), 0.0)
    v = max(self._filters["v_ego"].update(v_ego), 0.0)
    vl = max(self._filters["v_lead"].update(v_lead), 0.0)
    vr = self._filters["v_rel"].update(v_rel)
    gap, ttc = self._gap_and_ttc(d, v, vr)

    self.reading = LeadGapReading(present=True, zone=zone, d_rel=d, v_ego=v, v_lead=vl, v_rel=vr, gap=gap, ttc=ttc,
                                  t_follow=t_follow, target_gap=get_target_gap(t_follow, v), low_speed=v < LOW_SPEED)
    return self.reading

  @staticmethod
  def _gap_and_ttc(d_rel: float, v_ego: float, v_rel: float) -> tuple[float, float]:
    gap = d_rel / v_ego if v_ego > 0.1 else math.inf
    closing = -v_rel
    ttc = d_rel / closing if closing > MIN_CLOSING_SPEED else math.inf
    return gap, ttc


# ---- formatting ----

def format_gap(gap: float) -> str:
  if not math.isfinite(gap):
    return "–"
  return f"{min(gap, MAX_GAP_DISPLAY):.1f}"


def format_distance(d_rel: float, is_metric: bool) -> tuple[str, str]:
  if is_metric:
    return f"{max(d_rel, 0.0):.0f}", "m"
  return f"{max(d_rel, 0.0) * FT_PER_M:.0f}", "ft"


def primary_value(reading: LeadGapReading, is_metric: bool) -> tuple[str, str]:
  """Main number for every badge: seconds normally, distance at a crawl."""
  if reading.low_speed:
    return format_distance(reading.d_rel, is_metric)
  return format_gap(reading.gap), "s"


def speed_value(v: float, is_metric: bool) -> int:
  return round(max(v, 0.0) * (CV.MS_TO_KPH if is_metric else CV.MS_TO_MPH))


def speed_delta(v_rel: float, is_metric: bool) -> int:
  """Lead speed minus ego speed in display units, negative when the lead is slower."""
  return round(v_rel * (CV.MS_TO_KPH if is_metric else CV.MS_TO_MPH))
