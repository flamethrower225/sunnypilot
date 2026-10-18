"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Pace Border: a glow around the screen edge that tells the driver, without reading anything, whether they are slowing enough.

Each frame it works out the deceleration needed to settle at the gap the longitudinal planner targets behind the lead, and the
deceleration needed to stop just short of a planned stop (a light or sign, from the stop detector). The larger of the two, minus
the deceleration the car already has, is the shortfall. No shortfall, no border. As it grows the glow turns yellow, then orange,
then red, flashes faster and reaches further into the screen. When stopped and it is time to go, the edge pulses green.
"""
import math
import time

import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import ZONE_RGB, rgba, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import STOP_DISTANCE, GapZone, get_t_follow
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.detection import StopPhase, get_detection_feed
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.follow_ghost import desired_gap
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.stop_indicator import update_feed
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app

STOP_MARGIN = 2.0          # m, aim to stop this far short of the planned stop point
MIN_ROOM = 0.5             # m, room below this is treated as this, so the demand stays finite
LEAD_BRAKE_SHARE = 0.5     # share of the lead's own braking added to what we need
INSIDE_GAP_DEMAND = 2.0    # m/s^2 asked for when sitting at half the target gap without closing, scaling with how far inside
SHORTFALL_START = 0.4      # m/s^2 of missing deceleration before the border shows
SHORTFALL_FULL = 3.5       # m/s^2 at which it is at full strength
MIN_SPEED = 1.0            # m/s, below this there is nothing to slow for

FLASH_HZ = (0.8, 4.5)      # flashes per second at the lowest and highest urgency
DEPTH_PX = (60.0, 160.0)   # how far the glow reaches into the screen
PEAK_ALPHA = (0.45, 0.95)  # opacity of the outermost band at the top of a flash
FLASH_FLOOR = (0.35, 0.1)       # share of the glow left between flashes: steadier when mild, a hard strobe when urgent
GO_HZ, GO_DEPTH, GO_ALPHA, GO_TIME = 1.0, 90.0, 0.55, 2.5

RISE_RC, FALL_RC = 0.08, 0.45  # s: react fast, calm down slowly so it doesn't flicker
EDGE_LAYERS = 14
EDGE_RADIUS = 36.0

YELLOW, ORANGE, RED, GREEN = (ZONE_RGB[z] for z in (GapZone.CAUTION, GapZone.CLOSE, GapZone.CRITICAL, GapZone.GOOD))


# *** what the driver needs to do ***

def lead_demand(d_rel: float, v_ego: float, v_lead: float, a_lead: float, t_follow: float) -> float:
  """Deceleration (m/s^2) needed to settle at the planner's target gap behind the lead, rather than closer."""
  target = max(desired_gap(v_ego, v_lead, t_follow), STOP_DISTANCE)
  closing = v_ego - v_lead
  room = d_rel - target
  demand = 0.0
  if closing > 0:
    if room > MIN_ROOM:
      demand = closing ** 2 / (2 * room)
    else:  # already at or inside the target gap and still closing: what it takes not to get closer than the stop distance
      demand = closing ** 2 / (2 * max(d_rel - STOP_DISTANCE, MIN_ROOM))
  if room < 0:  # inside the target gap even if not closing: ease off, more the further inside
    demand = max(demand, INSIDE_GAP_DEMAND * min(-room / target, 1.0) * 2)
  return demand + LEAD_BRAKE_SHARE * max(-a_lead, 0.0)


def stop_demand(stop_x: float, v_ego: float) -> float:
  """Deceleration (m/s^2) needed to stop STOP_MARGIN short of the planned stop point."""
  return v_ego ** 2 / (2 * max(stop_x - STOP_MARGIN, MIN_ROOM))


def urgency(shortfall: float) -> float:
  """0 when the car is slowing enough (or nearly), 1 at SHORTFALL_FULL of missing deceleration."""
  return min(max((shortfall - SHORTFALL_START) / (SHORTFALL_FULL - SHORTFALL_START), 0.0), 1.0)


def mix(a: tuple, b: tuple, t: float) -> tuple:
  return tuple(x + (y - x) * t for x, y in zip(a, b, strict=True))


def urgency_color(u: float) -> tuple:
  """Yellow (ease off) through orange (brake) to red (brake hard)."""
  return mix(YELLOW, ORANGE, u / 0.5) if u < 0.5 else mix(ORANGE, RED, (u - 0.5) / 0.5)


def lerp(pair: tuple[float, float], t: float) -> float:
  return pair[0] + (pair[1] - pair[0]) * t


# *** the overlay ***

class PaceBorder:
  def __init__(self):
    self._urgency = FirstOrderFilter(0.0, FALL_RC, 1 / gui_app.target_fps)
    self._phase = 0.0
    self._last = time.monotonic()
    self._go_until = 0.0
    self._go_prev = False
    self.reason = ""  # "lead" or "stop", for tests and debugging

  def _demand(self) -> tuple[float, str]:
    sm = ui_state.sm
    v_ego = sm['carState'].vEgo
    if v_ego < MIN_SPEED or sm.recv_frame['carState'] < ui_state.started_frame:
      return 0.0, ""
    best, reason = 0.0, ""
    lead = sm['radarState'].leadOne
    radar_ok = sm.valid['radarState'] and sm.recv_frame['radarState'] >= ui_state.started_frame
    if radar_ok and lead.present:
      d = lead_demand(lead.dRel, v_ego, lead.vLeadK, lead.aLeadK, get_t_follow(sm['selfdriveState'].personality))
      if d > best:
        best, reason = d, "lead"
    stop = get_detection_feed().stop
    if stop.phase == StopPhase.APPROACH and stop.stop_x is not None and not stop.car_ahead:
      d = stop_demand(stop.stop_x, v_ego)
      if d > best:
        best, reason = d, "stop"
    return best, reason

  def update(self) -> None:
    now = time.monotonic()
    dt, self._last = min(now - self._last, 0.2), now
    if not ui_state.pace_border or not ui_state.started:
      self._urgency.x = 0.0
      self._go_until = 0.0
      return
    update_feed()
    demand, self.reason = self._demand()
    decel = max(-ui_state.sm['carState'].aEgo, 0.0)
    target = urgency(demand - decel)
    self._urgency.update_alpha(RISE_RC if target > self._urgency.x else FALL_RC)
    self._urgency.update(target)
    self._phase = (self._phase + lerp(FLASH_HZ, self._urgency.x) * dt) % 1.0  # integrate so a changing rate never jumps

    stop = get_detection_feed().stop
    e2e = ui_state.sm['longitudinalPlanSP'].e2eAlerts
    go = stop.phase == StopPhase.GO or bool(e2e.greenLightAlert) or bool(e2e.leadDepartAlert)
    if go and not self._go_prev:
      self._go_until = now + GO_TIME
    self._go_prev = go

  @property
  def level(self) -> float:
    return self._urgency.x

  def render(self, rect: rl.Rectangle) -> None:
    if not ui_state.pace_border:
      return
    u = self._urgency.x
    if u > 0.02:
      beat = 0.5 - 0.5 * math.cos(2 * math.pi * self._phase)
      strength = lerp(FLASH_FLOOR, u) + (1 - lerp(FLASH_FLOOR, u)) * beat
      draw_edge_glow(rect, urgency_color(u), lerp(DEPTH_PX, u), lerp(PEAK_ALPHA, u) * strength * min(u * 4, 1.0))
    elif time.monotonic() < self._go_until:
      beat = 0.5 - 0.5 * math.cos(2 * math.pi * GO_HZ * time.monotonic())
      draw_edge_glow(rect, GREEN, GO_DEPTH, GO_ALPHA * (0.3 + 0.7 * beat))


def draw_edge_glow(rect: rl.Rectangle, rgb: tuple, depth: float, alpha: float) -> None:
  """Nested rounded outlines, each inset deeper and fainter, so the edge fades into the picture. Bands overlap by half so there are
  no seams, and the corner radius shrinks with the inset so they stay parallel."""
  if alpha < 0.01:
    return
  step = depth / EDGE_LAYERS
  thickness = 2 * step
  for i in range(EDGE_LAYERS):
    inset = i * step
    layer = rl.Rectangle(rect.x + inset, rect.y + inset, rect.width - 2 * inset, rect.height - 2 * inset)
    radius = EDGE_RADIUS - inset
    if radius < thickness:  # raylib's thick outline breaks when the thickness passes the corner radius
      radius = 0.0
    falloff = (1.0 - i / EDGE_LAYERS) ** 2
    rl.draw_rectangle_rounded_lines_ex(layer, roundness(radius, layer.width, layer.height) if radius else 0.0, 12, thickness,
                                       rgba(rgb, min(alpha * falloff, 1.0)))
