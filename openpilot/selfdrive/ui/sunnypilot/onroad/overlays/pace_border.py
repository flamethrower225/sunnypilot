"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Pace Border: a glow around the screen edge that tells the driver, without reading anything, whether they are slowing enough.

Each frame it works out the deceleration needed to settle at the gap the longitudinal planner targets behind the lead, the
deceleration needed to stop just short of a planned stop (a light or sign, from the stop detector), and the deceleration needed to
be at a bend's advisory speed before its tightest point (the curve preview's speed), and with Camera Object Detection on, the
deceleration needed to stop for a red light (or a yellow there is room to stop for), a stop sign or a flashing red. A sign or flashing
red stops asking once the car has stopped at it, so pulling away after the stop is quiet. The largest of them, minus
the deceleration the car already has, is the shortfall. No shortfall, no border. As it grows the glow turns yellow, then orange,
then red, flashes faster and reaches further into the screen. When stopped and it is time to go, the edge pulses green.

Along the bottom edge sits a PAPI, like the row of four lights beside a runway that shows a pilot the glide path. Here the glide
path is the braking needed, and the lights compare it with the braking the car has, whenever there is something to slow for:

  red red red red (flashing)  far too little braking: brake hard
  white red red red           too little
  white white red red         on profile
  white white white red       more than needed: ease off
  white white white white     far more than needed: stopping well short

As on a runway, more red means less margin. The glow is the alarm; the PAPI is the precise reading, and shows even on profile.
"""
import math
import time

import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, ZONE_RGB, rgba
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import STOP_DISTANCE, GapZone, get_t_follow
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays import hard_brake_flash
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.curve_hologram import curve_ahead, curve_decel_needed
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.detection import StopPhase, get_detection_feed
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.edge_glow import draw_edge_glow
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.follow_ghost import desired_gap
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.light_view import current_light, current_stop_sign
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
LIGHT_RED, LIGHT_YELLOW, LIGHT_GREEN = 1, 2, 3   # SceneDetectionSP.LightState
GREEN_FRESH = 2.0          # s, a light that turned green this recently pulses the edge green while stopped
YELLOW_STOP_DECEL = 3.0    # m/s^2, a yellow that needs more than this to stop for is left out
SERVED_SPEED = 0.5         # m/s, stopped this slowly ...
SERVED_RANGE = 15.0        # m, ... this close to a stop sign or flashing red, it has been served

FLASH_HZ = (0.8, 4.5)      # flashes per second at the lowest and highest urgency
DEPTH_PX = (60.0, 160.0)   # how far the glow reaches into the screen
PEAK_ALPHA = (0.45, 0.95)  # opacity of the outermost band at the top of a flash
FLASH_FLOOR = (0.35, 0.1)       # share of the glow left between flashes: steadier when mild, a hard strobe when urgent
GO_HZ, GO_DEPTH, GO_ALPHA, GO_TIME = 1.0, 90.0, 0.55, 2.5

RISE_RC, FALL_RC = 0.08, 0.45  # s: react fast, calm down slowly so it doesn't flicker
EDGE_LAYERS = 14

SHOW_DEMAND = 0.6          # m/s^2 of needed braking before the PAPI comes on (something to slow for)
HIDE_DEMAND = 0.3          # and below this it goes off again
RED_THRESHOLDS = (-1.5, -0.5, 0.5, 1.5)  # deviation (needed minus actual braking, m/s^2) at which each red light comes on
HYSTERESIS = 0.15          # m/s^2, a light changes only once the deviation is this far past its threshold
DEMAND_RC, DECEL_RC, PAPI_FADE_RC = 0.15, 0.25, 0.2  # s

PAPI_WIDTH = 0.56          # share of the screen width the row of lights spans
PAPI_GAP = 36.0            # px between lights
PAPI_CORE_H = 16.0         # px, the bright bar at the bottom edge
PAPI_BOTTOM = 10.0         # px from the bottom edge
PAPI_HALO_H = 120.0        # px the light's glow reaches up into the picture
PAPI_RED = (255, 58, 52)
PAPI_WHITE = (255, 250, 240)

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


def red_lights(deviation: float, previous: int = 2) -> int:
  """How many of the four PAPI lights are red for a deviation (needed minus actual braking). Each threshold has hysteresis, so a
  deviation sitting on a boundary doesn't make a light flicker."""
  count = sum(deviation >= t for t in RED_THRESHOLDS)
  if count > previous and deviation < RED_THRESHOLDS[count - 1] + HYSTERESIS:
    count -= 1
  elif count < previous and deviation > RED_THRESHOLDS[count] - HYSTERESIS:
    count += 1
  return count


def light_colors(reds: int) -> list[tuple]:
  """Left to right, white lights first, as a PAPI reads from the approach."""
  return [PAPI_WHITE] * (4 - reds) + [PAPI_RED] * reds


# *** the overlay ***

class PaceBorder:
  def __init__(self):
    dt = 1 / gui_app.target_fps
    self._urgency = FirstOrderFilter(0.0, FALL_RC, dt)
    self._demand_f = FirstOrderFilter(0.0, DEMAND_RC, dt)
    self._decel_f = FirstOrderFilter(0.0, DECEL_RC, dt)
    self._papi_alpha = FirstOrderFilter(0.0, PAPI_FADE_RC, dt)
    self.papi_shown = False
    self.reds = 2
    self._phase = 0.0
    self._last = time.monotonic()
    self._go_until = 0.0
    self._go_prev = False
    self._served = False  # stopped at the stop sign or flashing red ahead: it asks nothing more until it is gone
    self.reason = ""  # "lead", "stop", "light", "sign" or "curve", for tests and debugging

  @staticmethod
  def _full_stop(sm) -> float | None:
    """Where the stop sign or flashing red ahead wants a full stop, m: they are stopped at and then driven through."""
    light = current_light(sm)
    if light is not None and light.flashing and light.state == LIGHT_RED:
      return light.stop_at
    sign = current_stop_sign(sm)
    return sign.stop_distance if sign is not None else None

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
    def ahead_of_lead(stop_x: float) -> bool:
      return math.isfinite(stop_x) and not (radar_ok and lead.present and lead.dRel < stop_x)

    light = current_light(sm)
    if light is not None and not light.flashing and light.state in (LIGHT_RED, LIGHT_YELLOW) and ahead_of_lead(light.stop_at):
      d = stop_demand(light.stop_at, v_ego)
      # a yellow you can't stop for gently is one to go through, not to brake hard for
      if d > best and (light.state == LIGHT_RED or d <= YELLOW_STOP_DECEL):
        best, reason = d, "light"
    full_stop = self._full_stop(sm)
    if full_stop is not None and not self._served and ahead_of_lead(full_stop):
      d = stop_demand(full_stop, v_ego)
      if d > best:
        best, reason = d, "sign"
    curve = curve_ahead(sm, ui_state.started_frame)
    if curve is not None:
      d = curve_decel_needed(v_ego, curve.speed, curve.x)
      if d > best:
        best, reason = d, "curve"
    return best, reason

  def update(self) -> None:
    now = time.monotonic()
    dt, self._last = min(now - self._last, 0.2), now
    if not ui_state.pace_border or not ui_state.started:
      self._urgency.x = 0.0
      self._served = False
      self._papi_alpha.x = 0.0
      self.papi_shown = False
      self._go_until = 0.0
      return
    update_feed()
    full_stop = self._full_stop(ui_state.sm)
    if full_stop is None:
      self._served = False
    elif ui_state.sm['carState'].vEgo < SERVED_SPEED and full_stop < SERVED_RANGE:
      self._served = True
    demand, self.reason = self._demand()
    decel = max(-ui_state.sm['carState'].aEgo, 0.0)
    target = urgency(demand - decel)
    self._urgency.update_alpha(RISE_RC if target > self._urgency.x else FALL_RC)
    self._urgency.update(target)
    self._phase = (self._phase + lerp(FLASH_HZ, self._urgency.x) * dt) % 1.0  # integrate so a changing rate never jumps

    # PAPI: smoothed needed vs actual braking, shown while there is something to slow for
    self._demand_f.update(demand)
    self._decel_f.update(decel)
    self.papi_shown = self._demand_f.x >= (HIDE_DEMAND if self.papi_shown else SHOW_DEMAND)
    self.reds = red_lights(self._demand_f.x - self._decel_f.x, self.reds) if self.papi_shown else 2
    self._papi_alpha.update(1.0 if self.papi_shown else 0.0)

    stop = get_detection_feed().stop
    e2e = ui_state.sm['longitudinalPlanSP'].e2eAlerts
    light = current_light()
    light_went_green = light is not None and light.state == LIGHT_GREEN and light.age < GREEN_FRESH and \
      ui_state.sm['carState'].vEgo < MIN_SPEED
    go = stop.phase == StopPhase.GO or bool(e2e.greenLightAlert) or bool(e2e.leadDepartAlert) or light_went_green
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
    beat = 0.5 - 0.5 * math.cos(2 * math.pi * self._phase)
    # the hard brake warning has the edge to itself while it shows, so two glows never stack
    edge_free = hard_brake_flash.warning_level() < 0.01
    if u > 0.02 and edge_free:
      strength = lerp(FLASH_FLOOR, u) + (1 - lerp(FLASH_FLOOR, u)) * beat
      draw_edge_glow(rect, urgency_color(u), lerp(DEPTH_PX, u), lerp(PEAK_ALPHA, u) * strength * min(u * 4, 1.0), EDGE_LAYERS)
    elif time.monotonic() < self._go_until and edge_free:
      go_beat = 0.5 - 0.5 * math.cos(2 * math.pi * GO_HZ * time.monotonic())
      draw_edge_glow(rect, GREEN, GO_DEPTH, GO_ALPHA * (0.3 + 0.7 * go_beat), EDGE_LAYERS)
    papi = self._papi_alpha.x
    if papi > 0.01:
      if self.reds == 4:  # all red flashes in step with the glow
        papi *= 0.25 + 0.75 * beat
      draw_papi(rect, light_colors(self.reds), papi)


def draw_papi(rect: rl.Rectangle, colors: list[tuple], alpha: float) -> None:
  """A row of four lights centered on the bottom edge: a bright bar at the edge whose glow fades up into the picture."""
  total = rect.width * PAPI_WIDTH
  w = (total - PAPI_GAP * (len(colors) - 1)) / len(colors)
  x = rect.x + (rect.width - total) / 2
  core_y = rect.y + rect.height - PAPI_BOTTOM - PAPI_CORE_H
  for rgb in colors:
    halo_top = core_y + PAPI_CORE_H / 2 - PAPI_HALO_H
    rl.draw_rectangle_gradient_v(int(x), int(halo_top), int(w), int(PAPI_HALO_H), rgba(rgb, 0.0), rgba(rgb, 0.55 * alpha))
    # dark housing, so the lights stand out even inside a red or orange edge glow
    rl.draw_rectangle_rounded(rl.Rectangle(x - 10, core_y - 10, w + 20, PAPI_CORE_H + 20), 1.0, 12, rgba(BLACK, 0.6 * alpha))
    rl.draw_rectangle_rounded(rl.Rectangle(x - 6, core_y - 6, w + 12, PAPI_CORE_H + 12), 1.0, 12, rgba(rgb, 0.35 * alpha))
    rl.draw_rectangle_rounded(rl.Rectangle(x, core_y, w, PAPI_CORE_H), 1.0, 12, rgba(rgb, alpha))
    x += w + PAPI_GAP
