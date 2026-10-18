"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

What scened reads from the cameras, as the HUD uses it: the traffic light, stop signs and people.
"""
import math
from dataclasses import dataclass

import pyray as rl

from openpilot.cereal import custom
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import WHITE, ZONE_RGB, draw_text_centered, draw_triangle, rgba
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.detection import StopPhase, get_detection_feed
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_tracks import RADAR_TO_CAMERA, service_fresh
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr_noop

LightState = custom.SceneDetectionSP.LightState
MIN_CONFIDENCE = 0.5
ARROW_NONE, ARROW_LEFT, ARROW_RIGHT, ARROW_STRAIGHT, ARROW_OTHER = 0, 1, 2, 3, 4   # SceneDetectionSP.Arrow
LIGHT_SETBACK = 10.0               # m, the stop line is about this far before a light when the model has no stop to go by
LIGHT_STOP_WINDOW = (-45.0, 5.0)   # m, a planned stop this far before (to just past) a light is the stop for it
SIGN_STOP_WINDOW = (-15.0, 10.0)   # a stop sign stands at its line
PEOPLE_MAX_AGE = 1.0               # s since scened last saw someone

LIGHT_RGB = {
  LightState.red: ZONE_RGB[GapZone.CRITICAL],
  LightState.yellow: ZONE_RGB[GapZone.CAUTION],
  LightState.green: ZONE_RGB[GapZone.GOOD],
}
LIGHT_WORD = {LightState.red: tr_noop("RED"), LightState.yellow: tr_noop("YELLOW"), LightState.green: tr_noop("GREEN")}
LIGHT_REASON = {LightState.red: tr_noop("Red light"), LightState.yellow: tr_noop("Yellow light"), LightState.green: tr_noop("Green light")}
LAMP_ORDER = (LightState.red, LightState.yellow, LightState.green)   # top to bottom
DARK_LAMP = (46, 50, 52)

HEAD_W, HEAD_H = 34.0, 84.0


@dataclass(frozen=True)
class LightView:
  state: int                  # LightState
  confidence: float
  distance: float             # m from the front bumper to the light, NaN when unknown
  age: float                  # s the state has held
  flashing: bool = False
  arrow: int = ARROW_NONE     # SceneDetectionSP.Arrow
  stop_distance: float = math.nan   # m from the front bumper to where we'd stop for it (see fused_stop_distance)

  @property
  def stop_at(self) -> float:
    """Where to stop for it: the fused stop distance, or the light's distance less the usual setback."""
    if math.isfinite(self.stop_distance):
      return self.stop_distance
    return max(self.distance - LIGHT_SETBACK, 0.0) if math.isfinite(self.distance) else math.nan


@dataclass(frozen=True)
class SignView:
  distance: float             # m from the front bumper to the stop sign
  confidence: float
  stop_distance: float        # m to where we'd stop for it


def fused_stop_distance(target: float, window: tuple[float, float], fallback_setback: float) -> float:
  """Where we'd actually stop for a light or sign. When the model plans a stop close to where the camera sees the light (or the
  sign), its stop point is the stop line, which is better than any guess: a light on the far side of a crossing can be 30 m past the
  line. Otherwise the light's distance less a typical setback. window is how far before (negative) and after the target the model's
  stop may be and still be this stop."""
  if not math.isfinite(target):
    return math.nan
  stop = get_detection_feed().stop
  if stop.phase == StopPhase.APPROACH and stop.stop_x is not None and not stop.car_ahead:
    model_stop = stop.stop_x - RADAR_TO_CAMERA
    if window[0] <= model_stop - target <= window[1]:
      return max(model_stop, 0.0)
  return max(target - fallback_setback, 0.0)


def _scene(sm):
  sm = sm if sm is not None else ui_state.sm
  if not ui_state.scene_detection or not service_fresh(sm, 'sceneDetectionSP', ui_state.started_frame):
    return None
  return sm['sceneDetectionSP']


def _raw(v) -> int:
  return int(v.raw if hasattr(v, "raw") else v)


def current_light(sm=None) -> LightView | None:
  """The light scened reports, when Camera Object Detection is on, it's arriving, and it's sure enough of a state."""
  scene = _scene(sm)
  if scene is None:
    return None
  tl = scene.trafficLight
  if tl.state == LightState.unknown or tl.confidence < MIN_CONFIDENCE:
    return None
  distance = float(tl.distance)
  return LightView(_raw(tl.state), float(tl.confidence), distance, float(tl.stateAge), bool(tl.flashing), _raw(tl.arrow),
                   fused_stop_distance(distance, LIGHT_STOP_WINDOW, LIGHT_SETBACK))


def current_stop_sign(sm=None) -> SignView | None:
  scene = _scene(sm)
  if scene is None or not scene.stopSign.present or not math.isfinite(scene.stopSign.distance):
    return None
  distance = float(scene.stopSign.distance)
  return SignView(distance, float(scene.stopSign.confidence), fused_stop_distance(distance, SIGN_STOP_WINDOW, 0.0))


def camera_people(sm=None) -> list:
  """Pedestrians and cyclists the cameras see, as sceneDetectionSP lists them (dRel, yRel left positive, kind)."""
  scene = _scene(sm)
  return [] if scene is None else [p for p in scene.people if p.lastSeen < PEOPLE_MAX_AGE]


def draw_signal_head(cx: float, cy: float, scale: float, state: int, alpha: float, arrow: int = ARROW_NONE, lit: bool = True) -> None:
  """A vertical signal head with the lit lamp glowing, an arrow on it when the state is an arrow's. lit=False draws the lamp dark,
  for the off half of a flash."""
  w, h = HEAD_W * scale, HEAD_H * scale
  body = rl.Rectangle(cx - w / 2, cy - h / 2, w, h)
  rl.draw_rectangle_rounded(body, 0.45, 8, rgba((18, 20, 21), alpha))
  rl.draw_rectangle_rounded_lines_ex(body, 0.45, 8, 2.0, rgba(WHITE, 0.35 * alpha))
  r = w * 0.32
  for i, lamp in enumerate(LAMP_ORDER):
    c = rl.Vector2(cx, cy - h / 2 + h * (i + 0.5) / 3)
    if lamp == state and lit:
      rl.draw_circle_v(c, r * 1.9, rgba(LIGHT_RGB[lamp], 0.25 * alpha))
      rl.draw_circle_v(c, r, rgba(LIGHT_RGB[lamp], alpha))
      if arrow in (ARROW_LEFT, ARROW_RIGHT, ARROW_STRAIGHT):
        _draw_arrow(c, r * 0.85, arrow, rgba((10, 12, 12), alpha))
    else:
      rl.draw_circle_v(c, r, rgba(DARK_LAMP, alpha))


def _draw_arrow(c: rl.Vector2, size: float, arrow: int, color: rl.Color) -> None:
  # drawn pointing right, then turned
  pts = [(1.0, 0.0), (0.05, -0.75), (0.05, 0.75)]
  shaft = [(-0.9, -0.22), (0.1, -0.22), (0.1, 0.22), (-0.9, 0.22)]
  rot = {ARROW_RIGHT: (1, 0), ARROW_LEFT: (-1, 0), ARROW_STRAIGHT: (0, -1)}[arrow]

  def place(p):
    x, y = p
    return (c.x + size * (x * rot[0] - y * rot[1]), c.y + size * (x * rot[1] + y * rot[0]))
  draw_triangle(place(pts[0]), place(pts[1]), place(pts[2]), color)
  a, b, cc, d = (place(p) for p in shaft)
  draw_triangle(a, b, cc, color)
  draw_triangle(a, cc, d, color)


def draw_stop_sign(cx: float, cy: float, radius: float, font, alpha: float) -> None:
  """A red octagon with a white rim and STOP on it."""
  for r, rgb in ((radius, WHITE), (radius * 0.85, ZONE_RGB[GapZone.CRITICAL])):
    octagon = [rl.Vector2(cx + r * math.cos(math.radians(22.5 + 45 * k)), cy + r * math.sin(math.radians(22.5 + 45 * k)))
               for k in range(8)]
    rl.draw_triangle_fan([rl.Vector2(cx, cy)] + octagon[::-1] + [octagon[-1]], 10, rgba(rgb, alpha))
  draw_text_centered(font, "STOP", max(int(radius * 0.58), 8), cx, cy, rgba(WHITE, alpha))
