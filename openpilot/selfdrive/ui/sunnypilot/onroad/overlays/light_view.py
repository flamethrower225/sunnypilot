"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The traffic light state scened reads from the cameras, as the HUD shows it.
"""
from dataclasses import dataclass

import pyray as rl

from openpilot.cereal import custom
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import WHITE, ZONE_RGB, rgba
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_tracks import service_fresh
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr_noop

LightState = custom.SceneDetectionSP.LightState
MIN_CONFIDENCE = 0.5

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
  state: int            # LightState
  confidence: float
  distance: float       # m from the front bumper, NaN when unknown
  age: float            # s the state has held


def current_light(sm=None) -> LightView | None:
  """The light scened reports, when Camera Object Detection is on, it's arriving, and it's sure enough of a state."""
  sm = sm if sm is not None else ui_state.sm
  if not ui_state.scene_detection or not service_fresh(sm, 'sceneDetectionSP', ui_state.started_frame):
    return None
  tl = sm['sceneDetectionSP'].trafficLight
  if tl.state == LightState.unknown or tl.confidence < MIN_CONFIDENCE:
    return None
  return LightView(int(tl.state.raw if hasattr(tl.state, "raw") else tl.state), float(tl.confidence), float(tl.distance), float(tl.stateAge))


def draw_signal_head(cx: float, cy: float, scale: float, state: int, alpha: float) -> None:
  """A vertical signal head with the lit lamp glowing."""
  w, h = HEAD_W * scale, HEAD_H * scale
  body = rl.Rectangle(cx - w / 2, cy - h / 2, w, h)
  rl.draw_rectangle_rounded(body, 0.45, 8, rgba((18, 20, 21), alpha))
  rl.draw_rectangle_rounded_lines_ex(body, 0.45, 8, 2.0, rgba(WHITE, 0.35 * alpha))
  r = w * 0.32
  for i, lamp in enumerate(LAMP_ORDER):
    c = rl.Vector2(cx, cy - h / 2 + h * (i + 0.5) / 3)
    if lamp == state:
      rl.draw_circle_v(c, r * 1.9, rgba(LIGHT_RGB[lamp], 0.25 * alpha))
      rl.draw_circle_v(c, r, rgba(LIGHT_RGB[lamp], alpha))
    else:
      rl.draw_circle_v(c, r, rgba(DARK_LAMP, alpha))
