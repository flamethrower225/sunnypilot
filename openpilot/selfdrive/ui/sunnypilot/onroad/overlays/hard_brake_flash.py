"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, WHITE, ZONE_RGB, rgba, draw_text, draw_text_centered, \
  draw_triangle, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

RED = ZONE_RGB[GapZone.CRITICAL]

HOLD_TIME = 1.0         # s the warning stays after the model stops predicting a hard brake, so it can't flicker at 20 fps
PULSE_HZ = 2.0
FADE_IN_RC = 0.06
FADE_OUT_RC = 0.35

EDGE_RADIUS = 36.0      # px, corner radius of the glow along the content rect
EDGE_STEP = 4.0         # px between bands
EDGE_LAYERS = 20        # so the glow reaches about 80 px in from the edge
EDGE_ALPHA = 0.3       # opacity of the outermost band at full strength

PILL_TOP = 360.0        # from the top of the HUD content rect, under the disengage meter
PILL_HEIGHT = 52.0
PILL_TEXT_SIZE = 28.0
PILL_ICON = 30.0


class HoldLatch:
  """True while the input is on and for hold seconds after it turns off. Time is passed in so it can be tested."""

  def __init__(self, hold: float):
    self.hold = hold
    self._last_active: float | None = None

  def reset(self) -> None:
    self._last_active = None

  def update(self, active: bool, now: float) -> bool:
    if active:
      self._last_active = now
      return True
    return self._last_active is not None and now - self._last_active < self.hold


def pulse(t: float, hz: float = PULSE_HZ) -> float:
  """0 to 1, a sine at hz that starts at its trough."""
  return 0.5 - 0.5 * math.cos(2 * math.pi * hz * t)


def model_fresh() -> bool:
  sm = ui_state.sm
  return sm.recv_frame["modelV2"] >= ui_state.started_frame and sm.valid["modelV2"] and sm.alive["modelV2"]


class HardBrakeFlash:
  """Pulsing red glow around the screen edge and a banner when the model predicts a hard brake (its own FCW signal).

  Uses modelV2.meta.hardBrakePredicted as is. modeld already requires the 5 m/s^2 probability to clear its thresholds on five
  frames in a row and the 3 m/s^2 probability on two, so it is debounced and calibrated, and it is the same condition that raises
  the forward collision warning. A looser threshold on the raw brake probabilities would only add false alarms.
  """

  def __init__(self):
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._latch = HoldLatch(HOLD_TIME)
    self._alpha = FirstOrderFilter(0.0, FADE_OUT_RC, 1 / gui_app.target_fps)

  def update(self) -> None:
    if not ui_state.hard_brake_flash:
      self._latch.reset()
      self._alpha.x = 0.0
      return

    predicted = model_fresh() and bool(ui_state.sm["modelV2"].meta.hardBrakePredicted)
    held = self._latch.update(predicted, rl.get_time())
    self._alpha.update_alpha(FADE_IN_RC if held else FADE_OUT_RC)
    self._alpha.update(1.0 if held else 0.0)

  def render(self, rect: rl.Rectangle) -> None:
    if not ui_state.hard_brake_flash:
      return
    alpha = self._alpha.x
    if alpha < 0.01:
      return

    beat = pulse(rl.get_time())
    self._draw_edge_glow(rect, alpha * (0.5 + 0.5 * beat))
    self._draw_banner(rect, alpha, beat)

  @staticmethod
  def _draw_edge_glow(rect: rl.Rectangle, strength: float) -> None:
    """Inner glow as nested rounded outlines, each inset one step deeper and fainter than the last. Every band is two steps thick, so
    neighbours overlap and leave no seams, and the corner radius shrinks with the inset so the bands stay parallel."""
    thickness = 2 * EDGE_STEP
    for i in range(EDGE_LAYERS):
      inset = i * EDGE_STEP
      layer = rl.Rectangle(rect.x + inset, rect.y + inset, rect.width - 2 * inset, rect.height - 2 * inset)
      radius = EDGE_RADIUS - inset
      if radius < thickness:  # raylib's thick outline breaks when the thickness passes the corner radius, and this close it's square anyway
        radius = 0.0
      falloff = (1.0 - i / EDGE_LAYERS) ** 2
      rl.draw_rectangle_rounded_lines_ex(layer, roundness(radius, layer.width, layer.height) if radius else 0.0, 12, thickness,
                                         rgba(RED, EDGE_ALPHA * falloff * strength))

  def _draw_banner(self, rect: rl.Rectangle, alpha: float, beat: float) -> None:
    text = tr("HARD BRAKE PREDICTED")
    text_sz = measure_text_cached(self._font_bold, text, int(PILL_TEXT_SIZE), 1.6)
    h = PILL_HEIGHT
    w = PILL_ICON + text_sz.x + 66
    x = rect.x + (rect.width - w) / 2
    y = rect.y + PILL_TOP - (1.0 - alpha) * 12
    pill = rl.Rectangle(x, y, w, h)
    round_pill = roundness(h / 2, w, h)

    # soft halo that breathes with the glow
    rl.draw_rectangle_rounded(rl.Rectangle(x - 6, y - 6, w + 12, h + 12), roundness(h / 2 + 6, w + 12, h + 12), 24,
                              rgba(RED, (0.12 + 0.14 * beat) * alpha))
    rl.draw_rectangle_rounded(pill, round_pill, 24, rgba(BLACK, 0.72 * alpha))
    rl.draw_rectangle_rounded(pill, round_pill, 24, rgba(RED, (0.22 + 0.2 * beat) * alpha))
    rl.draw_rectangle_rounded_lines_ex(pill, round_pill, 24, 3.0, rgba(RED, alpha))

    icon_cx = x + 26 + PILL_ICON / 2
    cy = y + h / 2
    self._draw_warning_triangle(icon_cx, cy, PILL_ICON, alpha)
    draw_text(self._font_bold, text, PILL_TEXT_SIZE, x + 26 + PILL_ICON + 14, cy - text_sz.y / 2, rgba(WHITE, alpha), spacing=1.6)

  def _draw_warning_triangle(self, cx: float, cy: float, size: float, alpha: float) -> None:
    half = size / 2
    top, bottom = cy - half * 0.92, cy + half * 0.78
    draw_triangle((cx, top), (cx + half, bottom), (cx - half, bottom), rgba(RED, alpha))
    draw_text_centered(self._font_bold, "!", size * 0.78, cx, cy + half * 0.2, rgba(BLACK, 0.85 * alpha))
