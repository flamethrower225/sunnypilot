"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The traffic light state scened reads from the cameras, for the HUD: a chip in the stop column with a signal head showing the lit lamp
and the distance, which hands its place to the stop card while that is up (the card then names the light itself).
"""
import math

import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, TEXT_MIN, WHITE, draw_text, rgba, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays import stop_indicator
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.light_view import HEAD_W, LIGHT_RGB, LIGHT_WORD, LightView, current_light, \
  draw_signal_head
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

CHIP_H = 96.0
CHIP_PAD = 16.0


def distance_text(distance: float) -> tuple[str, str]:
  if not math.isfinite(distance):
    return "", ""
  return stop_indicator.format_distance(distance, ui_state.is_metric)


class TrafficLightChip:
  def __init__(self):
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi = gui_app.font(FontWeight.SEMI_BOLD)
    self._alpha = FirstOrderFilter(0.0, 0.15, 1 / gui_app.target_fps)
    self._view: LightView | None = None

  def update(self) -> None:
    view = current_light() if ui_state.started else None
    if view is not None:
      self._view = view
    self._alpha.update(1.0 if view is not None else 0.0)

  def render(self, rect: rl.Rectangle) -> None:
    alpha = self._alpha.x
    view = self._view
    # the stop card has the column while it's up, and names the light itself
    if alpha < 0.01 or view is None or stop_indicator.card_bottom() is not None:
      return
    x, w = stop_indicator.column(rect)
    y = rect.y + stop_indicator.TOP
    rgb = LIGHT_RGB.get(view.state, WHITE)
    chip = rl.Rectangle(x, y, w, CHIP_H)
    rl.draw_rectangle_rounded(chip, roundness(22, w, CHIP_H), 12, rgba(BLACK, 0.62 * alpha))
    rl.draw_rectangle_rounded_lines_ex(chip, roundness(22, w, CHIP_H), 12, 3, rgba(rgb, 0.7 * alpha))
    draw_signal_head(x + CHIP_PAD + HEAD_W / 2, y + CHIP_H / 2, 1.0, view.state, alpha)

    tx = x + CHIP_PAD + HEAD_W + 18
    word = tr(LIGHT_WORD[view.state])
    word_sz = measure_text_cached(self._font_bold, word, 48)
    draw_text(self._font_bold, word, 48, tx, y + (CHIP_H - word_sz.y) / 2, rgba(rgb, alpha))
    value, unit = distance_text(view.distance)
    if value:
      text = f"{value} {unit}"
      sz = measure_text_cached(self._font_semi, text, TEXT_MIN)
      if tx + word_sz.x + 20 + sz.x <= x + w - CHIP_PAD:
        draw_text(self._font_semi, text, TEXT_MIN, x + w - CHIP_PAD - sz.x, y + (CHIP_H - sz.y) / 2, rgba(WHITE, 0.8 * alpha))
    stop_indicator.set_card_bottom(y + CHIP_H)
