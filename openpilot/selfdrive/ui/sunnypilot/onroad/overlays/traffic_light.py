"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The traffic light state scened reads from the cameras, for the HUD: a chip in the stop column with a signal head showing the lit lamp
and the distance, which hands its place to the stop card while that is up (the card then names the light itself).
"""
import math
import time

import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, TEXT_MIN, WHITE, ZONE_RGB, draw_text, rgba, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays import stop_indicator
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.light_view import HEAD_W, LIGHT_RGB, LIGHT_WORD, LightView, SignView, current_light, \
  current_stop_sign, draw_signal_head, draw_stop_sign
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.lib.text_measure import measure_text_cached

CHIP_H = 96.0
CHIP_PAD = 16.0
LIGHT_RED = 1
STOPPED_SPEED = 0.5    # m/s, waiting at a red below this: the chip counts the wait instead of the distance
FLASH_HZ = 1.0         # the lamp blinks at about the rate real flashing signals do
FLASHING_WORD = tr_noop("FLASHING")


def distance_text(distance: float) -> tuple[str, str]:
  if not math.isfinite(distance):
    return "", ""
  return stop_indicator.format_distance(distance, ui_state.is_metric)


def wait_text(seconds: float) -> str:
  seconds = int(max(seconds, 0.0))
  return f"{seconds // 60}:{seconds % 60:02d}"


class TrafficLightChip:
  """The light ahead (its lamp, arrow and flashing as read), how long we've waited at it while red, or else a stop sign ahead."""

  def __init__(self):
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi = gui_app.font(FontWeight.SEMI_BOLD)
    self._alpha = FirstOrderFilter(0.0, 0.15, 1 / gui_app.target_fps)
    self._view: LightView | SignView | None = None
    self._wait_since: float | None = None

  def update(self) -> None:
    view: LightView | SignView | None = None
    if ui_state.started:
      view = current_light() or current_stop_sign()
    if view is not None:
      self._view = view
    stopped_at_red = isinstance(view, LightView) and view.state == LIGHT_RED and not view.flashing and \
      ui_state.sm['carState'].vEgo < STOPPED_SPEED
    if not stopped_at_red:
      self._wait_since = None
    elif self._wait_since is None:
      self._wait_since = time.monotonic()
    self._alpha.update(1.0 if view is not None else 0.0)

  @property
  def waited(self) -> float | None:
    """Seconds stopped at the red light ahead, or None when not waiting at one."""
    return None if self._wait_since is None else time.monotonic() - self._wait_since

  def render(self, rect: rl.Rectangle) -> None:
    alpha = self._alpha.x
    view = self._view
    # the stop card has the column while it's up, and names the light itself
    if alpha < 0.01 or view is None or stop_indicator.card_bottom() is not None:
      return
    x, w = stop_indicator.column(rect)
    y = rect.y + stop_indicator.TOP
    chip = rl.Rectangle(x, y, w, CHIP_H)
    icon_cx, cy = x + CHIP_PAD + HEAD_W / 2, y + CHIP_H / 2

    if isinstance(view, LightView):
      rgb = LIGHT_RGB.get(view.state, WHITE)
      lit = not view.flashing or (time.monotonic() * FLASH_HZ) % 1.0 < 0.5
      word = tr(FLASHING_WORD) if view.flashing else tr(LIGHT_WORD[view.state])
      waited = self.waited
      value, unit = (wait_text(waited), "") if waited is not None else distance_text(view.distance)
    else:
      rgb = ZONE_RGB[GapZone.CRITICAL]
      word = tr("STOP")
      value, unit = distance_text(view.distance)

    rl.draw_rectangle_rounded(chip, roundness(22, w, CHIP_H), 12, rgba(BLACK, 0.62 * alpha))
    rl.draw_rectangle_rounded_lines_ex(chip, roundness(22, w, CHIP_H), 12, 3, rgba(rgb, 0.7 * alpha))
    if isinstance(view, LightView):
      draw_signal_head(icon_cx, cy, 1.0, view.state, alpha, view.arrow, lit)
    else:
      draw_stop_sign(icon_cx, cy, HEAD_W * 0.95, self._font_bold, alpha)

    tx = x + CHIP_PAD + HEAD_W + 18
    word_size = 40 if isinstance(view, LightView) and view.flashing else 48   # FLASHING is a longer word
    word_sz = measure_text_cached(self._font_bold, word, word_size)
    draw_text(self._font_bold, word, word_size, tx, y + (CHIP_H - word_sz.y) / 2, rgba(rgb, alpha))
    if value:
      text = f"{value} {unit}".strip()
      sz = measure_text_cached(self._font_semi, text, TEXT_MIN)
      if tx + word_sz.x + 20 + sz.x <= x + w - CHIP_PAD:
        draw_text(self._font_semi, text, TEXT_MIN, x + w - CHIP_PAD - sz.x, y + (CHIP_H - sz.y) / 2, rgba(WHITE, 0.8 * alpha))
    stop_indicator.set_card_bottom(y + CHIP_H)

