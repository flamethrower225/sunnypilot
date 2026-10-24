"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import time

import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, WHITE, ZONE_RGB, rgba, draw_text, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.selfdrive.harness.harness_check import alert_text
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

RED = ZONE_RGB[GapZone.CRITICAL]

SHOW_S = 30.0           # the card stays this long once the check says the harness needs a pin swap
READ_PERIOD_S = 1.0     # HarnessCheck is read from params at most this often
FADE_RC = 0.25

CARD_TOP = 500.0        # from the top of the HUD content rect, below the hard brake banner
PAD_X, PAD_Y = 34.0, 22.0
TITLE_SIZE = 44.0
HINT_SIZE = 36.0
LINE_GAP = 10.0


class HarnessAlert:
  """Red card when this drive's harness check says the camera's steering messages are on the wrong bus.

  Independent of Drive Notifications, so turning those off can't hide a wiring problem; only the Harness Check toggle does.
  Results are stamped with wall-clock time by harnessd, so a result written before this drive started never raises it.
  """

  def __init__(self):
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi = gui_app.font(FontWeight.SEMI_BOLD)
    self._alpha = FirstOrderFilter(0.0, FADE_RC, 1 / gui_app.target_fps)
    self._reset()

  def _reset(self) -> None:
    self._drive_start: float | None = None
    self._next_read = 0.0
    self._title: str | None = None
    self._shown_at: float | None = None

  def update(self) -> None:
    if not ui_state.harness_check or not ui_state.started:
      self._reset()
      self._alpha.x = 0.0
      return

    now = time.time()
    if self._drive_start is None:
      self._drive_start = now
    if self._shown_at is None and now >= self._next_read:
      self._next_read = now + READ_PERIOD_S
      self._title = alert_text(ui_state.params.get("HarnessCheck"), self._drive_start)
      if self._title is not None:
        self._shown_at = now
    visible = self._shown_at is not None and now - self._shown_at < SHOW_S
    self._alpha.update(1.0 if visible else 0.0)

  def render(self, rect: rl.Rectangle) -> None:
    alpha = self._alpha.x
    if not ui_state.harness_check or alpha < 0.01 or self._title is None:
      return

    title = self._title
    hint = tr("Settings > Developer > Harness Check")
    title_sz = measure_text_cached(self._font_bold, title, int(TITLE_SIZE))
    hint_sz = measure_text_cached(self._font_semi, hint, int(HINT_SIZE))
    w = max(title_sz.x, hint_sz.x) + 2 * PAD_X
    h = title_sz.y + LINE_GAP + hint_sz.y + 2 * PAD_Y
    x = rect.x + (rect.width - w) / 2
    y = rect.y + CARD_TOP - (1.0 - alpha) * 12
    card = rl.Rectangle(x, y, w, h)
    r = roundness(24.0, w, h)

    rl.draw_rectangle_rounded(card, r, 24, rgba(BLACK, 0.78 * alpha))
    rl.draw_rectangle_rounded(card, r, 24, rgba(RED, 0.3 * alpha))
    rl.draw_rectangle_rounded_lines_ex(card, r, 24, 3.0, rgba(RED, alpha))
    draw_text(self._font_bold, title, TITLE_SIZE, x + (w - title_sz.x) / 2, y + PAD_Y, rgba(WHITE, alpha))
    draw_text(self._font_semi, hint, HINT_SIZE, x + (w - hint_sz.x) / 2, y + PAD_Y + title_sz.y + LINE_GAP, rgba(WHITE, 0.85 * alpha))
