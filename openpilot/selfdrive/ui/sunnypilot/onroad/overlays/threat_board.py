"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Threat board: the radar tracks closing on us, most urgent first, in the right column of the HUD.
"""
import math
import time

import numpy as np
import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.developer_ui import DeveloperUiState
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, TEXT_MIN, WHITE, ZONE_RGB, rgba, draw_text, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone, ZoneHysteresis, format_distance
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_tracks import Lane, RadarFeed, RadarTrack, select_threats, ttc_zone
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached

PANEL_W = 420.0
EDGE_MARGIN = 40.0
PANEL_Y = 250.0               # from the HUD rect, below the wheel button
DEV_UI_SHIFT = 230.0          # the developer UI's right column
PAD = 14.0
TOP_PAD = 14.0
SLOT_H = 86.0                 # row pitch
ROW_H = 76.0
MAX_ROWS = 3                  # more than three rows is reading, not glancing
BOTTOM_PAD = 14.0

TTC_SIZE, DIST_SIZE, UNIT_SIZE = 56, 44, TEXT_MIN
LANE_ICON_BAR, LANE_ICON_GAP, LANE_ICON_H = 14.0, 6.0, 48.0
LEAD_OUTLINE = 5.0
LANE_INDEX = {Lane.LEFT: 0, Lane.CENTER: 1, Lane.RIGHT: 2}


def board_rect(rect: rl.Rectangle, developer_ui, rows_height: float) -> rl.Rectangle:
  shift = DEV_UI_SHIFT if developer_ui in (DeveloperUiState.RIGHT, DeveloperUiState.BOTH) else 0.0
  return rl.Rectangle(rect.x + rect.width - EDGE_MARGIN - PANEL_W - shift, rect.y + PANEL_Y, PANEL_W, TOP_PAD + rows_height + BOTTOM_PAD)


def format_ttc(ttc: float) -> str:
  return f"{ttc:.1f}" if ttc < 10 else f"{ttc:.0f}"


class _Row:
  """One track's row, which keeps sliding and fading after the track has stopped being a threat."""

  def __init__(self, dt: float, slot: int):
    self.slot = FirstOrderFilter(float(slot), 0.1, dt)
    self.alpha = FirstOrderFilter(0.0, 0.12, dt)
    self.color = FirstOrderFilter(np.array(ZONE_RGB[GapZone.GOOD], dtype=float), 0.15, dt, initialized=False)
    self.hysteresis = ZoneHysteresis(dt)
    self.zone = GapZone.GOOD
    self.track: RadarTrack | None = None
    self.lead = 0   # 1 for leadOne, 2 for leadTwo


class ThreatBoard:
  """Up to three rows, most urgent first, each tinted in its severity color: a three-lane icon with the track's lane lit, the
  distance and a large time to collision. The lead's row has a white outline, the second lead's a dim one. Stationary things
  (signs, guardrails) never get a row, and the board is hidden while nothing is closing."""

  def __init__(self):
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi_bold = gui_app.font(FontWeight.SEMI_BOLD)
    dt = self._dt = 1 / gui_app.target_fps
    self._feed = RadarFeed(dt)
    self._alpha = FirstOrderFilter(0.0, 0.15, dt)
    self._rows_height = FirstOrderFilter(0.0, 0.12, dt)
    self._rows: dict[int, _Row] = {}

  def update(self) -> None:
    if not ui_state.threat_board:
      self._alpha.x = 0.0
      self._rows.clear()
      return
    self._feed.update(ui_state.sm, ui_state.started_frame, time.monotonic())
    ranked = select_threats(self._feed.tracks, self._feed.lead_ids, MAX_ROWS) if self._feed.available else []

    seen = set()
    for slot, t in enumerate(ranked):
      row = self._rows.get(t.track_id)
      if row is None:
        row = self._rows[t.track_id] = _Row(self._dt, slot)
      row.track = t
      row.lead = 1 if t.track_id == self._feed.lead_ids[0] else (2 if t.track_id == self._feed.lead_ids[1] else 0)
      row.zone = row.hysteresis.update(ttc_zone(t.ttc))
      row.color.update(np.array(ZONE_RGB[row.zone], dtype=float))
      row.slot.update(slot)
      row.alpha.update(1.0)
      seen.add(t.track_id)

    for track_id in list(self._rows):
      if track_id not in seen:
        row = self._rows[track_id]
        if row.alpha.update(0.0) < 0.02:
          del self._rows[track_id]

    if ranked:
      self._rows_height.update(len(ranked) * SLOT_H - (SLOT_H - ROW_H))
    self._alpha.update(1.0 if ranked else 0.0)

  def render(self, rect: rl.Rectangle) -> None:
    if not ui_state.threat_board:
      return
    alpha = self._alpha.x
    if alpha < 0.01:
      return

    panel = board_rect(rect, ui_state.developer_ui, self._rows_height.x)
    radius = roundness(22, panel.width, panel.height)
    rl.draw_rectangle_rounded(panel, radius, 16, rgba(BLACK, 0.62 * alpha))
    rl.draw_rectangle_rounded_lines_ex(panel, radius, 16, 3, rgba(WHITE, 0.28 * alpha))

    now = rl.get_time()
    for row in sorted(self._rows.values(), key=lambda r: r.slot.x):
      self._draw_row(panel, row, now, alpha)

  def _draw_row(self, panel: rl.Rectangle, row: _Row, now: float, panel_alpha: float) -> None:
    t = row.track
    alpha = row.alpha.x * panel_alpha
    if t is None or alpha < 0.02:
      return
    rgb = row.color.x
    x = panel.x + PAD
    y = panel.y + TOP_PAD + row.slot.x * SLOT_H
    w = panel.width - 2 * PAD
    cy = y + ROW_H / 2

    box = rl.Rectangle(x, y, w, ROW_H)
    row_radius = roundness(16, w, ROW_H)
    glow = 0.5 + 0.5 * math.sin(now * 2 * math.pi / 0.8) if row.zone == GapZone.CRITICAL else 0.0
    rl.draw_rectangle_rounded(box, row_radius, 12, rgba(rgb, (0.2 + 0.14 * glow) * alpha))
    if row.lead:
      rl.draw_rectangle_rounded_lines_ex(box, row_radius, 12, LEAD_OUTLINE, rgba(WHITE, (0.95 if row.lead == 1 else 0.4) * alpha))

    cx = x + 18
    cx += self._draw_lane_icon(cx, cy, t.lane, rgb, alpha) + 18

    distance, unit = format_distance(t.d_rel, ui_state.is_metric)
    self._draw_value(cx, cy, distance, DIST_SIZE, unit, rgba(WHITE, alpha), rgba(WHITE, 0.7 * alpha))

    # time to collision, right aligned and the largest thing on the row
    ttc_text = format_ttc(t.ttc) if math.isfinite(t.ttc) else "–"
    ttc_w = self._value_width(ttc_text, TTC_SIZE, "s", gap=3.0)
    self._draw_value(x + w - 16 - ttc_w, cy, ttc_text, TTC_SIZE, "s", rgba(rgb, alpha), rgba(rgb, 0.85 * alpha), gap=3.0)

  @staticmethod
  def _draw_lane_icon(x: float, cy: float, lane: Lane, rgb, alpha: float) -> float:
    """Three lane bars with the track's lane lit, readable without reading a letter. Returns the width used."""
    lit = LANE_INDEX[lane]
    top = cy - LANE_ICON_H / 2
    for i in range(3):
      bar = rl.Rectangle(x + i * (LANE_ICON_BAR + LANE_ICON_GAP), top, LANE_ICON_BAR, LANE_ICON_H)
      rl.draw_rectangle_rounded(bar, 0.5, 6, rgba(rgb, alpha) if i == lit else rgba(WHITE, 0.2 * alpha))
    return 3 * LANE_ICON_BAR + 2 * LANE_ICON_GAP

  def _value_width(self, value: str, size: float, unit: str, gap: float = 6.0) -> float:
    return measure_text_cached(self._font_bold, value, int(size)).x + gap + measure_text_cached(self._font_semi_bold, unit, UNIT_SIZE).x

  def _draw_value(self, x: float, cy: float, value: str, size: float, unit: str, value_color: rl.Color, unit_color: rl.Color,
                  gap: float = 6.0) -> float:
    """Number with a unit on its baseline, vertically centered on cy. Returns the width used."""
    num_sz = measure_text_cached(self._font_bold, value, int(size))
    unit_sz = measure_text_cached(self._font_semi_bold, unit, UNIT_SIZE)
    top = cy - num_sz.y / 2
    draw_text(self._font_bold, value, size, x, top, value_color)
    draw_text(self._font_semi_bold, unit, UNIT_SIZE, x + num_sz.x + gap, top + num_sz.y * 0.8 - unit_sz.y * 0.8, unit_color)
    return num_sz.x + gap + unit_sz.x
