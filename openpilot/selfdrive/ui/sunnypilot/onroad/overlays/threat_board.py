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
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, INK, WHITE, ZONE_RGB, rgba, draw_text, \
  draw_delta_arrow, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone, ZoneHysteresis, format_distance, speed_value
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_tracks import RadarFeed, RadarTrack, select_threats, ttc_zone
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

PANEL_W = 400.0
EDGE_MARGIN = 40.0
PANEL_Y = 250.0               # from the HUD rect, below the wheel button
DEV_UI_SHIFT = 230.0          # the developer UI's right column
PAD = 14.0
HEADER_H = 52.0
SLOT_H = 60.0                 # row pitch
ROW_H = 52.0
MAX_ROWS = 4
BOTTOM_PAD = 6.0

NUM_SIZE, UNIT_SIZE = 28, 15
LANE_CHIP = (36.0, 32.0)


def board_rect(rect: rl.Rectangle, developer_ui, rows_height: float) -> rl.Rectangle:
  shift = DEV_UI_SHIFT if developer_ui in (DeveloperUiState.RIGHT, DeveloperUiState.BOTH) else 0.0
  return rl.Rectangle(rect.x + rect.width - EDGE_MARGIN - PANEL_W - shift, rect.y + PANEL_Y, PANEL_W, HEADER_H + rows_height + BOTTOM_PAD)


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
  """Up to four rows, most urgent first: a stripe in the severity color, the lane (L/C/R), distance, closing speed and time to
  collision. The lead is outlined and tagged. Stationary things (signs, guardrails) never get a row, and the board is hidden
  while nothing is closing."""

  def __init__(self):
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi_bold = gui_app.font(FontWeight.SEMI_BOLD)
    dt = self._dt = 1 / gui_app.target_fps
    self._feed = RadarFeed(dt)
    self._alpha = FirstOrderFilter(0.0, 0.15, dt)
    self._rows_height = FirstOrderFilter(0.0, 0.12, dt)
    self._rows: dict[int, _Row] = {}
    self._worst = FirstOrderFilter(np.array(ZONE_RGB[GapZone.GOOD], dtype=float), 0.15, dt, initialized=False)

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
      self._worst.update(np.array(ZONE_RGB[max(self._rows[t.track_id].zone for t in ranked)], dtype=float))
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
    self._draw_header(panel, alpha)

    now = rl.get_time()
    # lead tags are drawn last so they sit over the neighboring row
    rows = sorted(self._rows.values(), key=lambda r: r.slot.x)
    for row in rows:
      self._draw_row(panel, row, now, alpha)
    for row in rows:
      if row.lead and row.track is not None:
        self._draw_lead_tag(panel, row, alpha)

  def _draw_header(self, panel: rl.Rectangle, alpha: float) -> None:
    now = rl.get_time()
    rgb = self._worst.x
    pulse = 0.5 + 0.5 * math.sin(now * 2 * math.pi / 1.0)
    center = rl.Vector2(panel.x + PAD + 14, panel.y + 26)
    rl.draw_circle_v(center, 11 + 2 * pulse, rgba(rgb, (0.16 + 0.12 * pulse) * alpha))
    rl.draw_circle_v(center, 6.5, rgba(rgb, alpha))
    draw_text(self._font_bold, tr("THREATS"), 21, center.x + 22, panel.y + 13, rgba(WHITE, 0.92 * alpha), spacing=2.5)

    line_y = panel.y + HEADER_H - 4
    rl.draw_line_ex(rl.Vector2(panel.x + PAD, line_y), rl.Vector2(panel.x + panel.width - PAD, line_y), 1.5, rgba(WHITE, 0.14 * alpha))

  def _draw_row(self, panel: rl.Rectangle, row: _Row, now: float, panel_alpha: float) -> None:
    t = row.track
    alpha = row.alpha.x * panel_alpha
    if t is None or alpha < 0.02:
      return
    rgb = row.color.x
    x = panel.x + PAD
    y = panel.y + HEADER_H + row.slot.x * SLOT_H
    w = panel.width - 2 * PAD
    cy = y + ROW_H / 2

    box = rl.Rectangle(x, y, w, ROW_H)
    row_radius = roundness(14, w, ROW_H)
    glow = 0.5 + 0.5 * math.sin(now * 2 * math.pi / 0.8) if row.zone == GapZone.CRITICAL else 0.0
    rl.draw_rectangle_rounded(box, row_radius, 12, rgba(rgb, (0.1 + 0.12 * glow) * alpha))
    if row.lead:
      rl.draw_rectangle_rounded_lines_ex(box, row_radius, 12, 2.0, rgba(WHITE, 0.5 * alpha))
    rl.draw_rectangle_rounded(rl.Rectangle(x + 9, y + 9, 6, ROW_H - 18), 1.0, 8, rgba(rgb, alpha))

    # lane
    chip = rl.Rectangle(x + 27, cy - LANE_CHIP[1] / 2, LANE_CHIP[0], LANE_CHIP[1])
    rl.draw_rectangle_rounded(chip, 0.4, 8, rgba(WHITE, 0.16 * alpha))
    lane = str(t.lane)
    lane_sz = measure_text_cached(self._font_bold, lane, 23)
    draw_text(self._font_bold, lane, 23, chip.x + (chip.width - lane_sz.x) / 2, cy - lane_sz.y / 2, rgba(WHITE, alpha))

    # distance
    distance, unit = format_distance(t.d_rel, ui_state.is_metric)
    cx = x + 74
    cx += self._draw_value(cx, cy, distance, unit, rgba(WHITE, alpha), rgba(WHITE, 0.65 * alpha))

    # closing speed
    cx = x + 168
    speed = speed_value(t.closing_speed, ui_state.is_metric)
    if t.closing_speed >= 0.5:
      cx += draw_delta_arrow(-1, cx, cy, 14.0, rgba(WHITE, 0.85 * alpha)) + 6
    speed_unit = tr("km/h") if ui_state.is_metric else tr("mph")
    self._draw_value(cx, cy, str(speed), speed_unit, rgba(WHITE, alpha), rgba(WHITE, 0.65 * alpha))

    # time to collision, right aligned
    ttc_text = format_ttc(t.ttc) if math.isfinite(t.ttc) else "–"
    unit_sz = measure_text_cached(self._font_semi_bold, "s", UNIT_SIZE)
    num_sz = measure_text_cached(self._font_bold, ttc_text, NUM_SIZE)
    right = x + w - 14
    self._draw_value(right - num_sz.x - 3 - unit_sz.x, cy, ttc_text, "s", rgba(rgb, alpha), rgba(rgb, 0.8 * alpha), gap=3.0)

  def _draw_value(self, x: float, cy: float, value: str, unit: str, value_color: rl.Color, unit_color: rl.Color, gap: float = 4.0) -> float:
    """Number with a small unit on its baseline, vertically centered on cy. Returns the width used."""
    num_sz = measure_text_cached(self._font_bold, value, NUM_SIZE)
    unit_sz = measure_text_cached(self._font_semi_bold, unit, UNIT_SIZE)
    top = cy - num_sz.y / 2
    draw_text(self._font_bold, value, NUM_SIZE, x, top, value_color)
    draw_text(self._font_semi_bold, unit, UNIT_SIZE, x + num_sz.x + gap, top + num_sz.y * 0.8 - unit_sz.y * 0.8, unit_color)
    return num_sz.x + gap + unit_sz.x

  def _draw_lead_tag(self, panel: rl.Rectangle, row: _Row, panel_alpha: float) -> None:
    alpha = row.alpha.x * panel_alpha
    label = tr("LEAD") if row.lead == 1 else tr("2ND")
    size = 12
    sz = measure_text_cached(self._font_bold, label, size, 1.2)
    pill_w, pill_h = sz.x + 16, sz.y + 3
    x = panel.x + PAD + 22
    y = panel.y + HEADER_H + row.slot.x * SLOT_H - pill_h / 2 + 1
    pill = rl.Rectangle(x, y, pill_w, pill_h)
    rl.draw_rectangle_rounded(pill, 1.0, 10, rgba(WHITE, 0.95 * alpha))
    draw_text(self._font_bold, label, size, x + 8, y + 1.5, rgba(INK, alpha), spacing=1.2)
