"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

import numpy as np
import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, INK, WHITE, rgba, zone_label, delta_label, \
  draw_text, draw_value_with_unit, measure_value_with_unit, draw_delta_arrow, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone, LeadGapReading, primary_value, speed_value
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

LEAD_HALF_WIDTH = 0.9  # m
LEAD_HEIGHT = 1.5      # m
BOX_PADDING = 0.12     # fraction of the projected car size added on each side
MIN_BOX_W, MIN_BOX_H = 110.0, 80.0
MAX_BOX_W_FRACTION = 0.6

K = 21.0  # px per layout unit, matches the other badge styles at full width
STEM = 1.1 * K
MARGIN = 20.0


class LeadLock:
  """Corner brackets locked onto the lead car in the zone color, with a large tag hanging under its bumper."""

  def __init__(self):
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi_bold = gui_app.font(FontWeight.SEMI_BOLD)
    self._box_filter = FirstOrderFilter(np.zeros(4), 0.1, 1 / gui_app.target_fps, initialized=False)
    self._has_box = False

  def set_box(self, box: np.ndarray | None) -> None:
    """Screen box (left, top, right, bottom) around the lead, or None when there is no lead to lock onto."""
    if box is None:
      self._has_box = False
      return
    if not self._has_box:
      self._box_filter.initialized = False
    self._has_box = True
    self._box_filter.update(box)

  @staticmethod
  def pad_box(left: float, top: float, right: float, bottom: float, rect: rl.Rectangle) -> np.ndarray:
    w, h = right - left, bottom - top
    pad_w, pad_h = w * BOX_PADDING, h * BOX_PADDING
    left, right, top, bottom = left - pad_w, right + pad_w, top - pad_h, bottom + pad_h * 0.5
    cx = (left + right) / 2
    w = float(np.clip(right - left, MIN_BOX_W, rect.width * MAX_BOX_W_FRACTION))
    h = max(bottom - top, MIN_BOX_H)
    return np.array([cx - w / 2, bottom - h, cx + w / 2, bottom])

  def render(self, rect: rl.Rectangle, reading: LeadGapReading, zone_rgb: np.ndarray, alpha: float, is_metric: bool) -> None:
    if not self._box_filter.initialized:
      return

    left, top, right, bottom = self._box_filter.x
    if reading.zone == GapZone.CRITICAL:
      scale = 1.0 + 0.05 * (0.5 + 0.5 * math.sin(rl.get_time() * 2 * math.pi / 1.2))
      cx, cy = (left + right) / 2, (top + bottom) / 2
      left, right = cx - (cx - left) * scale, cx + (right - cx) * scale
      top, bottom = cy - (cy - top) * scale, cy + (bottom - cy) * scale

    self._draw_brackets(left, top, right, bottom, zone_rgb, alpha)
    self._draw_tag(rect, (left + right) / 2, bottom, reading, zone_rgb, alpha, is_metric)

  @staticmethod
  def _draw_brackets(left: float, top: float, right: float, bottom: float, zone_rgb, alpha: float) -> None:
    w, h = right - left, bottom - top
    arm_x, arm_y = w * 0.28, h * 0.32
    corners = (
      ((left, top), (1, 1)),
      ((right, top), (-1, 1)),
      ((left, bottom), (1, -1)),
      ((right, bottom), (-1, -1)),
    )
    for thickness, color in ((22.0, rgba(zone_rgb, 0.25 * alpha)), (9.0, rgba(zone_rgb, alpha))):
      for (x, y), (dx, dy) in corners:
        corner = rl.Vector2(x, y)
        rl.draw_line_ex(corner, rl.Vector2(x + dx * arm_x, y), thickness, color)
        rl.draw_line_ex(corner, rl.Vector2(x, y + dy * arm_y), thickness, color)
        rl.draw_circle_v(corner, thickness / 2, color)

  def _draw_tag(self, rect: rl.Rectangle, cx: float, box_bottom: float, reading: LeadGapReading, zone_rgb, alpha: float,
                is_metric: bool) -> None:
    value, unit = primary_value(reading, is_metric)
    gap_num, gap_unit, gap_label = 6.2 * K, 2.6 * K, 1.1 * K
    if reading.show_ttc:
      label = tr("CLOSING")
    else:
      label = zone_label(reading).upper()
    gap_sz = measure_value_with_unit(self._font_bold, value, gap_num, self._font_bold, unit, gap_unit, gap=0.2 * K)
    label_sz = measure_text_cached(self._font_bold, label, int(gap_label), 0.14 * K)
    gap_w = max(gap_sz.x, label_sz.x) + 3.2 * K

    speed = str(speed_value(reading.v_lead, is_metric))
    speed_unit = tr("km/h") if is_metric else tr("mph")
    spd_num, spd_unit, spd_detail = 3.6 * K, 1.4 * K, 1.3 * K
    spd_sz = measure_value_with_unit(self._font_bold, speed, spd_num, self._font_semi_bold, speed_unit, spd_unit, gap=0.3 * K)
    if reading.show_ttc:
      direction, detail = 0, tr("TTC {:.1f} s").format(reading.ttc)
    else:
      direction, detail = delta_label(reading, is_metric)
    detail_sz = measure_text_cached(self._font_bold, detail, int(spd_detail))
    arrow = spd_detail * 0.7 if direction else 0.0
    detail_w = detail_sz.x + (arrow + 0.25 * K if arrow else 0.0)
    spd_w = max(spd_sz.x, detail_w) + 3.2 * K

    tag_w = gap_w + spd_w
    tag_h = gap_sz.y * 0.92 + label_sz.y + 1.1 * K

    x = float(np.clip(cx - tag_w / 2, rect.x + MARGIN, rect.x + rect.width - MARGIN - tag_w))
    y = min(box_bottom + STEM, rect.y + rect.height - MARGIN - tag_h)

    if y > box_bottom:
      rl.draw_line_ex(rl.Vector2(cx, box_bottom), rl.Vector2(cx, y), 0.3 * K, rgba(zone_rgb, alpha))

    radius = 1.6 * K
    tag = rl.Rectangle(x, y, tag_w, tag_h)
    rl.draw_rectangle_rounded(rl.Rectangle(x + 0.25 * K, y + 0.4 * K, tag_w, tag_h), roundness(radius, tag_w, tag_h), 12,
                              rgba(BLACK, 0.3 * alpha))
    rl.draw_rectangle_rounded(tag, roundness(radius, tag_w, tag_h), 12, rgba(BLACK, 0.75 * alpha))
    gap_rect = rl.Rectangle(x, y, gap_w, tag_h)
    rl.draw_rectangle_rounded(gap_rect, roundness(radius, gap_w, tag_h), 12, rgba(zone_rgb, alpha))
    rl.draw_rectangle_rec(rl.Rectangle(x + gap_w - radius, y, radius, tag_h), rgba(zone_rgb, alpha))

    ink = rgba(INK, alpha)
    content_h = gap_sz.y * 0.92 + label_sz.y
    gy = y + (tag_h - content_h) / 2
    draw_value_with_unit(self._font_bold, value, gap_num, self._font_bold, unit, gap_unit, x + (gap_w - gap_sz.x) / 2, gy, ink, ink,
                         gap=0.2 * K)
    draw_text(self._font_bold, label, gap_label, x + (gap_w - label_sz.x) / 2, gy + gap_sz.y * 0.92, rgba(INK, 0.75 * alpha),
              spacing=0.14 * K)

    sx = x + gap_w + 1.6 * K
    spd_content_h = spd_sz.y + 0.2 * K + detail_sz.y
    sy = y + (tag_h - spd_content_h) / 2
    draw_value_with_unit(self._font_bold, speed, spd_num, self._font_semi_bold, speed_unit, spd_unit, sx, sy, rgba(WHITE, alpha),
                         rgba(WHITE, 0.7 * alpha), gap=0.3 * K)
    detail_y = sy + spd_sz.y + 0.2 * K
    detail_color = rgba(zone_rgb, alpha) if reading.show_ttc else rgba(WHITE, 0.8 * alpha)
    sx += draw_delta_arrow(direction, sx, detail_y + detail_sz.y / 2, arrow, detail_color)
    if arrow:
      sx += 0.25 * K
    draw_text(self._font_bold, detail, spd_detail, sx, detail_y, detail_color)
