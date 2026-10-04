"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np
import pyray as rl

from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, INK, WHITE, ZONE_RGB, rgba, gauge_scale, zone_label, \
  draw_text, draw_text_centered, draw_value_with_unit, draw_delta_arrow, draw_car_glyph, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import LeadGapReading, primary_value, speed_value, speed_delta
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

MAX_WIDTH = 1260
WIDTH_FRACTION = 0.62
UNITS = 60       # layout is specified in 1/60ths of the strip width
GAP_MAX = 4.0    # s at the right end of the track
DISTANCE_TICKS_METRIC = (0, 10, 20, 30)          # m, at a crawl
DISTANCE_TICKS_IMPERIAL = (0, 25, 50, 75, 100)   # ft, at a crawl

PAD_TOP, PAD_BOTTOM, PAD_X = 1.4, 1.2, 2.4
READ_WIDTH, COLUMN_GAP = 15.0, 2.2
TRACK_HEIGHT = 9.2
RAIL_LEFT, RAIL_RIGHT = 5.6, 1.2
CAR_WIDTH = 4.6


class FollowStrip:
  """Top-down schematic: your car on the left, the lead car sliding along a 0-4 s track colored by zone."""

  def __init__(self):
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi_bold = gui_app.font(FontWeight.SEMI_BOLD)

  @staticmethod
  def width(rect: rl.Rectangle) -> float:
    return min(MAX_WIDTH, rect.width * WIDTH_FRACTION)

  def render(self, rect: rl.Rectangle, bottom_offset: float, reading: LeadGapReading, zone_rgb: np.ndarray, alpha: float,
             is_metric: bool) -> None:
    w = self.width(rect)
    k = w / UNITS
    h = (PAD_TOP + TRACK_HEIGHT + PAD_BOTTOM) * k
    x0 = rect.x + (rect.width - w) / 2
    y0 = rect.y + rect.height - bottom_offset - h
    box = rl.Rectangle(x0, y0, w, h)

    rl.draw_rectangle_rounded(box, roundness(2.4 * k, w, h), 16, rgba(BLACK, 0.62 * alpha))
    rl.draw_rectangle_rounded_lines_ex(box, roundness(2.4 * k, w, h), 16, 0.3 * k, rgba(WHITE, 0.28 * alpha))

    content_y = y0 + PAD_TOP * k
    self._draw_readout(x0 + PAD_X * k, content_y, k, reading, zone_rgb, alpha, is_metric)

    track_x = x0 + (PAD_X + READ_WIDTH + COLUMN_GAP) * k
    track_w = w - (2 * PAD_X + READ_WIDTH + COLUMN_GAP) * k
    rail_x = track_x + RAIL_LEFT * k
    rail_w = track_w - (RAIL_LEFT + RAIL_RIGHT) * k
    self._draw_track(track_x, rail_x, rail_w, content_y, k, reading, zone_rgb, alpha, is_metric)

    if reading.show_ttc:
      self._draw_ttc_chip(x0 + w - PAD_X * k, y0, k, reading, zone_rgb, alpha)

  def _draw_readout(self, x: float, y: float, k: float, reading: LeadGapReading, zone_rgb, alpha: float, is_metric: bool) -> None:
    value, unit = primary_value(reading, is_metric)
    num_size, unit_size, label_size = 7 * k, 3 * k, 1.35 * k
    num_h = measure_text_cached(self._font_bold, value, int(num_size)).y
    block_h = num_h + label_size * 1.2
    top = y + (TRACK_HEIGHT * k - block_h) / 2
    color = rgba(zone_rgb, alpha)
    draw_value_with_unit(self._font_bold, value, num_size, self._font_bold, unit, unit_size, x, top, color, color, gap=0.3 * k)
    draw_text(self._font_bold, zone_label(reading).upper(), label_size, x, top + num_h, color, spacing=0.16 * k)

  def _draw_track(self, track_x: float, rail_x: float, rail_w: float, y: float, k: float, reading: LeadGapReading, zone_rgb,
                  alpha: float, is_metric: bool) -> None:
    scale = gauge_scale(reading, is_metric, GAP_MAX, DISTANCE_TICKS_METRIC, DISTANCE_TICKS_IMPERIAL)
    bar_y, bar_h = y + 4.2 * k, 1.1 * k
    for zone, lo, hi in scale.bands:
      seg_x = rail_x + rail_w * lo / scale.max_value
      rl.draw_rectangle_rec(rl.Rectangle(seg_x, bar_y, rail_w * (hi - lo) / scale.max_value, bar_h), rgba(ZONE_RGB[zone], 0.5 * alpha))

    tick_color = rgba(WHITE, 0.6 * alpha)
    tick_size = 1.15 * k
    for i, (value, label) in enumerate(scale.ticks):
      last = i == len(scale.ticks) - 1
      label = f"{label} {scale.unit}" if last else label
      sz = measure_text_cached(self._font_semi_bold, label, int(tick_size))
      tx = rail_x + rail_w * value / scale.max_value
      tx = tx if i == 0 else (tx - sz.x if last else tx - sz.x / 2)
      draw_text(self._font_semi_bold, label, tick_size, tx, y + 6.1 * k, tick_color)

    car_w = CAR_WIDTH * k
    car_y = y + 3.6 * k
    draw_car_glyph(track_x, car_y, car_w, rgba(WHITE, alpha))

    if scale.target is not None:
      notch_x = rail_x + rail_w * float(np.clip(scale.target / scale.max_value, 0.0, 1.0))
      notch = rl.Rectangle(notch_x - 0.175 * k, y + 3.7 * k, 0.35 * k, 2.1 * k)
      rl.draw_rectangle_rounded(notch, 0.6, 4, rgba(WHITE, alpha))
      draw_text_centered(self._font_bold, tr("TARGET"), k, notch_x, notch.y + notch.height + 2.4 * k, rgba(WHITE, alpha), spacing=0.08 * k)

    lead_cx = rail_x + rail_w * scale.value / scale.max_value
    draw_car_glyph(lead_cx - car_w / 2, car_y, car_w, rgba(zone_rgb, alpha))

    # lead speed and delta above the lead car
    speed = str(speed_value(reading.v_lead, is_metric))
    speed_unit = tr("km/h") if is_metric else tr("mph")
    delta = speed_delta(reading.v_rel, is_metric)
    direction = int(np.sign(delta))
    big, small = 2.3 * k, 1.3 * k
    speed_sz = measure_text_cached(self._font_bold, speed, int(big))
    unit_text = f"{speed_unit}"
    unit_sz = measure_text_cached(self._font_semi_bold, unit_text, int(small))
    delta_text = str(abs(delta)) if direction else ""
    delta_sz = measure_text_cached(self._font_semi_bold, delta_text, int(small))
    arrow = small * 0.75 if direction else 0.0
    total = speed_sz.x + 0.25 * k + unit_sz.x + (0.4 * k + arrow + 0.15 * k + delta_sz.x if direction else 0)
    lx = lead_cx - total / 2
    ly = car_y - 0.3 * k - speed_sz.y
    lx += draw_text(self._font_bold, speed, big, lx, ly, rgba(WHITE, alpha)).x + 0.25 * k
    small_y = ly + speed_sz.y * 0.8 - unit_sz.y * 0.8
    muted = rgba(WHITE, 0.7 * alpha)
    lx += draw_text(self._font_semi_bold, unit_text, small, lx, small_y, muted).x + 0.4 * k
    lx += draw_delta_arrow(direction, lx, small_y + unit_sz.y * 0.55, arrow, muted)
    if arrow:
      lx += 0.15 * k
    draw_text(self._font_semi_bold, delta_text, small, lx, small_y, muted)

  def _draw_ttc_chip(self, right: float, strip_top: float, k: float, reading: LeadGapReading, zone_rgb, alpha: float) -> None:
    text = tr("CLOSING • TTC {:.1f} S").format(reading.ttc)
    size = 1.4 * k
    sz = measure_text_cached(self._font_bold, text, int(size), 0.08 * k)
    pad_x, pad_y = 1.0 * k, 0.35 * k
    chip = rl.Rectangle(right - sz.x - 2 * pad_x, strip_top - sz.y / 2 - pad_y, sz.x + 2 * pad_x, sz.y + 2 * pad_y)
    rl.draw_rectangle_rounded(chip, 1.0, 12, rgba(zone_rgb, alpha))
    draw_text(self._font_bold, text, size, chip.x + pad_x, chip.y + pad_y, rgba(INK, alpha), spacing=0.08 * k)
