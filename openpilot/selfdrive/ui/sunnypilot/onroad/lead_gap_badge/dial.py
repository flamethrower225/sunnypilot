"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

import numpy as np
import pyray as rl

from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, WHITE, ZONE_RGB, GaugeScale, rgba, gauge_scale, \
  zone_label, delta_label, draw_text_centered, draw_arc, draw_delta_arrow
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import LeadGapReading, primary_value, speed_value
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

SCALE = 2.2
DISC_RADIUS = 100 * SCALE
INNER_DISC_RADIUS = 62 * SCALE
ARC_RADIUS = 80 * SCALE
ARC_WIDTH = 13 * SCALE
BAND_RADIUS = ARC_RADIUS + 11 * SCALE
BAND_WIDTH = 3 * SCALE
TICK_RADIUS = ARC_RADIUS - 22 * SCALE

START_ANGLE = 150.0  # 7 o'clock, sweeping clockwise over the top to 5 o'clock
SWEEP = 240.0
GAP_MAX = 3.0  # s at full arc
DISTANCE_TICKS_METRIC = (0, 6, 12, 18)      # m, at a crawl
DISTANCE_TICKS_IMPERIAL = (0, 20, 40, 60)   # ft, at a crawl


def scale_angle(value: float, scale: GaugeScale) -> float:
  return START_ANGLE + SWEEP * float(np.clip(value / scale.max_value, 0.0, 1.0))


class GapDial:
  """Round gauge: the arc fills 0-3 s in the zone color, a white tick marks openpilot's target gap."""

  def __init__(self):
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi_bold = gui_app.font(FontWeight.SEMI_BOLD)

  @staticmethod
  def size() -> float:
    return 2 * DISC_RADIUS

  def render(self, rect: rl.Rectangle, bottom_offset: float, reading: LeadGapReading, zone_rgb: np.ndarray, alpha: float,
             is_metric: bool) -> None:
    center = rl.Vector2(rect.x + rect.width / 2, rect.y + rect.height - bottom_offset - DISC_RADIUS)

    rl.draw_circle_v(center, DISC_RADIUS, rgba(BLACK, 0.5 * alpha))
    rl.draw_circle_v(center, INNER_DISC_RADIUS, rgba(BLACK, 0.44 * alpha))

    scale = gauge_scale(reading, is_metric, GAP_MAX, DISTANCE_TICKS_METRIC, DISTANCE_TICKS_IMPERIAL)

    # zone scale just outside the arc
    for zone, lo, hi in scale.bands:
      draw_arc(center, BAND_RADIUS, BAND_WIDTH, scale_angle(lo, scale), scale_angle(hi, scale), rgba(ZONE_RGB[zone], 0.55 * alpha))

    draw_arc(center, ARC_RADIUS, ARC_WIDTH, START_ANGLE, START_ANGLE + SWEEP, rgba(WHITE, 0.12 * alpha), round_caps=True)

    if scale.value > 0.01 * scale.max_value:
      draw_arc(center, ARC_RADIUS, ARC_WIDTH, START_ANGLE, scale_angle(scale.value, scale), rgba(zone_rgb, alpha), round_caps=True)

    if scale.target is not None:
      angle = math.radians(scale_angle(scale.target, scale))
      direction = rl.Vector2(math.cos(angle), math.sin(angle))
      inner, outer = ARC_RADIUS - 9 * SCALE, ARC_RADIUS + 9 * SCALE
      rl.draw_line_ex(rl.Vector2(center.x + direction.x * inner, center.y + direction.y * inner),
                      rl.Vector2(center.x + direction.x * outer, center.y + direction.y * outer), 3.5 * SCALE, rgba(WHITE, alpha))

    tick_color = rgba(WHITE, 0.55 * alpha)
    for i, (value, label) in enumerate(scale.ticks):
      angle = math.radians(scale_angle(value, scale))
      label = f"{label}+" if i == len(scale.ticks) - 1 else label
      draw_text_centered(self._font_bold, label, 10 * SCALE, center.x + TICK_RADIUS * math.cos(angle),
                         center.y + TICK_RADIUS * math.sin(angle), tick_color)

    if reading.show_ttc:
      header = tr("TTC {:.1f} S").format(reading.ttc)
    else:
      header = zone_label(reading).upper()
    draw_text_centered(self._font_bold, header, 10.5 * SCALE, center.x, center.y - 29 * SCALE, rgba(zone_rgb, alpha), spacing=1.5 * SCALE)

    value, unit = primary_value(reading, is_metric)
    draw_text_centered(self._font_bold, value, 48 * SCALE, center.x, center.y + 1 * SCALE, rgba(WHITE, alpha))
    unit_text = tr("SECONDS") if unit == "s" else (tr("FEET") if unit == "ft" else tr("METERS"))
    draw_text_centered(self._font_bold, unit_text, 10 * SCALE, center.x, center.y + 31 * SCALE, rgba(WHITE, 0.65 * alpha), spacing=SCALE)

    speed_unit = tr("km/h") if is_metric else tr("mph")
    draw_text_centered(self._font_bold, f"{speed_value(reading.v_lead, is_metric)} {speed_unit}", 17 * SCALE,
                       center.x, center.y + 52 * SCALE, rgba(WHITE, alpha))

    direction, delta_text = delta_label(reading, is_metric)
    delta_size = 10 * SCALE
    delta_color = rgba(WHITE, 0.7 * alpha)
    text_sz = measure_text_cached(self._font_semi_bold, delta_text, int(delta_size))
    arrow = delta_size * 0.8 if direction else 0.0
    x = center.x - (text_sz.x + arrow + (delta_size * 0.35 if arrow else 0)) / 2
    y = center.y + 70.5 * SCALE
    x += draw_delta_arrow(direction, x, y, arrow, delta_color)
    if arrow:
      x += delta_size * 0.35
    rl.draw_text_ex(self._font_semi_bold, delta_text, rl.Vector2(x, y - text_sz.y / 2), delta_size, 0, delta_color)
