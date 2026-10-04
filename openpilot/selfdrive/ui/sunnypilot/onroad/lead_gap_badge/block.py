"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np
import pyray as rl

from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, INK, rgba, zone_label, draw_text, \
  draw_value_with_unit, measure_value_with_unit, draw_delta_arrow, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone, LeadGapReading, primary_value, speed_value, \
  speed_delta
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

MAX_WIDTH = 567
WIDTH_FRACTION = 0.27
UNITS = 27  # layout is specified in 1/27ths of the tile width

BARS = {GapZone.NONE: 0, GapZone.CRITICAL: 1, GapZone.CLOSE: 2, GapZone.CAUTION: 3, GapZone.GOOD: 4, GapZone.OPEN: 4}


class SignalBlock:
  """A solid tile filled with the zone color, numerals as big as the speedometer, and distance bars like an OEM cruise display."""

  def __init__(self):
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi_bold = gui_app.font(FontWeight.SEMI_BOLD)

  @staticmethod
  def width(rect: rl.Rectangle) -> float:
    return min(MAX_WIDTH, rect.width * WIDTH_FRACTION)

  @staticmethod
  def _layout(k: float) -> dict[str, float]:
    return {"pad_top": 1.4 * k, "pad_x": 2.0 * k, "pad_bottom": 1.6 * k, "row": 1.45 * k, "num": 11.5 * k, "unit": 3.8 * k,
            "bars": 2.4 * k, "lead": 2.2 * k, "lead_unit": 1.2 * k, "gap": 0.4 * k}

  def height(self, rect: rl.Rectangle) -> float:
    k = self.width(rect) / UNITS
    lay = self._layout(k)
    num_h = measure_text_cached(self._font_bold, "0", int(lay["num"])).y * 0.86
    return lay["pad_top"] + lay["row"] * 1.2 + lay["gap"] + num_h + lay["gap"] + lay["bars"] + lay["pad_bottom"]

  def render(self, rect: rl.Rectangle, bottom_offset: float, reading: LeadGapReading, zone_rgb: np.ndarray, alpha: float,
             is_metric: bool) -> None:
    w = self.width(rect)
    k = w / UNITS
    lay = self._layout(k)
    h = self.height(rect)
    x0 = rect.x + (rect.width - w) / 2
    y0 = rect.y + rect.height - bottom_offset - h
    tile = rl.Rectangle(x0, y0, w, h)

    ring = 0.35 * k
    rl.draw_rectangle_rounded(rl.Rectangle(x0 - ring, y0 - ring, w + 2 * ring, h + 2 * ring), roundness(2.6 * k + ring, w, h), 16,
                              rgba(BLACK, 0.35 * alpha))
    rl.draw_rectangle_rounded(tile, roundness(2.6 * k, w, h), 16, rgba(zone_rgb, 0.92 * alpha))

    ink = rgba(INK, alpha)
    x, right = x0 + lay["pad_x"], x0 + w - lay["pad_x"]
    y = y0 + lay["pad_top"]
    draw_text(self._font_bold, tr("GAP"), lay["row"], x, y, ink, spacing=0.12 * k)
    status = tr("TTC {:.1f}s").format(reading.ttc) if reading.show_ttc else zone_label(reading).upper()
    status_sz = measure_text_cached(self._font_bold, status, int(lay["row"]), 0.12 * k)
    draw_text(self._font_bold, status, lay["row"], right - status_sz.x, y, ink, spacing=0.12 * k)
    y += lay["row"] * 1.2 + lay["gap"]

    value, unit = primary_value(reading, is_metric)
    num_sz = measure_value_with_unit(self._font_bold, value, lay["num"], self._font_bold, unit, lay["unit"], gap=0.4 * k)
    # the numeral line box has generous headroom, pull it up so the digits sit centered in the tile
    draw_value_with_unit(self._font_bold, value, lay["num"], self._font_bold, unit, lay["unit"], x0 + (w - num_sz.x) / 2,
                         y - num_sz.y * 0.08, ink, ink, gap=0.4 * k)
    y += num_sz.y * 0.86 + lay["gap"]

    bar_w, bar_gap = 1.1 * k, 0.5 * k
    for i in range(4):
      bh = lay["bars"] * (0.35 + 0.65 * i / 3)
      bar = rl.Rectangle(x + i * (bar_w + bar_gap), y + lay["bars"] - bh, bar_w, bh)
      rl.draw_rectangle_rounded(bar, 0.4, 4, rgba(INK, (1.0 if i < BARS[reading.zone] else 0.2) * alpha))

    speed = str(speed_value(reading.v_lead, is_metric))
    speed_unit = tr("km/h") if is_metric else tr("mph")
    delta = speed_delta(reading.v_rel, is_metric)
    speed_sz = measure_value_with_unit(self._font_bold, speed, lay["lead"], self._font_semi_bold, speed_unit, lay["lead_unit"], gap=0.2 * k)
    delta_text = str(abs(delta)) if delta else ""
    delta_sz = measure_text_cached(self._font_bold, delta_text, int(lay["lead"]))
    arrow = lay["lead"] * 0.5 if delta else 0.0
    total = speed_sz.x + (0.4 * k + arrow + 0.15 * k + delta_sz.x if delta else 0.0)
    lx = right - total
    ly = y + lay["bars"] - speed_sz.y * 0.85
    lx += draw_value_with_unit(self._font_bold, speed, lay["lead"], self._font_semi_bold, speed_unit, lay["lead_unit"], lx, ly, ink,
                               rgba(INK, 0.7 * alpha), gap=0.2 * k).x
    if delta:
      lx += 0.4 * k
      lx += draw_delta_arrow(int(np.sign(delta)), lx, ly + speed_sz.y * 0.5, arrow, ink) + 0.15 * k
      draw_text(self._font_bold, delta_text, lay["lead"], lx, ly, ink)
