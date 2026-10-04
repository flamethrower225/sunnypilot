"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
from collections import deque

import pyray as rl

from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, WHITE, ZONE_RGB, rgba, draw_text, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone, LeadGapReading
from openpilot.system.ui.lib.multilang import tr

WINDOW = 30.0         # s of history
SAMPLE_PERIOD = 0.25  # s
GAP_MAX = 3.0         # s at the top of the graph
SPARKLINE_HEIGHT = 74
SPARKLINE_GAP = 14    # px between the badge and the graph


class GapHistory:
  """The last WINDOW seconds of time gap, sampled at a fixed rate so the graph scrolls at a steady speed.
  Samples are (gap s, zone); gap is nan when there was no lead or the car was crawling."""

  def __init__(self, dt: float):
    self._dt = dt
    self._timer = 0.0
    self.samples: deque[tuple[float, GapZone]] = deque(maxlen=int(round(WINDOW / SAMPLE_PERIOD)))

  def update(self, reading: LeadGapReading) -> None:
    self._timer += self._dt
    if self._timer + 1e-6 < SAMPLE_PERIOD:
      return
    self._timer -= SAMPLE_PERIOD
    if reading.present and not reading.low_speed and math.isfinite(reading.gap):
      self.samples.append((min(reading.gap, GAP_MAX), reading.zone))
    else:
      self.samples.append((math.nan, GapZone.NONE))

  def clear(self) -> None:
    self.samples.clear()
    self._timer = 0.0


def draw_sparkline(font: rl.Font, history: GapHistory, x: float, y: float, w: float, target_gap: float | None, alpha: float) -> None:
  h = SPARKLINE_HEIGHT
  rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, h), roundness(18, w, h), 10, rgba(BLACK, 0.55 * alpha))

  label = tr("{}s").format(int(WINDOW))
  label_sz = draw_text(font, label, 20, x + 14, y + h - 30, rgba(WHITE, 0.5 * alpha))
  plot_x, plot_w = x + 14 + label_sz.x + 12, w - (14 + label_sz.x + 12) - 18
  plot_y, plot_h = y + 10, h - 20

  def gy(gap: float) -> float:
    return plot_y + plot_h * (1 - gap / GAP_MAX)

  if target_gap is not None and 0 < target_gap < GAP_MAX:
    ty = gy(target_gap)
    dash, step = 10.0, 18.0
    dx = 0.0
    while dx < plot_w:
      rl.draw_line_ex(rl.Vector2(plot_x + dx, ty), rl.Vector2(plot_x + min(dx + dash, plot_w), ty), 2, rgba(WHITE, 0.35 * alpha))
      dx += step

  samples = history.samples
  slots = samples.maxlen or 1
  start = slots - len(samples)
  prev = None
  last_point = None
  for i, (gap, zone) in enumerate(samples):
    if math.isnan(gap):
      prev = None
      continue
    point = rl.Vector2(plot_x + plot_w * (start + i) / max(slots - 1, 1), gy(gap))
    if prev is not None:
      rl.draw_line_ex(prev, point, 4, rgba(ZONE_RGB[zone], alpha))
    prev = point
    last_point = (point, zone)

  if last_point is not None:
    point, zone = last_point
    rl.draw_circle_v(point, 7, rgba(ZONE_RGB[zone], alpha))
