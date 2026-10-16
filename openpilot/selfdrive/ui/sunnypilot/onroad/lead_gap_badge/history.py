"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
from collections import deque

import pyray as rl

from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, WHITE, ZONE_RGB, rgba, roundness, zone_bands
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone, LeadGapReading

WINDOW = 30.0         # s of history
SAMPLE_PERIOD = 0.25  # s
GAP_MAX = 3.0         # s at the top of the graph
SPARKLINE_HEIGHT = 130
SPARKLINE_GAP = 14    # px between the badge and the graph
PLOT_INSET = 14
TRACE_WIDTH = 10
NOW_DOT = 14


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


def draw_sparkline(history: GapHistory, x: float, y: float, w: float, t_follow: float, target_gap: float | None, alpha: float) -> None:
  """No text: the trace sits over the zone colors, so how high it is reads as close or comfortable at a glance,
  and its slope shows whether the gap is shrinking."""
  h = SPARKLINE_HEIGHT
  rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, h), roundness(20, w, h), 10, rgba(BLACK, 0.6 * alpha))
  plot_x, plot_y = x + PLOT_INSET, y + PLOT_INSET
  plot_w, plot_h = w - 2 * PLOT_INSET, h - 2 * PLOT_INSET

  def gy(gap: float) -> float:
    return plot_y + plot_h * (1 - gap / GAP_MAX)

  for zone, lo, hi in zone_bands(t_follow, GAP_MAX):
    rl.draw_rectangle_rec(rl.Rectangle(plot_x, gy(hi), plot_w, gy(lo) - gy(hi)), rgba(ZONE_RGB[zone], 0.16 * alpha))

  if target_gap is not None and 0 < target_gap < GAP_MAX:
    ty = gy(target_gap)
    dash, step = 18.0, 30.0
    dx = 0.0
    while dx < plot_w:
      rl.draw_line_ex(rl.Vector2(plot_x + dx, ty), rl.Vector2(plot_x + min(dx + dash, plot_w), ty), 4, rgba(WHITE, 0.6 * alpha))
      dx += step

  samples = history.samples
  slots = samples.maxlen or 1
  start = slots - len(samples)
  col_w = plot_w / max(slots - 1, 1)
  points: list[tuple[rl.Vector2, GapZone] | None] = []
  for i, (gap, zone) in enumerate(samples):
    if math.isnan(gap):
      points.append(None)
      continue
    p = rl.Vector2(plot_x + col_w * (start + i), gy(gap))
    points.append((p, zone))
    # filled area under the trace
    rl.draw_rectangle_rec(rl.Rectangle(p.x - col_w / 2, p.y, col_w + 0.5, plot_y + plot_h - p.y), rgba(ZONE_RGB[zone], 0.3 * alpha))

  half = TRACE_WIDTH / 2
  prev = None
  for item in points:
    if item is None:
      prev = None
      continue
    p, zone = item
    color = rgba(ZONE_RGB[zone], alpha)
    if prev is not None:
      rl.draw_line_ex(prev, p, TRACE_WIDTH, color)
    rl.draw_circle_v(p, half, color)  # round joints
    prev = p

  last = next((item for item in reversed(points) if item is not None), None)
  if last is not None:
    p, zone = last
    rl.draw_circle_v(p, NOW_DOT + 4, rgba(WHITE, alpha))
    rl.draw_circle_v(p, NOW_DOT, rgba(ZONE_RGB[zone], alpha))
