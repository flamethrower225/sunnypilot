"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
from dataclasses import dataclass

import numpy as np
import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone, OPEN_GAP, CLOSE_LIMITS, CRITICAL_LIMITS, \
  CAUTION_T_FOLLOW_FACTOR, FT_PER_M, STOP_DISTANCE, LeadGapReading, speed_delta
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.lib.text_measure import measure_text_cached

ZONE_RGB = {
  GapZone.NONE: (154, 163, 160),
  GapZone.OPEN: (90, 184, 255),
  GapZone.GOOD: (52, 209, 122),
  GapZone.CAUTION: (255, 210, 63),
  GapZone.CLOSE: (255, 138, 43),
  GapZone.CRITICAL: (255, 77, 79),
}

ZONE_LABELS = {
  GapZone.NONE: tr_noop("No lead"),
  GapZone.OPEN: tr_noop("Open road"),
  GapZone.GOOD: tr_noop("Good gap"),
  GapZone.CAUTION: tr_noop("Tight"),
  GapZone.CLOSE: tr_noop("Close"),
  GapZone.CRITICAL: tr_noop("Too close"),
}

INK = (11, 15, 14)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def rgba(rgb, alpha: float) -> rl.Color:
  return rl.Color(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(np.clip(alpha, 0.0, 1.0) * 255))


def zone_bands(t_follow: float, max_gap: float) -> list[tuple[GapZone, float, float]]:
  """Gap ranges (s) of each zone ignoring time to collision, for drawing scales."""
  caution = max(CLOSE_LIMITS[0], CAUTION_T_FOLLOW_FACTOR * t_follow)
  bands = [
    (GapZone.CRITICAL, 0.0, CRITICAL_LIMITS[0]),
    (GapZone.CLOSE, CRITICAL_LIMITS[0], CLOSE_LIMITS[0]),
    (GapZone.CAUTION, CLOSE_LIMITS[0], caution),
    (GapZone.GOOD, caution, OPEN_GAP),
    (GapZone.OPEN, OPEN_GAP, max_gap),
  ]
  return [(z, a, min(b, max_gap)) for z, a, b in bands if a < max_gap and b > a]


def distance_bands(max_distance: float) -> list[tuple[GapZone, float, float]]:
  """Distance ranges (m) of each zone at a crawl, ignoring time to collision."""
  caution = CAUTION_T_FOLLOW_FACTOR * STOP_DISTANCE
  bands = [
    (GapZone.CRITICAL, 0.0, CRITICAL_LIMITS[1]),
    (GapZone.CLOSE, CRITICAL_LIMITS[1], CLOSE_LIMITS[1]),
    (GapZone.CAUTION, CLOSE_LIMITS[1], caution),
    (GapZone.GOOD, caution, max_distance),
  ]
  return [(z, a, min(b, max_distance)) for z, a, b in bands if a < max_distance and b > a]


@dataclass(frozen=True)
class GaugeScale:
  """What a gauge plots: time gap in seconds normally, distance in meters at a crawl."""
  max_value: float
  value: float
  bands: list[tuple[GapZone, float, float]]
  ticks: list[tuple[float, str]]
  unit: str
  target: float | None


def gauge_scale(reading: LeadGapReading, is_metric: bool, max_gap: float, distance_ticks_metric: tuple[int, ...],
                distance_ticks_imperial: tuple[int, ...]) -> GaugeScale:
  if reading.low_speed:
    ticks, unit, to_m = (distance_ticks_metric, "m", 1.0) if is_metric else (distance_ticks_imperial, "ft", 1 / FT_PER_M)
    max_distance = ticks[-1] * to_m
    return GaugeScale(max_distance, min(reading.d_rel, max_distance), distance_bands(max_distance),
                      [(t * to_m, str(t)) for t in ticks], unit, None)

  value = min(reading.gap, max_gap) if math.isfinite(reading.gap) else max_gap
  ticks = [(float(t), str(t)) for t in range(int(max_gap) + 1)]
  return GaugeScale(max_gap, value, zone_bands(reading.t_follow, max_gap), ticks, "s", reading.target_gap)


class ZoneColor:
  """Fades between zone colors instead of snapping."""

  def __init__(self, dt: float, rc: float = 0.15):
    self._filter = FirstOrderFilter(np.array(ZONE_RGB[GapZone.NONE], dtype=float), rc, dt, initialized=False)

  def update(self, zone: GapZone) -> np.ndarray:
    return self._filter.update(np.array(ZONE_RGB[zone], dtype=float))

  @property
  def rgb(self) -> np.ndarray:
    return self._filter.x


def zone_label(reading: LeadGapReading) -> str:
  return tr(ZONE_LABELS[reading.zone])


def delta_label(reading: LeadGapReading, is_metric: bool) -> tuple[int, str]:
  """Returns (direction, text): direction is -1 when the lead is slower, 1 when faster, 0 when matched."""
  delta = speed_delta(reading.v_rel, is_metric)
  if delta == 0:
    return 0, tr("same speed")
  if delta < 0:
    return -1, tr("{} slower").format(-delta)
  return 1, tr("{} faster").format(delta)


def draw_text(font: rl.Font, text: str, size: float, x: float, y: float, color: rl.Color, spacing: float = 0) -> rl.Vector2:
  rl.draw_text_ex(font, text, rl.Vector2(x, y), size, spacing, color)
  return measure_text_cached(font, text, int(size), spacing)


def draw_text_centered(font: rl.Font, text: str, size: float, cx: float, cy: float, color: rl.Color, spacing: float = 0) -> rl.Vector2:
  sz = measure_text_cached(font, text, int(size), spacing)
  rl.draw_text_ex(font, text, rl.Vector2(cx - sz.x / 2, cy - sz.y / 2), size, spacing, color)
  return sz


def draw_value_with_unit(value_font: rl.Font, value: str, value_size: float, unit_font: rl.Font, unit: str, unit_size: float,
                         x: float, y: float, value_color: rl.Color, unit_color: rl.Color, gap: float = 6.0) -> rl.Vector2:
  """Draws a number with a smaller unit sharing its baseline, top-left anchored. Returns the total size."""
  value_sz = draw_text(value_font, value, value_size, x, y, value_color)
  unit_sz = measure_text_cached(unit_font, unit, int(unit_size)) if unit else rl.Vector2(0, 0)
  if unit:
    # Inter's baseline sits at roughly 80% of the line box
    unit_y = y + value_sz.y * 0.8 - unit_sz.y * 0.8
    draw_text(unit_font, unit, unit_size, x + value_sz.x + gap, unit_y, unit_color)
  return rl.Vector2(value_sz.x + (gap + unit_sz.x if unit else 0), value_sz.y)


def measure_value_with_unit(value_font: rl.Font, value: str, value_size: float, unit_font: rl.Font, unit: str, unit_size: float,
                            gap: float = 6.0) -> rl.Vector2:
  value_sz = measure_text_cached(value_font, value, int(value_size))
  if not unit:
    return value_sz
  unit_sz = measure_text_cached(unit_font, unit, int(unit_size))
  return rl.Vector2(value_sz.x + gap + unit_sz.x, value_sz.y)


def draw_triangle(p1, p2, p3, color: rl.Color) -> None:
  """raylib only fills triangles wound counter-clockwise on screen, so fix the winding."""
  cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
  if cross > 0:
    p2, p3 = p3, p2
  rl.draw_triangle(rl.Vector2(*p1), rl.Vector2(*p2), rl.Vector2(*p3), color)


def draw_delta_arrow(direction: int, x: float, cy: float, size: float, color: rl.Color) -> float:
  """Small up or down triangle; returns the width used."""
  if direction == 0:
    return 0.0
  half = size / 2
  if direction < 0:
    draw_triangle((x, cy - half * 0.7), (x + size, cy - half * 0.7), (x + half, cy + half * 0.7), color)
  else:
    draw_triangle((x, cy + half * 0.7), (x + size, cy + half * 0.7), (x + half, cy - half * 0.7), color)
  return size


def draw_arc(center: rl.Vector2, radius: float, width: float, start_angle: float, end_angle: float, color: rl.Color,
             round_caps: bool = False) -> None:
  """Ring segment, angles in degrees clockwise from 3 o'clock (raylib's convention)."""
  if end_angle - start_angle < 0.05:
    return
  segments = max(8, int((end_angle - start_angle) / 3))
  rl.draw_ring(center, radius - width / 2, radius + width / 2, start_angle, end_angle, segments, color)
  if round_caps:
    for angle, cap_start in ((start_angle, start_angle - 180), (end_angle, end_angle)):
      rad = math.radians(angle)
      cap_center = rl.Vector2(center.x + radius * math.cos(rad), center.y + radius * math.sin(rad))
      rl.draw_circle_sector(cap_center, width / 2, cap_start, cap_start + 180, 12, color)


def draw_car_glyph(x: float, y: float, w: float, color: rl.Color) -> None:
  """Top-down car pointing right, w wide and w/2 tall, top-left anchored."""
  h = w / 2
  sx, sy = w / 48, h / 24
  body = rl.Rectangle(x + 1 * sx, y + 2 * sy, 46 * sx, 20 * sy)
  rl.draw_rectangle_rounded(body, 0.7, 8, color)
  shade = rl.Color(0, 0, 0, int(color.a * 0.45))
  rl.draw_rectangle_rounded(rl.Rectangle(x + 27 * sx, y + 5 * sy, 9 * sx, 14 * sy), 0.5, 4, shade)
  rl.draw_rectangle_rounded(rl.Rectangle(x + 9 * sx, y + 5.5 * sy, 7 * sx, 13 * sy), 0.5, 4, rl.Color(0, 0, 0, int(color.a * 0.3)))


def roundness(radius: float, w: float, h: float) -> float:
  return min(1.0, 2 * radius / max(min(w, h), 1.0))
