"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

A glow along the inside of the screen edge, shared by Pace Border and the hard brake warning.
"""
from dataclasses import dataclass

import pyray as rl

from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import rgba, roundness

EDGE_RADIUS = 36.0   # px, corner radius at the outermost band


@dataclass(frozen=True)
class _Band:
  rect: rl.Rectangle
  roundness: float
  thickness: float
  falloff: float


_cache: dict[tuple, list[_Band]] = {}
MAX_CACHED = 32


def edge_bands(rect: rl.Rectangle, depth: float, layers: int) -> list[_Band]:
  """Nested rounded outlines, each inset deeper and fainter, so the edge fades into the picture. Bands overlap by half so there are
  no seams, and the corner radius shrinks with the inset so they stay parallel. Only rebuilt when the size changes: depth is
  rounded to whole pixels, which no one can see, so a glow that stays put builds its rectangles once."""
  depth = float(round(depth))
  key = (rect.x, rect.y, rect.width, rect.height, depth, layers)
  bands = _cache.get(key)
  if bands is None:
    step = depth / layers
    thickness = 2 * step
    bands = []
    for i in range(layers):
      inset = i * step
      layer = rl.Rectangle(rect.x + inset, rect.y + inset, rect.width - 2 * inset, rect.height - 2 * inset)
      radius = EDGE_RADIUS - inset
      if radius < thickness:  # raylib's thick outline breaks when the thickness passes the corner radius, and this close it's square
        radius = 0.0
      bands.append(_Band(layer, roundness(radius, layer.width, layer.height) if radius else 0.0, thickness, (1.0 - i / layers) ** 2))
    if len(_cache) >= MAX_CACHED:
      _cache.clear()
    _cache[key] = bands
  return bands


def draw_edge_glow(rect: rl.Rectangle, rgb, depth: float, alpha: float, layers: int) -> None:
  if alpha < 0.01:
    return
  for band in edge_bands(rect, depth, layers):
    rl.draw_rectangle_rounded_lines_ex(band.rect, band.roundness, 12, band.thickness, rgba(rgb, alpha * band.falloff))
