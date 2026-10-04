"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np
import pyray as rl

from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, WHITE
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import LeadGapReading, primary_value, speed_value
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.shader_polygon import draw_polygon, Gradient

# stock throttle path colors (rgb, alpha) the zone tint fades in from
BASE_PATH_COLORS = [((13, 248, 122), 0.4), ((114, 255, 92), 0.35), ((114, 255, 92), 0.0)]
ZONE_PATH_ALPHAS = (0.6, 0.45, 0.0)

DECAL_NEAR = 9.0        # m ahead where the nearest painted line starts
DECAL_WIDTH = 2.6       # m across the lane
DECAL_LEAD_CLEARANCE = 3.0  # m kept free in front of the lead
GAP_LINE_LENGTH = 7.5   # m along the road for the gap numerals
LEAD_LINE_LENGTH = 3.2  # m along the road for the lead speed
LINE_SPACING = 1.0      # m
MIN_DECAL_ROOM = 6.0    # m, below this only the path is tinted
STRIPS = 3              # subdivisions per glyph, keeps the perspective from warping


def path_gradient(zone_rgb, alpha: float) -> Gradient:
  colors = []
  for (base_rgb, base_a), zone_a in zip(BASE_PATH_COLORS, ZONE_PATH_ALPHAS, strict=True):
    rgb = np.array(base_rgb) * (1 - alpha) + np.array(zone_rgb) * alpha
    a = base_a * (1 - alpha) + zone_a * alpha
    colors.append(rl.Color(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(a * 255)))
  return Gradient(start=(0.0, 1.0), end=(0.0, 0.0), colors=colors, stops=[0.0, 0.5, 1.0])


def decal_lines(reading: LeadGapReading, is_metric: bool) -> list[tuple[str, float, float, bool]]:
  """(text, along-road length m, share of the lane width, colored by zone) from farthest to nearest."""
  value, unit = primary_value(reading, is_metric)
  speed_unit = tr("KM/H") if is_metric else tr("MPH")
  gap_text = f"{value}{unit}" if unit == "s" else f"{value} {unit.upper()}"
  return [
    (f"{speed_value(reading.v_lead, is_metric)} {speed_unit}", LEAD_LINE_LENGTH, 0.75, False),
    (gap_text, GAP_LINE_LENGTH, 1.0, True),
  ]


def decal_room(reading: LeadGapReading) -> float:
  return reading.d_rel - DECAL_LEAD_CLEARANCE - DECAL_NEAR


class RoadPaint:
  """Paint style: the path takes the zone color and the gap is painted on the road like a lane marking."""

  def __init__(self):
    self._font = gui_app.font(FontWeight.BOLD)

  def draw_path(self, rect: rl.Rectangle, path_points: np.ndarray, zone_rgb, alpha: float) -> None:
    draw_polygon(rect, path_points, gradient=path_gradient(zone_rgb, alpha))

  def draw_decal(self, projector, reading: LeadGapReading, zone_rgb, alpha: float, is_metric: bool) -> None:
    if not projector.ready or decal_room(reading) < MIN_DECAL_ROOM:
      return

    lines = decal_lines(reading, is_metric)
    total = sum(length for _, length, _, _ in lines) + LINE_SPACING * (len(lines) - 1)
    scale = min(1.0, decal_room(reading) / total)
    x_far = DECAL_NEAR + total * scale
    center = float(projector.path_y(DECAL_NEAR + total * scale / 2))

    for text, length, width_share, zone_colored in lines:
      length *= scale
      color = rgba_tuple(zone_rgb if zone_colored else WHITE, 0.92 * alpha)
      outline = rgba_tuple(BLACK, 0.5 * alpha)
      width = DECAL_WIDTH * width_share
      # dark outline first, nudged a little each way, then the paint
      for du, dv in ((-0.06, 0.0), (0.06, 0.0), (0.0, -0.25), (0.0, 0.25)):
        self._draw_ground_text(projector, text, center + du, x_far + dv, length, width, outline)
      self._draw_ground_text(projector, text, center, x_far, length, width, color)
      x_far -= length + LINE_SPACING * scale

  def _draw_ground_text(self, projector, text: str, center_y: float, x_far: float, length: float, width: float, color) -> None:
    """Lays text flat on the road: the top of the glyphs at x_far, their bottom at x_far - length, centered on center_y."""
    font = self._font
    scale = 1.0  # glyph metrics in font pixels, stretched below to the ground box
    pen, boxes = 0.0, []
    for ch in text:
      idx = rl.get_glyph_index(font, ord(ch))
      glyph, rec = font.glyphs[idx], font.recs[idx]
      pad = font.glyphPadding
      x0 = pen + glyph.offsetX * scale - pad
      y0 = glyph.offsetY * scale - pad
      boxes.append((x0, y0, x0 + rec.width + 2 * pad, y0 + rec.height + 2 * pad, rec, pad))
      pen += (glyph.advanceX if glyph.advanceX else rec.width) * scale
    if not boxes or pen <= 0:
      return

    line_h = float(font.baseSize)
    tex_w, tex_h = font.texture.width, font.texture.height

    def ground(px, py):
      lateral = center_y + (px / pen - 0.5) * width
      along = x_far - (py / line_h) * length
      return projector.to_screen(along, lateral)

    rl.rl_set_texture(font.texture.id)
    rl.rl_begin(rl.RL_QUADS)
    for x0, y0, x1, y1, rec, pad in boxes:
      u0, u1 = (rec.x - pad) / tex_w, (rec.x + rec.width + pad) / tex_w
      for s in range(STRIPS):
        ya, yb = y0 + (y1 - y0) * s / STRIPS, y0 + (y1 - y0) * (s + 1) / STRIPS
        va = ((rec.y - pad) + (rec.height + 2 * pad) * s / STRIPS) / tex_h
        vb = ((rec.y - pad) + (rec.height + 2 * pad) * (s + 1) / STRIPS) / tex_h
        corners = [ground(x0, ya), ground(x0, yb), ground(x1, yb), ground(x1, ya)]
        if any(c is None for c in corners):
          continue
        uvs = [(u0, va), (u0, vb), (u1, vb), (u1, va)]
        # raylib culls clockwise quads, so wind them counter-clockwise on screen
        (ax, ay), (bx, by), (cx, cy) = corners[0], corners[1], corners[2]
        if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) > 0:
          corners, uvs = corners[::-1], uvs[::-1]
        for (sx, sy), (u, v) in zip(corners, uvs, strict=True):
          rl.rl_color4ub(*color)
          rl.rl_tex_coord2f(u, v)
          rl.rl_vertex2f(sx, sy)
    rl.rl_end()
    rl.rl_set_texture(0)


def rgba_tuple(rgb, alpha: float) -> tuple[int, int, int, int]:
  return int(rgb[0]), int(rgb[1]), int(rgb[2]), int(np.clip(alpha, 0.0, 1.0) * 255)
