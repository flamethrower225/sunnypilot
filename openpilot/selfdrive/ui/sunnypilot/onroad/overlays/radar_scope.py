"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Radar scope: a top-down minimap of the model's lane lines and every radar track, in the left column of the HUD.
"""
import math
import time
from dataclasses import dataclass

import numpy as np
import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.developer_ui import get_bottom_dev_ui_offset
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, INK, WHITE, ZONE_RGB, rgba, draw_text, \
  draw_triangle, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone, FT_PER_M
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_tracks import CLASS_RGB, RadarFeed, RadarTrack, TrackClass, display_zone
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

PANEL_W, PANEL_H = 400.0, 460.0
PANEL_X, PANEL_Y = 40.0, 290.0     # from the HUD rect, clear of the MAX box and speed limit sign above
ROCKET_FUEL_SHIFT = 40.0           # the rocket fuel bar takes the left edge
PAD = 16.0
HEADER_H = 48.0
EGO_W, EGO_H = 27.0, 52.0
BOTTOM_PAD = 16.0

MAX_RANGE = 120.0                  # m at the top of the plot
MAX_LATERAL = 12.0                 # m either side
RINGS_METRIC = (25.0, 50.0, 100.0)
RINGS_IMPERIAL = (100.0, 200.0, 300.0)   # ft
LANE_SAMPLES = 30
LANE_DISTANCES = np.linspace(0.0, 1.0, LANE_SAMPLES) ** 2 * MAX_RANGE   # even steps on the compressed range axis

VELOCITY_TICK = 5.0                # px per m/s of closing or opening speed
MAX_TICK = 34.0
MIN_TICK_SPEED = 0.5               # m/s
PING_PERIOD = 3.6                  # s


@dataclass(frozen=True)
class ScopeFrame:
  """Maps car-frame meters to screen pixels. Range is square-root compressed so the near field, where it matters,
  gets most of the room."""
  cx: float
  origin_y: float      # screen y of the front bumper
  px_per_m: float      # lateral
  height: float        # px from the bumper to MAX_RANGE

  def range_frac(self, d):
    return np.sqrt(np.clip(d, 0.0, MAX_RANGE) / MAX_RANGE)

  def project(self, d: float, y_right: float) -> tuple[float, float]:
    return self.cx + y_right * self.px_per_m, self.origin_y - float(self.range_frac(d)) * self.height

  def project_arrays(self, d: np.ndarray, y_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return self.cx + y_right * self.px_per_m, self.origin_y - self.range_frac(d) * self.height

  @staticmethod
  def visible(d: float, y_right: float) -> bool:
    return 0.0 < d <= MAX_RANGE and abs(y_right) <= MAX_LATERAL


def scope_rect(rect: rl.Rectangle, rocket_fuel: bool, bottom_dev_ui: float = 0.0) -> rl.Rectangle:
  """bottom_dev_ui is how far the developer UI's bottom bar pushes the driver monitoring icon up; the panel gives that room back."""
  return rl.Rectangle(rect.x + PANEL_X + (ROCKET_FUEL_SHIFT if rocket_fuel else 0.0), rect.y + PANEL_Y, PANEL_W, PANEL_H - bottom_dev_ui)


def scope_frame(panel: rl.Rectangle) -> ScopeFrame:
  origin_y = panel.y + panel.height - BOTTOM_PAD - EGO_H
  top = panel.y + HEADER_H + 10.0
  return ScopeFrame(panel.x + panel.width / 2, origin_y, (panel.width - 2 * PAD) / (2 * MAX_LATERAL), origin_y - top)


def ring_ranges(is_metric: bool) -> list[tuple[float, str]]:
  """(range in m, label) of each range ring."""
  if is_metric:
    return [(r, f"{r:.0f}") for r in RINGS_METRIC]
  return [(r / FT_PER_M, f"{r:.0f}") for r in RINGS_IMPERIAL]


def ring_points(radius: float, step: float = 1.5) -> list[tuple[float, float]]:
  """(d, y_right) along a circle of this radius around the bumper, clipped to the scope's lateral extent."""
  points = []
  for i in range(int(2 * MAX_LATERAL / step) + 1):
    lateral = -MAX_LATERAL + i * step
    if abs(lateral) < radius:
      points.append((math.sqrt(radius * radius - lateral * lateral), lateral))
  return points


class RadarScope:
  """Top-down minimap: lane lines from the model, range rings, and a dot per radar track.
  Dots are cyan when moving the same way, gray when stationary, hollow pink when oncoming, and severity colored when closing on us.
  The lead has a solid ring, the second lead a dashed one, and a tick on each moving dot points the way it's pulling (up is faster than us)."""

  def __init__(self):
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi_bold = gui_app.font(FontWeight.SEMI_BOLD)
    dt = 1 / gui_app.target_fps
    self._feed = RadarFeed(dt)
    self._alpha = FirstOrderFilter(0.0, 0.2, dt)
    self._lane_y: np.ndarray | None = None     # (4, LANE_SAMPLES) lateral position at LANE_DISTANCES, NaN where the model has none
    self._lane_alpha = np.zeros(4)

  def update(self) -> None:
    if not ui_state.radar_scope:
      self._alpha.x = 0.0
      return
    sm = ui_state.sm
    self._feed.update(sm, ui_state.started_frame, time.monotonic())
    self._alpha.update(1.0 if self._feed.available else 0.0)
    if self._alpha.x > 0.01:
      self._update_lanes(sm)

  def _update_lanes(self, sm) -> None:
    self._lane_y = None
    if sm.recv_frame['modelV2'] < ui_state.started_frame:
      return
    model = sm['modelV2']
    lines = model.laneLines
    if len(lines) < 4:
      return
    probs = np.array(model.laneLineProbs, dtype=float)
    lane_y = np.full((4, LANE_SAMPLES), np.nan)
    for i in range(4):
      xs = np.array(lines[i].x, dtype=float)
      ys = np.array(lines[i].y, dtype=float)
      if xs.size < 2 or xs.size != ys.size:
        continue
      inside = (LANE_DISTANCES >= xs[0]) & (LANE_DISTANCES <= xs[-1])
      lane_y[i, inside] = np.interp(LANE_DISTANCES[inside], xs, ys)
    self._lane_y = lane_y
    self._lane_alpha = np.array([0.35 + 0.65 * float(np.clip(probs[i], 0.0, 1.0)) if i < probs.size else 0.5 for i in range(4)])

  def render(self, rect: rl.Rectangle) -> None:
    if not ui_state.radar_scope:
      return
    alpha = self._alpha.x
    if alpha < 0.01:
      return

    panel = scope_rect(rect, ui_state.rocket_fuel, get_bottom_dev_ui_offset())
    frame = scope_frame(panel)
    now = rl.get_time()

    rl.draw_rectangle_rounded(panel, roundness(22, panel.width, panel.height), 16, rgba(BLACK, 0.62 * alpha))
    rl.draw_rectangle_rounded_lines_ex(panel, roundness(22, panel.width, panel.height), 16, 3, rgba(WHITE, 0.28 * alpha))
    self._draw_header(panel, alpha)
    self._draw_rings(frame, ui_state.is_metric, now, alpha)
    self._draw_lanes(frame, alpha)
    self._draw_tracks(frame, now, alpha)
    self._draw_ego(frame, alpha)

  def _draw_header(self, panel: rl.Rectangle, alpha: float) -> None:
    title_y = panel.y + 15
    draw_text(self._font_bold, tr("RADAR"), 21, panel.x + PAD + 6, title_y, rgba(WHITE, 0.9 * alpha), spacing=2.5)

    count = len(self._feed.tracks)
    caption = f"{count} " + (tr("TRACK") if count == 1 else tr("TRACKS"))
    size = measure_text_cached(self._font_semi_bold, caption, 19, 1.0)
    draw_text(self._font_semi_bold, caption, 19, panel.x + panel.width - PAD - 6 - size.x, title_y + 1, rgba(WHITE, 0.6 * alpha), spacing=1.0)

    line_y = panel.y + HEADER_H
    rl.draw_line_ex(rl.Vector2(panel.x + PAD, line_y), rl.Vector2(panel.x + panel.width - PAD, line_y), 1.5, rgba(WHITE, 0.14 * alpha))

  def _draw_ring(self, frame: ScopeFrame, radius: float, thickness: float, color: rl.Color, dashed: bool = False) -> None:
    pts = [frame.project(d, y) for d, y in ring_points(radius)]
    for i in range(len(pts) - 1):
      if dashed and i % 2:
        continue
      rl.draw_line_ex(rl.Vector2(*pts[i]), rl.Vector2(*pts[i + 1]), thickness, color)

  def _draw_rings(self, frame: ScopeFrame, is_metric: bool, now: float, alpha: float) -> None:
    rings = ring_ranges(is_metric)
    for radius, _ in rings:
      self._draw_ring(frame, radius, 1.6, rgba(WHITE, 0.17 * alpha), dashed=True)

    # a faint ping sweeping outward from the bumper
    phase = (now % PING_PERIOD) / PING_PERIOD
    if phase < 0.85:
      self._draw_ring(frame, phase / 0.85 * MAX_RANGE * 0.98, 2.5, rgba(WHITE, 0.14 * (1 - phase / 0.85) * alpha))

    unit = "m" if is_metric else "ft"
    for i, (radius, label) in enumerate(rings):
      d, y = ring_points(radius)[0]
      x, py = frame.project(d, y)
      text = f"{label} {unit}" if i == len(rings) - 1 else label
      size = measure_text_cached(self._font_semi_bold, text, 15)
      draw_text(self._font_semi_bold, text, 15, x + 2, py - size.y - 1, rgba(WHITE, 0.52 * alpha))

  def _draw_lanes(self, frame: ScopeFrame, alpha: float) -> None:
    if self._lane_y is None:
      return
    xs, ys = [], []
    for i in range(4):
      x, y = frame.project_arrays(LANE_DISTANCES, self._lane_y[i])
      xs.append(x)
      ys.append(y)

    # our lane, shaded
    left, right = self._lane_y[1], self._lane_y[2]
    fill = rgba(CLASS_RGB[TrackClass.MOVING], 0.1 * alpha)
    for j in range(LANE_SAMPLES - 1):
      if not (np.isfinite(left[j:j + 2]).all() and np.isfinite(right[j:j + 2]).all()):
        continue
      if max(abs(left[j]), abs(right[j]), abs(left[j + 1]), abs(right[j + 1])) > MAX_LATERAL:
        continue
      a, b, c, d = (xs[1][j], ys[1][j]), (xs[2][j], ys[2][j]), (xs[2][j + 1], ys[2][j + 1]), (xs[1][j + 1], ys[1][j + 1])
      draw_triangle(a, b, c, fill)
      draw_triangle(a, c, d, fill)

    for i in (0, 3, 1, 2):
      bright = i in (1, 2)
      color = rgba(WHITE, (0.85 if bright else 0.3) * self._lane_alpha[i] * alpha)
      thickness = 3.5 if bright else 2.5
      lat = self._lane_y[i]
      for j in range(LANE_SAMPLES - 1):
        if not (np.isfinite(lat[j]) and np.isfinite(lat[j + 1])) or max(abs(lat[j]), abs(lat[j + 1])) > MAX_LATERAL:
          continue
        rl.draw_line_ex(rl.Vector2(xs[i][j], ys[i][j]), rl.Vector2(xs[i][j + 1], ys[i][j + 1]), thickness, color)

  def _draw_tracks(self, frame: ScopeFrame, now: float, alpha: float) -> None:
    feed = self._feed
    lead_one, lead_two = feed.lead_one, feed.lead_two
    tracks = [t for t in feed.tracks if frame.visible(t.d_rel, t.y_right)]
    # stationary clutter first so moving traffic and threats land on top
    tracks.sort(key=lambda t: (t.cls != TrackClass.STATIONARY, t.ttc < math.inf))
    for t in tracks:
      self._draw_dot(frame, t, now, alpha)

    for lead, ring in ((lead_two, "dashed"), (lead_one, "solid")):
      if lead is not None and frame.visible(lead.d_rel, lead.y_right):
        self._draw_lead_ring(frame, lead, ring, now, alpha)
    # vision-only leads have no dot to ring, so mark where the camera sees them
    for lead, ring in zip(feed.vision_leads, ("solid", "dashed"), strict=True):
      if lead is not None and frame.visible(lead.d_rel, lead.y_right):
        self._draw_lead_ring(frame, lead, ring, now, alpha * 0.8)

  @staticmethod
  def _draw_dot(frame: ScopeFrame, t: RadarTrack, now: float, alpha: float) -> None:
    x, y = frame.project(t.d_rel, t.y_right)
    zone = display_zone(t)
    rgb = ZONE_RGB[zone] if zone is not None else CLASS_RGB[t.cls]
    stationary = t.cls == TrackClass.STATIONARY
    a = (0.5 if stationary else 0.95) * alpha
    radius = 5.0 if stationary else 7.5

    if zone is not None:
      pulse = 0.5 + 0.5 * math.sin(now * 2 * math.pi / (0.6 if zone == GapZone.CRITICAL else 1.2))
      rl.draw_circle_v(rl.Vector2(x, y), radius + 4 + 3 * pulse, rgba(rgb, (0.14 + 0.1 * pulse) * alpha))
      radius = 8.5

    if not stationary and abs(t.v_rel) >= MIN_TICK_SPEED:
      # points up when it's pulling away, down when we're gaining on it
      length = min(max(abs(t.v_rel) * VELOCITY_TICK, 7.0), MAX_TICK)
      sign = -1.0 if t.v_rel > 0 else 1.0
      start = y + sign * radius * 0.6
      end = y + sign * (radius + length)
      rl.draw_line_ex(rl.Vector2(x, start), rl.Vector2(x, end), 3.0, rgba(rgb, 0.75 * alpha))
      rl.draw_circle_v(rl.Vector2(x, end), 1.9, rgba(rgb, 0.75 * alpha))

    if t.cls == TrackClass.ONCOMING and zone is None:
      rl.draw_ring(rl.Vector2(x, y), radius - 2.8, radius, 0, 360, 24, rgba(rgb, a))
    else:
      rl.draw_circle_v(rl.Vector2(x, y), radius, rgba(rgb, a))
      if not stationary:
        rl.draw_circle_v(rl.Vector2(x - radius * 0.25, y - radius * 0.25), radius * 0.35, rgba(WHITE, 0.35 * alpha))

  @staticmethod
  def _draw_lead_ring(frame: ScopeFrame, lead: RadarTrack, style: str, now: float, alpha: float) -> None:
    center = rl.Vector2(*frame.project(lead.d_rel, lead.y_right))
    if style == "solid":
      pulse = 0.5 + 0.5 * math.sin(now * 2 * math.pi / 1.6)
      radius = 14.5 + 1.2 * pulse
      rl.draw_ring(center, radius - 2.6, radius, 0, 360, 36, rgba(WHITE, alpha))
      rl.draw_ring(center, radius + 2.5, radius + 6.5, 0, 360, 36, rgba(WHITE, (0.1 + 0.08 * pulse) * alpha))
    else:
      radius = 12.5
      for k in range(6):
        start = k * 60.0 + (now * 30.0) % 60.0
        rl.draw_ring(center, radius - 2.2, radius, start, start + 36.0, 8, rgba(WHITE, 0.85 * alpha))

  def _draw_ego(self, frame: ScopeFrame, alpha: float) -> None:
    cx, top = frame.cx, frame.origin_y
    rl.draw_circle_gradient(rl.Vector2(cx, top + EGO_H * 0.55), 52, rgba(WHITE, 0.16 * alpha), rgba(WHITE, 0.0))
    body = rl.Rectangle(cx - EGO_W / 2, top, EGO_W, EGO_H)
    rl.draw_rectangle_rounded(body, 0.5, 10, rgba(WHITE, alpha))
    shade = rgba(INK, 0.5 * alpha)
    rl.draw_rectangle_rounded(rl.Rectangle(cx - EGO_W * 0.36, top + EGO_H * 0.2, EGO_W * 0.72, EGO_H * 0.2), 0.5, 6, shade)
    rl.draw_rectangle_rounded(rl.Rectangle(cx - EGO_W * 0.32, top + EGO_H * 0.68, EGO_W * 0.64, EGO_H * 0.14), 0.5, 6, rgba(INK, 0.35 * alpha))
    # nose marker, the origin radar distances are measured from
    draw_triangle((cx - 5, top - 3), (cx + 5, top - 3), (cx, top - 11), rgba(WHITE, 0.9 * alpha))
