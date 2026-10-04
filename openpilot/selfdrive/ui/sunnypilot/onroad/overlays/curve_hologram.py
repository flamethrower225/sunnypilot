"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

import numpy as np
import pyray as rl

from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import INK, WHITE, ZONE_RGB, draw_triangle, draw_value_with_unit, \
  measure_value_with_unit, rgba
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.projection import RoadProjector
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.shader_polygon import Gradient, draw_polygon

# Curve hologram: when the model predicts a real bend ahead, two glowing rails run along the planned path, colored by
# the predicted lateral acceleration, and a gate stands over the road at the tightest point with an advisory speed.

SHOW_LAT_ACC = 1.0        # m/s², the rails appear when the largest predicted lateral acceleration ahead exceeds this
HIDE_LAT_ACC = 0.9        # m/s², and go away again below this
ADVISORY_LAT_ACC = 2.0    # m/s², lateral acceleration the advisory speed is computed for (A_LAT_REG_MAX in the vision controller)
APEX_MIN_X = 10.0         # m, a gate closer than this is already behind the camera's useful view
APEX_MAX_X = 120.0        # m, and one farther than this is too small to read
APEX_NEAR_PEAK = 0.97     # fraction of the peak that counts as the top of the bend (a flat bend has no single maximum)
SPEED_STEP = 5            # advisory speeds are rounded to this

RAIL_OFFSET = 1.0         # m either side of the planned path
RAIL_HALF_WIDTH = 0.08    # m
GLOW_PASSES = ((0.36, 0.06), (0.2, 0.1))  # (half width m, alpha) of the soft glow under each rail, widest first
GATE_HALF_WIDTH = 1.8     # m
GATE_HEIGHT = 1.6         # m
RAIL_X_NEAR = 3.0         # m
RAIL_X_FAR = 140.0        # m
STOP_X_NEAR = 5.0         # m, gradient stops start where the road enters the bottom of the screen
RAIL_ALPHA_X = (5.0, 20.0, 60.0, 140.0)
RAIL_ALPHA = (0.8, 0.95, 0.8, 0.3)
LOOKAHEAD_TINT = 0.65     # a rail point is colored at least this fraction of the worst lateral acceleration still ahead of it
RAIL_SAMPLES = 40
GRADIENT_STOPS = 16       # shader limit is 20

# lateral acceleration (m/s²) -> color: green below 1.3, yellow 1.3-1.6, orange at 2.0, red from 2.4
LAT_ACC_BREAKPOINTS = (1.25, 1.35, 1.6, 1.8, 2.0, 2.4)
LAT_ACC_COLORS = np.array([ZONE_RGB[z] for z in (GapZone.GOOD, GapZone.CAUTION, GapZone.CAUTION, GapZone.CLOSE, GapZone.CLOSE,
                                                  GapZone.CRITICAL)], dtype=float)

BEAD_PERIOD = 2.4         # s, a bead takes this long to run from the car to the gate
BEADS_PER_RAIL = 3
BEAD_TAIL = 3             # points per bead, the head and its streak
BEAD_TAIL_STEP = 0.02    # trail spacing as a fraction of the run


# ---- pure math ----

def lateral_accels(yaw_rates, velocities) -> np.ndarray:
  """Predicted lateral acceleration (m/s²) at each model point, |yaw rate * speed|. Same as the vision controller."""
  n = min(len(yaw_rates), len(velocities))
  return np.abs(np.asarray(yaw_rates[:n], dtype=float) * np.asarray(velocities[:n], dtype=float))


def lat_acc_color(lat_acc: float) -> tuple[float, float, float]:
  r, g, b = (float(np.interp(lat_acc, LAT_ACC_BREAKPOINTS, LAT_ACC_COLORS[:, c])) for c in range(3))
  return r, g, b


def rails_active(peak_lat_acc: float, was_active: bool) -> bool:
  """Rails come on above SHOW_LAT_ACC and stay on until the prediction drops below HIDE_LAT_ACC, so a peak hovering around
  the threshold doesn't flicker."""
  return peak_lat_acc >= (HIDE_LAT_ACC if was_active else SHOW_LAT_ACC)


def rail_severity(lat_accs, tint: float = LOOKAHEAD_TINT) -> np.ndarray:
  """Lateral acceleration used to color each point of the rails: its own value, but never less than `tint` times the worst
  value ahead of it. Far bends are squeezed into a few pixels near the horizon, so this lets the whole rail carry the
  warning on the approach instead of staying green until the last moment."""
  lat = np.asarray(lat_accs, dtype=float)
  if lat.size == 0:
    return lat
  ahead = np.maximum.accumulate(lat[::-1])[::-1]
  return np.maximum(lat, tint * ahead)


def find_apex(xs, lat_accs, min_x: float = APEX_MIN_X, max_x: float = APEX_MAX_X) -> int | None:
  """Index of the tightest point of the bend ahead, or None when it is not between min_x and max_x.

  The peak is taken over the whole horizon, so a bend that is already behind us or still far away gives no gate. Within
  the window, the strongest point among those within APEX_NEAR_PEAK of the peak wins, which puts the gate at the start of
  a constant-radius bend instead of nowhere."""
  n = min(len(xs), len(lat_accs))
  if n == 0:
    return None
  xs = np.asarray(xs[:n], dtype=float)
  lat = np.asarray(lat_accs[:n], dtype=float)
  peak = float(lat.max())
  if peak <= 0.0:
    return None
  candidates = np.flatnonzero((xs >= min_x) & (xs <= max_x) & (lat >= APEX_NEAR_PEAK * peak))
  if candidates.size == 0:
    return None
  return int(candidates[np.argmax(lat[candidates])])


def curve_direction(xs, ys, idx: int) -> int:
  """+1 when the path bends right at idx, -1 when it bends left (ys is right positive)."""
  n = min(len(xs), len(ys))
  lo, hi = max(idx - 1, 0), min(idx + 1, n - 1)
  if hi - lo < 2:
    return 1 if ys[hi] >= ys[lo] else -1
  slope_before = (ys[idx] - ys[lo]) / max(xs[idx] - xs[lo], 1e-3)
  slope_after = (ys[hi] - ys[idx]) / max(xs[hi] - xs[idx], 1e-3)
  return 1 if slope_after >= slope_before else -1


def curvature_speed(yaw_rate: float, v: float, lat_acc: float = ADVISORY_LAT_ACC) -> float:
  """Speed (m/s) at which this bend would pull `lat_acc` m/s²: sqrt(lat_acc / curvature), curvature = yaw rate / v."""
  curvature = abs(yaw_rate) / max(v, 0.1)
  return math.sqrt(lat_acc / curvature) if curvature > 1e-6 else math.inf


def advisory_speed(vision_active: bool, vision_v_target: float, yaw_rate: float, v: float) -> float:
  """Advisory speed (m/s): the smart cruise control vision target while that controller is steering the speed, otherwise
  the speed the bend itself calls for."""
  if vision_active and vision_v_target > 0.5:
    return float(vision_v_target)
  return curvature_speed(yaw_rate, v)


def display_speed(v: float, is_metric: bool, step: int = SPEED_STEP) -> int | None:
  """m/s to mph or km/h rounded to the nearest step, never below one step. None when there is no finite speed."""
  if not math.isfinite(v):
    return None
  value = v * (CV.MS_TO_KPH if is_metric else CV.MS_TO_MPH)
  return max(step, int(math.floor(value / step + 0.5)) * step)


# ---- drawing helpers ----

def _fill_convex(points: list[tuple[float, float]], color: rl.Color) -> None:
  """raylib only fills fans wound counter-clockwise on screen, so fix the winding."""
  area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1], strict=True))
  pts = points if area < 0 else points[::-1]
  rl.draw_triangle_fan(pts, len(pts), color)


def _lerp_pt(a, b, t: float) -> tuple[float, float]:
  return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def _line(a, b, thickness: float, color: rl.Color) -> None:
  rl.draw_line_ex(rl.Vector2(*a), rl.Vector2(*b), thickness, color)


class CurveHologram:
  """Rails along the planned path colored by predicted lateral acceleration, and an advisory speed gate at the apex."""

  def __init__(self):
    dt = 1 / gui_app.target_fps
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi_bold = gui_app.font(FontWeight.SEMI_BOLD)
    self._rail_alpha = FirstOrderFilter(0.0, 0.25, dt)
    self._gate_alpha = FirstOrderFilter(0.0, 0.2, dt)
    self._apex_x = FirstOrderFilter(0.0, 0.12, dt, initialized=False)
    self._on = False
    self._has_gate = False
    self._xs = np.empty(0)
    self._lat = np.empty(0)
    self._severity = np.empty(0)
    self._apex_lat = 0.0
    self._direction = 1
    self._speed: int | None = None
    self._is_metric = False

  def _reset(self) -> None:
    self._rail_alpha.x = 0.0
    self._gate_alpha.x = 0.0
    self._apex_x.initialized = False
    self._on = False
    self._has_gate = False

  def update(self) -> None:
    if not ui_state.curve_hologram:
      self._reset()
      return

    sm = ui_state.sm
    if sm.recv_frame['modelV2'] < ui_state.started_frame:
      self._on = False
      self._has_gate = False
    else:
      model = sm['modelV2']
      xs = np.asarray(model.position.x, dtype=float)
      yaw = np.asarray(model.orientationRate.z, dtype=float)
      vel = np.asarray(model.velocity.x, dtype=float)
      lat = lateral_accels(yaw, vel)
      n = min(len(lat), len(xs))
      peak = float(lat[:n].max()) if n else 0.0
      self._on = rails_active(peak, self._on)
      idx = find_apex(xs[:n], lat[:n]) if self._on else None

      if self._on:
        self._xs, self._lat = xs[:n], lat[:n]
        self._severity = rail_severity(self._lat)
      if idx is None:
        self._has_gate = False
      else:
        if not self._has_gate:
          self._apex_x.initialized = False
        self._has_gate = True
        self._apex_x.update(float(xs[idx]))
        self._apex_lat = float(lat[idx])
        self._direction = curve_direction(xs[:n], np.asarray(model.position.y, dtype=float), idx)
        vision = sm['longitudinalPlanSP'].smartCruiseControl.vision
        self._is_metric = ui_state.is_metric
        self._speed = display_speed(advisory_speed(vision.active, vision.vTarget, float(yaw[idx]), float(vel[idx])), self._is_metric)

    self._rail_alpha.update(1.0 if self._on else 0.0)
    self._gate_alpha.update(1.0 if self._has_gate and self._speed is not None else 0.0)

  def render(self, rect: rl.Rectangle, projector: RoadProjector) -> None:
    if not ui_state.curve_hologram or self._rail_alpha.x < 0.01 or not projector.ready or self._xs.size < 2:
      return

    self._draw_rails(rect, projector, self._rail_alpha.x)
    if self._gate_alpha.x >= 0.01:
      # a gate close by is huge on screen, so it eases in over the first stretch instead of filling the view
      near = 0.35 + 0.65 * float(np.clip((self._apex_x.x - APEX_MIN_X) / 15.0, 0.0, 1.0))
      self._draw_gate(rect, projector, self._gate_alpha.x * self._rail_alpha.x * near)

  # ---- rails ----

  def _gradient_stops(self, rect: rl.Rectangle, projector: RoadProjector, x0: float, x1: float) -> list[tuple[float, tuple, float]]:
    """(screen-space stop, rgb, alpha fraction) along the path. Spaced evenly in 1/x, which is roughly even on screen."""
    stops: list[tuple[float, tuple, float]] = []
    for x in 1.0 / np.linspace(1.0 / max(x0, STOP_X_NEAR), 1.0 / x1, GRADIENT_STOPS):
      p = projector.to_screen(float(x), float(projector.path_y(x)), 0.0)
      if p is None:
        continue
      t = 1.0 - (p[1] - rect.y) / rect.height  # same convention as the model renderer: 0 at the bottom, 1 at the top
      if stops and t <= stops[-1][0]:
        continue
      severity = float(np.interp(x, self._xs, self._severity))
      stops.append((t, lat_acc_color(severity), float(np.interp(x, RAIL_ALPHA_X, RAIL_ALPHA))))
    return stops

  @staticmethod
  def _make_gradient(stops, alpha: float) -> Gradient:
    colors = [rl.Color(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(min(a * alpha, 1.0) * 255)) for _, rgb, a in stops]  # rgba() is slow in a loop
    return Gradient(start=(0.0, 1.0), end=(0.0, 0.0), colors=colors, stops=[t for t, _, _ in stops])

  def _draw_rails(self, rect: rl.Rectangle, projector: RoadProjector, alpha: float) -> None:
    x0, x1 = RAIL_X_NEAR, min(projector.path_length, RAIL_X_FAR)
    if x1 - x0 < 10.0:
      return

    stops = self._gradient_stops(rect, projector, x0, x1)
    if not stops:
      return
    passes = [(half_width, self._make_gradient(stops, glow_alpha * alpha)) for half_width, glow_alpha in GLOW_PASSES]
    passes.append((RAIL_HALF_WIDTH, self._make_gradient(stops, alpha)))

    xs = x0 * (x1 / x0) ** np.linspace(0.0, 1.0, RAIL_SAMPLES)  # denser near the car, where the road is big on screen
    ys = projector.path_y(xs)
    for half_width, gradient in passes:
      for side in (-1.0, 1.0):
        poly = projector.ribbon(xs, ys + side * RAIL_OFFSET, half_width)
        if poly.shape[0] > 2:
          draw_polygon(rect, poly, gradient=gradient)

    self._draw_beads(projector, x0, min(x1, float(self._apex_x.x)) if self._has_gate else x1, alpha)

  def _draw_beads(self, projector: RoadProjector, x0: float, x1: float, alpha: float) -> None:
    """Bright beads with fading streaks running down the rails toward the gate."""
    x0 = max(x0, STOP_X_NEAR + 1.0)
    if x1 - x0 < 8.0:
      return
    phase = (rl.get_time() / BEAD_PERIOD) % 1.0
    for k in range(BEADS_PER_RAIL):
      s = (phase + k / BEADS_PER_RAIL) % 1.0
      envelope = math.sin(math.pi * s)
      for side in (-1.0, 1.0):
        pts = []
        for j in range(BEAD_TAIL):
          sj = s - j * BEAD_TAIL_STEP
          if sj <= 0.0:
            break
          x = 1.0 / (1.0 / x0 + (1.0 / x1 - 1.0 / x0) * sj)  # even speed on screen
          p = projector.to_screen(x, float(projector.path_y(x)) + side * RAIL_OFFSET, 0.0)
          if p is None:
            break
          pts.append((x, p))
        if not pts:
          continue
        x_head, head = pts[0]
        rgb = lat_acc_color(float(np.interp(x_head, self._xs, self._severity)))
        r = float(np.clip(95.0 / x_head, 1.8, 6.5))
        for j in range(len(pts) - 1):
          fade = alpha * envelope * (1.0 - (j + 1) / BEAD_TAIL)
          _line(pts[j][1], pts[j + 1][1], r * 1.5 * (1.0 - j / BEAD_TAIL), rgba(WHITE, 0.8 * fade))
        rl.draw_circle_v(rl.Vector2(*head), r * 2.8, rgba(rgb, 0.22 * alpha * envelope))
        rl.draw_circle_v(rl.Vector2(*head), r, rgba(WHITE, 0.95 * alpha * envelope))

  # ---- gate ----

  def _draw_gate(self, rect: rl.Rectangle, projector: RoadProjector, alpha: float) -> None:
    ax = float(self._apex_x.x)
    y_c = float(projector.path_y(ax))
    theta = math.atan((float(projector.path_y(ax + 2.0)) - float(projector.path_y(max(ax - 2.0, 0.1)))) / 4.0)
    nx, ny = -math.sin(theta), math.cos(theta)  # right-hand normal of the path
    left = (ax - GATE_HALF_WIDTH * nx, y_c - GATE_HALF_WIDTH * ny)
    right = (ax + GATE_HALF_WIDTH * nx, y_c + GATE_HALF_WIDTH * ny)
    projected = [projector.to_screen(px, py, h) for px, py in (left, right) for h in (0.0, GATE_HEIGHT)]
    corners = [c for c in projected if c is not None]
    if len(corners) != len(projected):
      return
    (blx, bly), (tlx, tly), (brx, bry), (trx, try_) = corners
    bl, tl, br, tr = (blx, bly), (tlx, tly), (brx, bry), (trx, try_)

    rgb = lat_acc_color(self._apex_lat)
    px_per_m = math.hypot(brx - blx, bry - bly) / (2 * GATE_HALF_WIDTH)
    th = float(np.clip(0.09 * px_per_m, 2.5, 9.0))
    pulse = 0.85 + 0.15 * math.sin(rl.get_time() * 2 * math.pi / 1.4)

    # glassy curtain between the posts, clear at the road and brightest at the bar
    glass = tuple(0.55 * c + 0.45 * 255 for c in rgb)
    slices = 6
    for i in range(slices):
      a, b = i / slices, (i + 1) / slices
      quad = [_lerp_pt(bl, tl, a), _lerp_pt(br, tr, a), _lerp_pt(br, tr, b), _lerp_pt(bl, tl, b)]
      _fill_convex(quad, rgba(glass, (0.015 + 0.02 * (i + 1) ** 1.2) * alpha))
    _line(bl, br, th * 0.7, rgba(rgb, 0.55 * alpha))  # footing on the road

    for a, b in ((bl, tl), (br, tr)):
      _line(a, b, th * 2.8, rgba(rgb, 0.2 * alpha * pulse))
    _line(tl, tr, th * 3.2, rgba(rgb, 0.2 * alpha * pulse))
    for a, b in ((bl, tl), (br, tr)):
      _line(a, b, th, rgba(WHITE, 0.95 * alpha))
    _line(tl, tr, th * 1.3, rgba(WHITE, 0.95 * alpha))
    for a, b in ((bl, tl), (br, tr)):
      _line(a, b, th * 0.55, rgba(rgb, alpha))
    _line(tl, tr, th * 0.7, rgba(rgb, alpha))
    for p in (tl, tr):
      rl.draw_circle_v(rl.Vector2(*p), th * 0.85, rgba(WHITE, alpha))
    for p in (bl, br):
      rl.draw_circle_v(rl.Vector2(*p), th * 2.4, rgba(rgb, 0.3 * alpha * pulse))
      rl.draw_circle_v(rl.Vector2(*p), th * 1.1, rgba(WHITE, alpha))

    scale = float(np.clip(0.55 + px_per_m / 80.0, 0.8, 1.3))
    self._draw_label(rect, (tl[0] + tr[0]) / 2, min(tl[1], tr[1]) - th * 1.5, scale, rgb, alpha, pulse)

  def _draw_label(self, rect: rl.Rectangle, cx: float, bottom: float, scale: float, rgb, alpha: float, pulse: float) -> None:
    speed = str(self._speed)
    unit = tr("km/h") if self._is_metric else tr("mph")
    num_size, unit_size = 34.0 * scale, 15.0 * scale
    arrow_w, arrow_h = 15.0 * scale, 24.0 * scale
    pad_x, pad_y, gap, stem = 14.0 * scale, 6.0 * scale, 10.0 * scale, 10.0 * scale

    text_sz = measure_value_with_unit(self._font_bold, speed, num_size, self._font_semi_bold, unit, unit_size, gap=4.0 * scale)
    w = pad_x * 2 + arrow_w + gap + text_sz.x
    h = text_sz.y * 0.92 + pad_y * 2
    x = float(np.clip(cx - w / 2, rect.x + 10, rect.x + rect.width - 10 - w))
    y = max(bottom - stem - h, rect.y + 10)

    _line((cx, y + h), (cx, bottom), 2.5 * scale, rgba(rgb, 0.9 * alpha))
    radius = h / 2
    roundness = min(1.0, 2 * radius / min(w, h))
    pill = rl.Rectangle(x, y, w, h)
    rl.draw_rectangle_rounded(rl.Rectangle(x + 2, y + 4, w, h), roundness, 14, rgba((0, 0, 0), 0.3 * alpha))
    rl.draw_rectangle_rounded(pill, roundness, 14, rgba(INK, 0.82 * alpha))
    rl.draw_rectangle_rounded_lines_ex(pill, roundness, 14, 3.0 * scale, rgba(rgb, (0.8 + 0.2 * pulse) * alpha))

    ax = x + pad_x
    cy = y + h / 2
    if self._direction > 0:
      draw_triangle((ax, cy - arrow_h / 2), (ax + arrow_w, cy), (ax, cy + arrow_h / 2), rgba(rgb, alpha))
    else:
      draw_triangle((ax + arrow_w, cy - arrow_h / 2), (ax, cy), (ax + arrow_w, cy + arrow_h / 2), rgba(rgb, alpha))
    draw_value_with_unit(self._font_bold, speed, num_size, self._font_semi_bold, unit, unit_size, ax + arrow_w + gap, y + pad_y * 0.9,
                         rgba(WHITE, alpha), rgba(WHITE, 0.7 * alpha), gap=4.0 * scale)
