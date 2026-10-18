"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
from dataclasses import dataclass

import numpy as np
import pyray as rl

from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import INK, STROKE_MIN, TEXT_MIN, WHITE, ZONE_RGB, draw_triangle, \
  clamp, draw_value_with_unit, measure_value_with_unit, rgba
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone
from openpilot.selfdrive.ui.sunnypilot.model_plan import ModelPlan, model_plan
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.projection import RoadProjector, draw_ribbons
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.shader_polygon import Gradient

# Curve hologram: when the model predicts a real bend ahead, two glowing rails run along the planned path, colored by
# the predicted lateral acceleration, and a gate stands over the road at the tightest point with an advisory speed.

SHOW_LAT_ACC = 1.0        # m/s², the rails appear when the largest predicted lateral acceleration ahead exceeds this
HIDE_LAT_ACC = 0.9        # m/s², and go away again below this
ADVISORY_LAT_ACC = 2.0    # m/s², lateral acceleration the advisory speed is computed for (A_LAT_REG_MAX in the vision controller)
APEX_MIN_X = 10.0         # m, a gate closer than this is already behind the camera's useful view
APEX_MAX_X = 120.0        # m, and one farther than this is too small to read
APEX_NEAR_PEAK = 0.97     # fraction of the peak that counts as the top of the bend (a flat bend has no single maximum)
SPEED_STEP = 5            # advisory speeds are rounded to this
TOO_FAST_SHOW = 2.7       # m/s (6 mph) over the advisory speed before the label flashes red
TOO_FAST_HIDE = 1.3       # m/s, and it stops below this
TOO_FAST_PERIOD = 0.7     # s
TOO_FAST_RGB = ZONE_RGB[GapZone.CRITICAL]

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

# the gate's speed label keeps one size at every distance, so it reads at a glance even while the gate is far away
LABEL_NUM_SIZE, LABEL_UNIT_SIZE = 56.0, float(TEXT_MIN)
GATE_LINE = (float(STROKE_MIN), 13.0)   # px post and bar thickness range

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


def lat_acc_colors(lat_accs) -> np.ndarray:
  """Colors (N x 3) for many lateral accelerations at once."""
  lat = np.asarray(lat_accs, dtype=float)
  return np.stack([np.interp(lat, LAT_ACC_BREAKPOINTS, LAT_ACC_COLORS[:, c]) for c in range(3)], axis=-1)


def lat_acc_color(lat_acc: float) -> tuple[float, float, float]:
  r, g, b = (float(c) for c in lat_acc_colors(lat_acc))
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


@dataclass(frozen=True)
class CurveAhead:
  x: float        # m to the tightest point of the bend
  speed: float    # m/s the bend calls for (advisory_speed)
  right: bool     # bends to the right


CURVE_LEAD_IN = 10.0     # m, be at the bend's speed this far before its tightest point
CURVE_AHEAD_RANGE = (APEX_MIN_X, 200.0)  # m, wider than the gate's: an advisory wants to know about a bend sooner
_curve_plan: ModelPlan | None = None
_curve: CurveAhead | None = None


def curve_decel_needed(v_ego: float, curve_speed: float, curve_x: float) -> float:
  """m/s^2 of braking needed to be at the bend's speed CURVE_LEAD_IN before its tightest point."""
  if v_ego <= curve_speed:
    return 0.0
  return (v_ego ** 2 - curve_speed ** 2) / (2 * max(curve_x - CURVE_LEAD_IN, 1.0))


def curve_ahead(sm, started_frame: int) -> CurveAhead | None:
  """The tightest point of the bend ahead and the speed it calls for, worked out once per model frame for everything that asks (the
  drive notifications and Pace Border), whether or not the curve preview is on."""
  global _curve_plan, _curve
  plan = model_plan(sm, started_frame)
  if plan is not _curve_plan:
    _curve_plan, _curve = plan, None
    idx = find_apex(plan.x, plan.lat_acc, *CURVE_AHEAD_RANGE) if plan.size else None
    if idx is not None:
      vision = sm['longitudinalPlanSP'].smartCruiseControl.vision
      speed = advisory_speed(vision.active, vision.vTarget, float(plan.yaw_rate[idx]), float(plan.v[idx]))
      _curve = CurveAhead(float(plan.x[idx]), speed, curve_direction(plan.x, plan.y, idx) > 0)
  return _curve


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
    self._advisory = math.inf    # m/s
    self._too_fast = False
    self._apex_target = 0.0
    self._is_metric = False
    self._plan: ModelPlan | None = None

  def _reset(self) -> None:
    self._rail_alpha.x = 0.0
    self._gate_alpha.x = 0.0
    self._apex_x.initialized = False
    self._on = False
    self._has_gate = False
    self._plan = None

  def update(self) -> None:
    if not ui_state.curve_hologram:
      self._reset()
      return

    sm = ui_state.sm
    plan = model_plan(sm, ui_state.started_frame)
    if plan.size == 0:
      self._on = False
      self._has_gate = False
    elif plan is not self._plan:  # the bend only changes with a new model frame
      self._plan = plan
      xs, lat = plan.x, plan.lat_acc
      peak = float(lat.max())
      self._on = rails_active(peak, self._on)
      idx = find_apex(xs, lat) if self._on else None

      if self._on:
        self._xs, self._lat = xs, lat
        self._severity = rail_severity(self._lat)
      if idx is None:
        self._has_gate = False
      else:
        if not self._has_gate:
          self._apex_x.initialized = False
        self._has_gate = True
        self._apex_target = float(xs[idx])
        self._apex_lat = float(lat[idx])
        self._direction = curve_direction(xs, plan.y, idx)
        vision = sm['longitudinalPlanSP'].smartCruiseControl.vision
        self._is_metric = ui_state.is_metric
        self._advisory = advisory_speed(vision.active, vision.vTarget, float(plan.yaw_rate[idx]), float(plan.v[idx]))
        self._speed = display_speed(self._advisory, self._is_metric)
    v_ego = sm['carState'].vEgo
    self._too_fast = self._has_gate and v_ego > self._advisory + (TOO_FAST_HIDE if self._too_fast else TOO_FAST_SHOW)
    if self._has_gate:
      self._apex_x.update(self._apex_target)

    self._rail_alpha.update(1.0 if self._on else 0.0)
    self._gate_alpha.update(1.0 if self._has_gate and self._speed is not None else 0.0)

  def render(self, rect: rl.Rectangle, projector: RoadProjector) -> None:
    if not ui_state.curve_hologram or self._rail_alpha.x < 0.01 or not projector.ready or self._xs.size < 2:
      return

    self._draw_rails(rect, projector, self._rail_alpha.x)
    if self._gate_alpha.x >= 0.01:
      # a gate close by is huge on screen, so it eases in over the first stretch instead of filling the view
      near = 0.35 + 0.65 * clamp((self._apex_x.x - APEX_MIN_X) / 15.0, 0.0, 1.0)
      self._draw_gate(rect, projector, self._gate_alpha.x * self._rail_alpha.x * near)

  # ---- rails ----

  def _gradient_stops(self, rect: rl.Rectangle, projector: RoadProjector, x0: float, x1: float) -> list[tuple[float, tuple, float]]:
    """(screen-space stop, rgb, alpha fraction) along the path. Spaced evenly in 1/x, which is roughly even on screen."""
    xs = 1.0 / np.linspace(1.0 / max(x0, STOP_X_NEAR), 1.0 / x1, GRADIENT_STOPS)
    screen, ok = projector.to_screen_many(xs, projector.path_y(xs))
    ts = 1.0 - (screen[:, 1] - rect.y) / rect.height  # same convention as the model renderer: 0 at the bottom, 1 at the top
    colors = lat_acc_colors(np.interp(xs, self._xs, self._severity))
    alphas = np.interp(xs, RAIL_ALPHA_X, RAIL_ALPHA)
    stops: list[tuple[float, tuple, float]] = []
    for i in np.flatnonzero(ok):
      t = float(ts[i])
      if stops and t <= stops[-1][0]:
        continue
      stops.append((t, tuple(colors[i]), float(alphas[i])))
    return stops

  @staticmethod
  def _make_gradient(stops, alpha: float) -> Gradient:
    return Gradient(start=(0.0, 1.0), end=(0.0, 0.0), colors=[rgba(rgb, a * alpha) for _, rgb, a in stops], stops=[t for t, _, _ in stops])

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
    rails = [ys - RAIL_OFFSET, ys + RAIL_OFFSET]
    polygons = projector.ribbons(xs, rails * len(passes), [hw for hw, _ in passes for _ in rails])
    for i, (_, gradient) in enumerate(passes):
      draw_ribbons(rect, polygons[2 * i:2 * i + 2], gradient)

    self._draw_beads(projector, x0, min(x1, float(self._apex_x.x)) if self._has_gate else x1, alpha)

  def _draw_beads(self, projector: RoadProjector, x0: float, x1: float, alpha: float) -> None:
    """Bright beads with fading streaks running down the rails toward the gate."""
    x0 = max(x0, STOP_X_NEAR + 1.0)
    if x1 - x0 < 8.0:
      return
    phase = (rl.get_time() / BEAD_PERIOD) % 1.0
    runs = (phase + np.arange(BEADS_PER_RAIL) / BEADS_PER_RAIL) % 1.0
    # every point of every bead and its streak (bead, tail point) projected in one go, for each rail
    sj = runs[:, None] - np.arange(BEAD_TAIL)[None, :] * BEAD_TAIL_STEP
    xs = 1.0 / (1.0 / x0 + (1.0 / x1 - 1.0 / x0) * np.maximum(sj, 1e-3))  # even speed on screen
    path_y = projector.path_y(xs)
    head_rgb = lat_acc_colors(np.interp(xs[:, 0], self._xs, self._severity))
    for side in (-1.0, 1.0):
      screen, ok = projector.to_screen_many(xs, path_y + side * RAIL_OFFSET)
      ok &= sj > 0.0
      for k in range(BEADS_PER_RAIL):
        count = int(np.argmin(ok[k])) if not ok[k].all() else BEAD_TAIL  # a streak ends at its first point out of view
        if count == 0:
          continue
        envelope = math.sin(math.pi * runs[k])
        pts = screen[k, :count]
        r = clamp(140.0 / xs[k, 0], 4.0, 9.0)
        for j in range(count - 1):
          fade = alpha * envelope * (1.0 - (j + 1) / BEAD_TAIL)
          _line(pts[j], pts[j + 1], r * 1.5 * (1.0 - j / BEAD_TAIL), rgba(WHITE, 0.8 * fade))
        head = rl.Vector2(float(pts[0, 0]), float(pts[0, 1]))
        rl.draw_circle_v(head, r * 2.8, rgba(head_rgb[k], 0.22 * alpha * envelope))
        rl.draw_circle_v(head, r, rgba(WHITE, 0.95 * alpha * envelope))

  # ---- gate ----

  def _draw_gate(self, rect: rl.Rectangle, projector: RoadProjector, alpha: float) -> None:
    ax = float(self._apex_x.x)
    y_c = float(projector.path_y(ax))
    theta = math.atan((float(projector.path_y(ax + 2.0)) - float(projector.path_y(max(ax - 2.0, 0.1)))) / 4.0)
    nx, ny = -math.sin(theta), math.cos(theta)  # right-hand normal of the path
    left = (ax - GATE_HALF_WIDTH * nx, y_c - GATE_HALF_WIDTH * ny)
    right = (ax + GATE_HALF_WIDTH * nx, y_c + GATE_HALF_WIDTH * ny)
    screen, ok = projector.to_screen_many([left[0], left[0], right[0], right[0]], [left[1], left[1], right[1], right[1]],
                                          [0.0, GATE_HEIGHT, 0.0, GATE_HEIGHT])
    if not ok.all():
      return
    (blx, bly), (tlx, tly), (brx, bry), (trx, try_) = screen.tolist()
    bl, tl, br, tr = (blx, bly), (tlx, tly), (brx, bry), (trx, try_)

    rgb = lat_acc_color(self._apex_lat)
    px_per_m = math.hypot(brx - blx, bry - bly) / (2 * GATE_HALF_WIDTH)
    th = clamp(0.1 * px_per_m, *GATE_LINE)
    pulse = 0.85 + 0.15 * math.sin(rl.get_time() * 2 * math.pi / 1.4)

    # glassy curtain between the posts, clear at the road and brightest at the bar
    glass = tuple(0.55 * c + 0.45 * 255 for c in rgb)
    slices = 6
    for i in range(slices):
      lo, hi = i / slices, (i + 1) / slices
      quad = [_lerp_pt(bl, tl, lo), _lerp_pt(br, tr, lo), _lerp_pt(br, tr, hi), _lerp_pt(bl, tl, hi)]
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

    self._draw_label(rect, (tl[0] + tr[0]) / 2, min(tl[1], tr[1]) - th * 1.5, rgb, alpha, pulse)

  def _draw_label(self, rect: rl.Rectangle, cx: float, bottom: float, rgb, alpha: float, pulse: float) -> None:
    speed = str(self._speed)
    unit = tr("km/h") if self._is_metric else tr("mph")
    num_size, unit_size = LABEL_NUM_SIZE, LABEL_UNIT_SIZE
    arrow_w, arrow_h = 24.0, 38.0
    pad_x, pad_y, gap, stem = 20.0, 8.0, 14.0, 14.0

    text_sz = measure_value_with_unit(self._font_bold, speed, num_size, self._font_semi_bold, unit, unit_size, gap=6.0)
    w = pad_x * 2 + arrow_w + gap + text_sz.x
    h = text_sz.y * 0.92 + pad_y * 2
    x = clamp(cx - w / 2, rect.x + 10, rect.x + rect.width - 10 - w)
    y = max(bottom - stem - h, rect.y + 10)

    _line((cx, y + h), (cx, bottom), 5.0, rgba(rgb, 0.9 * alpha))
    radius = h / 2
    roundness = min(1.0, 2 * radius / min(w, h))
    pill = rl.Rectangle(x, y, w, h)
    rl.draw_rectangle_rounded(rl.Rectangle(x + 2, y + 4, w, h), roundness, 14, rgba((0, 0, 0), 0.3 * alpha))
    rl.draw_rectangle_rounded(pill, roundness, 14, rgba(INK, 0.82 * alpha))
    if self._too_fast:
      # faster than the bend calls for: the label flashes red, in step with the gate
      flash = 0.5 + 0.5 * math.sin(rl.get_time() * 2 * math.pi / TOO_FAST_PERIOD)
      rl.draw_rectangle_rounded(pill, roundness, 14, rgba(TOO_FAST_RGB, (0.25 + 0.35 * flash) * alpha))
      rl.draw_rectangle_rounded_lines_ex(pill, roundness, 14, 7.0, rgba(TOO_FAST_RGB, alpha))
    else:
      rl.draw_rectangle_rounded_lines_ex(pill, roundness, 14, 5.0, rgba(rgb, (0.8 + 0.2 * pulse) * alpha))

    ax = x + pad_x
    cy = y + h / 2
    if self._direction > 0:
      draw_triangle((ax, cy - arrow_h / 2), (ax + arrow_w, cy), (ax, cy + arrow_h / 2), rgba(rgb, alpha))
    else:
      draw_triangle((ax + arrow_w, cy - arrow_h / 2), (ax, cy), (ax + arrow_w, cy + arrow_h / 2), rgba(rgb, alpha))
    draw_value_with_unit(self._font_bold, speed, num_size, self._font_semi_bold, unit, unit_size, ax + arrow_w + gap, y + pad_y * 0.9,
                         rgba(WHITE, alpha), rgba(WHITE, 0.7 * alpha), gap=6.0)
