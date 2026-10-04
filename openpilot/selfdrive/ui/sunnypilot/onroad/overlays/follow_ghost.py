"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

import numpy as np
import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import INK, WHITE, draw_text, rgba
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import STOP_DISTANCE, get_t_follow
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.projection import RoadProjector
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

# Follow ghost: a car-sized footprint on the road where the longitudinal planner wants your front bumper to be.
# You drive up to it as the gap to the lead closes, and it fades away when you are there.

COMFORT_BRAKE = 2.5       # m/s², mirrors long_mpc.COMFORT_BRAKE (not imported, that module pulls in the acados solver)
GHOST_LENGTH = 4.5        # m
GHOST_WIDTH = 1.8         # m
GHOST_CORNER = 0.5        # m
GHOST_SKIRT = 0.38        # m, height of the translucent wall around the footprint that gives it some presence
GHOST_MIN_X = 4.0         # m, the ghost is gone when its front is this close
GHOST_FULL_X = 6.0        # m, and fully visible from here
GHOST_FAR_X = (110.0, 130.0)  # m, it fades out over this range, beyond which it is too small to mean anything
MIN_REAR_X = 2.5          # m, the rear edge is held off the car so the footprint can always be projected
SNAP_DISTANCE = 5.0       # m, a ghost that jumps this far (and 25% of dRel) is for a new lead, so skip the smoothing

CYAN = (88, 214, 255)
ICE = (226, 248, 255)
SHIMMER_PERIOD = 3.2      # s
SHIMMER_SWEEP = 0.7       # fraction of the period spent sweeping rear to front
SHIMMER_LENGTH = 1.0      # m


# ---- pure math ----

def desired_gap(v_ego: float, v_lead: float, t_follow: float) -> float:
  """Gap (m) from the lead that the longitudinal MPC settles at, front bumper to the lead's rear. Safe obstacle distance
  for our speed minus the distance the lead needs to stop at comfortable braking."""
  v_ego, v_lead = max(v_ego, 0.0), max(v_lead, 0.0)
  safe_distance = v_ego ** 2 / (2 * COMFORT_BRAKE) + t_follow * v_ego + STOP_DISTANCE
  return safe_distance - v_lead ** 2 / (2 * COMFORT_BRAKE)


def ghost_distance(d_rel: float, v_ego: float, v_lead: float, t_follow: float) -> float:
  """Distance ahead (m) of our front bumper to where it should be: dRel - desired gap. A lead much faster than us would put
  that beyond the lead, so the gap is never taken below the stop distance."""
  return d_rel - max(desired_gap(v_ego, v_lead, t_follow), STOP_DISTANCE)


def ghost_visibility(x_ghost: float) -> float:
  """0 when the ghost is within GHOST_MIN_X of the bumper (we are there), rising to 1 at GHOST_FULL_X, and fading out
  again when it is too far away to read."""
  near = np.clip((x_ghost - GHOST_MIN_X) / (GHOST_FULL_X - GHOST_MIN_X), 0.0, 1.0)
  far = np.clip((GHOST_FAR_X[1] - x_ghost) / (GHOST_FAR_X[1] - GHOST_FAR_X[0]), 0.0, 1.0)
  return float(near * far)


def footprint_outline(length: float, width: float, radius: float, arc_segments: int = 3) -> list[tuple[float, float]]:
  """Rounded rectangle centered on the origin as (u, v): u forward, v to the right. Convex, in order around the edge."""
  radius = min(radius, length / 2, width / 2)
  pts = []
  for cu, cv, start in ((length / 2 - radius, width / 2 - radius, 0.0), (-length / 2 + radius, width / 2 - radius, 90.0),
                        (-length / 2 + radius, -width / 2 + radius, 180.0), (length / 2 - radius, -width / 2 + radius, 270.0)):
    for k in range(arc_segments + 1):
      a = math.radians(start + 90.0 * k / arc_segments)
      pts.append((cu + radius * math.cos(a), cv + radius * math.sin(a)))
  return pts


def place(points: list[tuple[float, float]], x: float, y: float, heading: float) -> list[tuple[float, float]]:
  """Car-frame (x forward, y right) positions of footprint (u, v) points when the footprint is centered on (x, y) and its
  forward axis makes `heading` radians with x, turning toward the right."""
  c, s = math.cos(heading), math.sin(heading)
  return [(x + u * c - v * s, y + u * s + v * c) for u, v in points]


# ---- drawing helpers ----

def _fill_convex(points: list[tuple[float, float]], color: rl.Color) -> None:
  """raylib only fills fans wound counter-clockwise on screen, so fix the winding."""
  area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1], strict=True))
  pts = points if area < 0 else points[::-1]
  rl.draw_triangle_fan(pts, len(pts), color)


def _outline(points: list[tuple[float, float]], thickness: float, color: rl.Color, round_joints: bool = True) -> None:
  """Closed polyline, with round joints unless it is translucent enough that the overlap would show."""
  for a, b in zip(points, points[1:] + points[:1], strict=True):
    _line(a, b, thickness, color)
    if round_joints:
      rl.draw_circle_v(rl.Vector2(*b), thickness / 2, color)


def _line(a, b, thickness: float, color: rl.Color) -> None:
  rl.draw_line_ex(rl.Vector2(*a), rl.Vector2(*b), thickness, color)


class FollowGhost:
  """Footprint of a car on the road where your front bumper should be to hold the following distance."""

  def __init__(self):
    dt = 1 / gui_app.target_fps
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._x = FirstOrderFilter(0.0, 0.3, dt, initialized=False)
    self._alpha = FirstOrderFilter(0.0, 0.2, dt)
    self._present = False

  def _reset(self) -> None:
    self._alpha.x = 0.0
    self._x.initialized = False
    self._present = False

  def update(self) -> None:
    if not ui_state.follow_ghost:
      self._reset()
      return

    sm = ui_state.sm
    lead = sm['radarState'].leadOne
    present = bool(sm.valid['radarState'] and sm.recv_frame['radarState'] >= ui_state.started_frame and lead.present)

    if present:
      raw = ghost_distance(lead.dRel, sm['carState'].vEgo, lead.vLeadK, get_t_follow(sm['selfdriveState'].personality))
      new_lead = not self._present or (self._x.initialized and abs(raw - self._x.x) > max(SNAP_DISTANCE, 0.25 * lead.dRel))
      if new_lead:
        self._x.initialized = False
      self._x.update(raw)
    self._present = present

    # without a lead the last position is kept while the ghost fades out
    self._alpha.update(ghost_visibility(self._x.x) if present else 0.0)

  def render(self, rect: rl.Rectangle, projector: RoadProjector) -> None:
    if not ui_state.follow_ghost or self._alpha.x < 0.01 or not projector.ready:
      return

    x_front = float(self._x.x)
    length = min(GHOST_LENGTH, x_front - MIN_REAR_X)
    if length < 1.0:
      return
    alpha = self._alpha.x

    x_c = x_front - length / 2
    y_c = float(projector.path_y(x_c))
    heading = math.atan((float(projector.path_y(x_c + 1.0)) - float(projector.path_y(max(x_c - 1.0, 0.1)))) / 2.0)

    def project(local: list[tuple[float, float]], height: float = 0.0) -> list[tuple[float, float]] | None:
      projected = [projector.to_screen(px, py, height) for px, py in place(local, x_c, y_c, heading)]
      pts = [p for p in projected if p is not None]
      return pts if len(pts) == len(projected) else None

    shape = footprint_outline(length, GHOST_WIDTH, GHOST_CORNER)
    outline, rim = project(shape), project(shape, GHOST_SKIRT)
    front = project([(length / 2, -GHOST_WIDTH / 2 + GHOST_CORNER), (length / 2, GHOST_WIDTH / 2 - GHOST_CORNER)])
    if outline is None or rim is None or front is None:
      return

    now = rl.get_time()
    px_per_m = math.hypot(front[1][0] - front[0][0], front[1][1] - front[0][1]) / (GHOST_WIDTH - 2 * GHOST_CORNER)
    th = float(np.clip(0.045 * px_per_m, 2.2, 6.0))

    # soft fill with a sheen that sweeps from the rear to the front now and then
    _fill_convex(outline, rgba(CYAN, 0.15 * alpha))
    phase = (now % SHIMMER_PERIOD) / (SHIMMER_PERIOD * SHIMMER_SWEEP)
    if phase < 1.0:
      u0 = -length / 2 + phase * (length + SHIMMER_LENGTH) - SHIMMER_LENGTH
      u1 = u0 + SHIMMER_LENGTH
      u0, u1 = max(u0, -length / 2), min(u1, length / 2)
      half = GHOST_WIDTH / 2 - 0.08
      band = project([(u0, -half), (u1, -half), (u1, half), (u0, half)]) if u1 - u0 > 0.05 else None
      if band:
        _fill_convex(band, rgba(ICE, 0.2 * math.sin(math.pi * phase) * alpha))

    # a low translucent wall around the footprint, so it still reads from a camera only a meter above the road
    for i in range(len(outline)):
      j = (i + 1) % len(outline)
      _fill_convex([outline[i], outline[j], rim[j], rim[i]], rgba(CYAN, 0.08 * alpha))

    pulse = 0.88 + 0.12 * math.sin(now * 2 * math.pi / 1.6)
    _outline(rim, th * 0.55, rgba(CYAN, 0.7 * alpha), round_joints=False)
    _outline(outline, th * 3.0, rgba(CYAN, 0.16 * alpha * pulse), round_joints=False)
    _outline(outline, th, rgba(ICE, 0.95 * alpha))
    _line(front[0], front[1], th * 1.4, rgba(WHITE, alpha))

    scale = float(np.clip(0.55 + px_per_m / 90.0, 0.8, 1.2))
    self._draw_tag(rect, outline, scale, alpha)

  def _draw_tag(self, rect: rl.Rectangle, outline: list[tuple[float, float]], scale: float, alpha: float) -> None:
    """Tag beside the front of the footprint, on the right unless that runs off the screen. It sits off to the side so it
    doesn't cover the lead or the path."""
    label = tr("TARGET")
    size, spacing = 21.0 * scale, 2.2 * scale
    sz = measure_text_cached(self._font_bold, label, int(size), spacing)
    pad_x, pad_y, stem = 13.0 * scale, 4.0 * scale, 14.0 * scale
    w, h = sz.x + pad_x * 2, sz.y * 0.9 + pad_y * 2

    right = max(p[0] for p in outline)
    left = min(p[0] for p in outline)
    front_y = min(p[1] for p in outline)  # the front edge is the far one
    cy = front_y + 2.0 * scale
    if right + stem + w <= rect.x + rect.width - 10:
      x, anchor = right + stem, right
    else:
      x, anchor = left - stem - w, left
    y = max(cy - h / 2, rect.y + 10)

    _line((anchor, cy), (x if x > anchor else x + w, cy), 2.2 * scale, rgba(CYAN, 0.9 * alpha))
    roundness = min(1.0, 2 * (h / 2) / min(w, h))
    pill = rl.Rectangle(x, y, w, h)
    rl.draw_rectangle_rounded(rl.Rectangle(x + 1.5, y + 3, w, h), roundness, 12, rgba((0, 0, 0), 0.3 * alpha))
    rl.draw_rectangle_rounded(pill, roundness, 12, rgba(INK, 0.8 * alpha))
    rl.draw_rectangle_rounded_lines_ex(pill, roundness, 12, 2.2 * scale, rgba(CYAN, 0.9 * alpha))
    draw_text(self._font_bold, label, size, x + pad_x, y + pad_y * 0.8, rgba(WHITE, alpha), spacing)
