"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import time

import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, WHITE, rgba, draw_text, draw_triangle, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.detection import StopPhase, get_detection_feed
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.projection import RoadProjector
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.stop_indicator import STOP_RGB, WARN_RGB, format_distance, update_feed
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

HALF_WIDTH = 1.75   # m, a lane
THICKNESS = 1.2     # m along the road, painted stop lines are wide so they read in perspective
GLOW = 3.0          # m along the road
MIN_X = 3.0         # m, closer than this the line is under the hood
LOCK_ON = 0.6       # s the line sweeps in from farther out when a stop is first seen
LOCK_ON_FROM = 18.0  # m
LABEL_HEIGHT = 1.4  # m above the road
LABEL_SIZE = 44     # px, the distance is the part you read, so it is as large as the alert text
EDGE_WIDTH = 5.0    # px
STEM_WIDTH = 4.0    # px

BRAKE_DECEL = 2.0         # m/s^2, comfortable braking: the brake point is where it has to start to stop at the line
BRAKE_MARGIN = 2.0        # m short of the line, same as Pace Border aims for
ALREADY_BRAKING = 0.8     # m/s^2, once the car slows this hard the marker has done its job
BRAKE_DASHES = 4
BRAKE_RGB = WARN_RGB
BRAKE_LABEL_SIZE = 36  # px, the text floor


def quad(projector: RoadProjector, x: float, y_c: float, heading: float, half_len: float, half_w: float):
  """Four screen corners of a road-aligned rectangle centered at (x, y_c), or None if any is off view."""
  c, s = math.cos(heading), math.sin(heading)
  corners = []
  for dx, dy in ((-half_len, -half_w), (half_len, -half_w), (half_len, half_w), (-half_len, half_w)):
    p = projector.to_screen(x + dx * c - dy * s, y_c + dx * s + dy * c)
    if p is None:
      return None
    corners.append(p)
  return corners


def brake_point(stop_x: float, v_ego: float, decel: float = BRAKE_DECEL) -> float:
  """Distance (m) to where braking at decel has to start to stop BRAKE_MARGIN short of the line. At or below zero it's too late
  to brake gently."""
  return stop_x - BRAKE_MARGIN - v_ego ** 2 / (2 * decel)


def fill_quad(corners, color: rl.Color) -> None:
  draw_triangle(corners[0], corners[1], corners[2], color)
  draw_triangle(corners[0], corners[2], corners[3], color)


class StopLine:
  """A stop line painted across the lane where the model plans to stop."""

  def __init__(self):
    self._font = gui_app.font(FontWeight.BOLD)
    self._alpha = FirstOrderFilter(0.0, 0.12, 1 / gui_app.target_fps)
    self._x: float | None = None
    self._brake_alpha = FirstOrderFilter(0.0, 0.2, 1 / gui_app.target_fps)
    self._brake_x = 0.0

  def update(self) -> None:
    if not ui_state.stop_detection:
      self._alpha.x = 0.0
      self._x = None
      return
    update_feed()
    stop = get_detection_feed().stop
    visible = stop.phase == StopPhase.APPROACH and stop.stop_x is not None and stop.stop_x > MIN_X
    if visible:
      self._x = stop.stop_x  # already smoothed by the detector, same number as the card
    self._alpha.update(1.0 if visible else 0.0)

    # where gentle braking has to start, until the car is braking or it's too late for gentle
    cs = ui_state.sm['carState']
    brake_x = brake_point(stop.stop_x, cs.vEgo) if visible and stop.stop_x is not None else 0.0
    show = visible and not stop.car_ahead and brake_x > MIN_X and cs.aEgo > -ALREADY_BRAKING
    if show:
      self._brake_x = brake_x
    self._brake_alpha.update(1.0 if show else 0.0)

  def render(self, rect: rl.Rectangle, projector: RoadProjector) -> None:
    alpha = self._alpha.x
    if not ui_state.stop_detection or alpha < 0.01 or self._x is None or not projector.ready:
      return

    stop = get_detection_feed().stop
    x = float(self._x)
    since = time.monotonic() - stop.phase_since if stop.phase == StopPhase.APPROACH else LOCK_ON
    lock = min(1.0, since / LOCK_ON)
    # sweeps in from farther out at the moment of detection, then sits on the stop point
    x_draw = x + (1.0 - lock) ** 2 * LOCK_ON_FROM
    if x_draw <= MIN_X:
      return

    y_c = float(projector.path_y(x_draw))
    heading = math.atan((float(projector.path_y(x_draw + 1.0)) - float(projector.path_y(max(x_draw - 1.0, 0.1)))) / 2.0)

    glow = quad(projector, x_draw, y_c, heading, GLOW / 2, HALF_WIDTH + 0.3)
    bar = quad(projector, x_draw, y_c, heading, THICKNESS / 2, HALF_WIDTH)
    if glow is None or bar is None:
      return

    if self._brake_alpha.x > 0.01 and lock >= 1.0:
      self._draw_brake_point(projector, self._brake_alpha.x * alpha)

    pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 2 * math.pi / 1.4)
    fill_quad(glow, rgba(STOP_RGB, (0.18 + 0.12 * pulse) * alpha))
    fill_quad(bar, rgba(STOP_RGB, 0.9 * alpha * (0.5 + 0.5 * lock)))
    for a, b in ((bar[0], bar[3]), (bar[1], bar[2])):
      rl.draw_line_ex(rl.Vector2(*a), rl.Vector2(*b), EDGE_WIDTH, rgba(WHITE, 0.85 * alpha))

    anchor = projector.to_screen(x_draw, y_c, LABEL_HEIGHT)
    if anchor is None:
      return
    value, unit = format_distance(x, ui_state.is_metric)
    text = tr("STOP {} {}").format(value, unit)
    size = LABEL_SIZE
    sz = measure_text_cached(self._font, text, size, 1)
    w, h = sz.x + 36, sz.y + 14
    px = float(min(max(anchor[0] - w / 2, rect.x + 10), rect.x + rect.width - 10 - w))
    py = float(max(anchor[1] - h, rect.y + 10))
    pill = rl.Rectangle(px, py, w, h)
    rl.draw_rectangle_rounded(pill, roundness(h / 2, w, h), 12, rgba(BLACK, 0.7 * alpha))
    rl.draw_rectangle_rounded_lines_ex(pill, roundness(h / 2, w, h), 12, 4, rgba(STOP_RGB, alpha))
    draw_text(self._font, text, size, px + 18, py + 7, rgba(WHITE, alpha), spacing=1)
    # a short stem from the pill to the line
    base = projector.to_screen(x_draw, y_c)
    if base is not None and base[1] > py + h:
      rl.draw_line_ex(rl.Vector2(base[0], base[1]), rl.Vector2(anchor[0], py + h), STEM_WIDTH, rgba(STOP_RGB, 0.8 * alpha))

  def _draw_brake_point(self, projector: RoadProjector, alpha: float) -> None:
    """A dashed bar across the lane where gentle braking has to begin, and BRAKE beside it."""
    x = float(self._brake_x)
    y_c = float(projector.path_y(x))
    heading = math.atan((float(projector.path_y(x + 1.0)) - float(projector.path_y(max(x - 1.0, 0.1)))) / 2.0)
    dash_half = HALF_WIDTH / (2 * BRAKE_DASHES - 1)
    for k in range(BRAKE_DASHES):
      offset = -HALF_WIDTH + dash_half * (4 * k + 1)
      dash = quad(projector, x - offset * math.sin(heading), y_c + offset * math.cos(heading), heading, 0.3, dash_half)
      if dash is not None:
        fill_quad(dash, rgba(BRAKE_RGB, 0.9 * alpha))
    edge = projector.to_screen(x - HALF_WIDTH * 1.15 * math.sin(heading), y_c + HALF_WIDTH * 1.15 * math.cos(heading))
    if edge is not None:
      draw_text(self._font, tr("BRAKE"), BRAKE_LABEL_SIZE, edge[0] + 8, edge[1] - BRAKE_LABEL_SIZE, rgba(BRAKE_RGB, alpha), spacing=1)
