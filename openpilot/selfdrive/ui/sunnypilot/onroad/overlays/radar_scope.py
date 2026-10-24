"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Radar scope: a top-down minimap of the model's lane lines and every radar track, in the left column of the HUD.
"""
import math
from dataclasses import dataclass

import numpy as np
import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.developer_ui import get_bottom_dev_ui_offset
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, DOT_MIN, INK, WHITE, ZONE_RGB, rgba, draw_triangle, \
  roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone, FT_PER_M
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.light_view import camera_people
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_tracks import CLASS_RGB, RadarFeed, RadarTrack, TrackClass, display_zone, \
  shared_radar_feed
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app

PANEL_W, PANEL_H = 400.0, 410.0   # leaves room under it for the driver flag chips above the driver monitoring icon
PANEL_X, PANEL_Y = 40.0, 290.0     # from the HUD rect, clear of the MAX box and speed limit sign above
ROCKET_FUEL_SHIFT = 40.0           # the rocket fuel bar takes the left edge
PAD = 16.0
TOP_PAD = 20.0
EGO_W, EGO_H = 30.0, 56.0
BOTTOM_PAD = 16.0

MAX_RANGE = 120.0                  # m at the top of the plot
MAX_LATERAL = 12.0                 # m either side
RINGS_METRIC = (25.0, 50.0, 100.0)
RINGS_IMPERIAL = (100.0, 200.0, 300.0)   # ft
LANE_SAMPLES = 30
LANE_DISTANCES = np.linspace(0.0, 1.0, LANE_SAMPLES) ** 2 * MAX_RANGE   # even steps on the compressed range axis

VELOCITY_TICK = 6.0                # px per m/s of closing or opening speed
MAX_TICK = 44.0
MIN_TICK_SPEED = 0.5               # m/s
PING_PERIOD = 3.6                  # s

# sized to read at a glance: no text at all, the dots and lines are thick enough to see from the corner of an eye
DOT_STATIONARY = float(DOT_MIN)
DOT_MOVING = 12.0
DOT_THREAT = 15.0
LANE_WIDTH_OURS, LANE_WIDTH_OTHER = 7.0, 4.5
RING_WIDTH = 3.0
TICK_WIDTH = 6.0
LEAD_RING_RADIUS, LEAD_RING_WIDTH = 23.0, 6.0
BLINDSPOT_LATERAL = (2.2, 5.0)     # m from our centerline, the lane beside us
BLINDSPOT_AHEAD = 4.0              # m past the bumper the lit zone reaches
BRAKE_RGB = (255, 40, 30)
PERSON_RGB = ZONE_RGB[GapZone.CAUTION]
KIND_CYCLIST = 5                   # SceneDetectionSP.VehicleKind


FADE_RC = 0.15                     # s, a dot or figure fading in when found and out when lost
FLAG_RC = 0.2                      # s, brake lamps and the merging chevron coming and going


class Fade:
  """One thing on the scope by id: its last position and how visible it and its flags are, so it fades instead of popping."""

  def __init__(self, item, dt: float):
    self.item = item
    self.alpha = FirstOrderFilter(0.0, FADE_RC, dt)
    self.braking = FirstOrderFilter(0.0, FLAG_RC, dt)
    self.merging = FirstOrderFilter(0.0, FLAG_RC, dt)


def update_fades(fades: dict, items: dict, dt: float, flags=lambda item: (False, False)) -> dict:
  """Brings each id's fade toward shown (present in items) or gone, and drops the ones that have faded out."""
  out = {}
  for key, item in items.items():
    f = fades.get(key) or Fade(item, dt)
    f.item = item
    f.alpha.update(1.0)
    braking, merging = flags(item)
    f.braking.update(1.0 if braking else 0.0)
    f.merging.update(1.0 if merging else 0.0)
    out[key] = f
  for key, f in fades.items():
    if key not in out and f.alpha.update(0.0) > 0.02:
      f.braking.update(0.0)
      f.merging.update(0.0)
      out[key] = f
  return out


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
  top = panel.y + TOP_PAD
  return ScopeFrame(panel.x + panel.width / 2, origin_y, (panel.width - 2 * PAD) / (2 * MAX_LATERAL), origin_y - top)


def ring_radii(is_metric: bool) -> list[float]:
  """Range of each range ring in m, at round numbers in the display unit."""
  if is_metric:
    return list(RINGS_METRIC)
  return [r / FT_PER_M for r in RINGS_IMPERIAL]


def ring_points(radius: float, step: float = 1.5) -> list[tuple[float, float]]:
  """(d, y_right) along a circle of this radius around the bumper, clipped to the scope's lateral extent."""
  points = []
  for i in range(int(2 * MAX_LATERAL / step) + 1):
    lateral = -MAX_LATERAL + i * step
    if abs(lateral) < radius:
      points.append((math.sqrt(radius * radius - lateral * lateral), lateral))
  return points


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
  """[start, end) of every run of at least two consecutive True values."""
  edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
  return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2], strict=True) if b - a >= 2]


def draw_polyline(points: list, thickness: float, color: rl.Color) -> None:
  """A thick polyline in one draw call. The points can be plain (x, y) pairs, cffi converts them without building Vector2s."""
  if len(points) >= 2:
    rl.draw_spline_linear(points, len(points), thickness, color)


class RadarScope:
  """Top-down minimap: lane lines from the model, range rings, and a dot per radar track.
  Dots are cyan when moving the same way, gray when stationary, hollow pink when oncoming, and severity colored when closing on us.
  Cars the cameras found (Camera Object Detection) are rounded squares in the same colors.
  The lead has a solid ring, the second lead a dashed one, and a tick on each moving dot points the way it's pulling (up is faster than us).
  A car whose brake lights are on has two red lamps at its back, one drifting into our lane a chevron pointing in, and the people the
  cameras see are small amber figures (with a wheel under a cyclist)."""

  def __init__(self):
    dt = self._dt = 1 / gui_app.target_fps
    self._feed = RadarFeed(dt)  # until the first update hands over the shared one
    self._alpha = FirstOrderFilter(0.0, 0.2, dt)
    self._lane_y: np.ndarray | None = None     # (4, LANE_SAMPLES) lateral position at LANE_DISTANCES, NaN where the model has none
    self._lane_alpha = np.zeros(4)
    self._rings_key: tuple | None = None
    self._blindspots = (FirstOrderFilter(0.0, 0.1, dt), FirstOrderFilter(0.0, 0.1, dt))  # left, right
    self._lanes_model = None
    self._rings: list[tuple[rl.Vector2, rl.Vector2]] = []
    self._track_fades: dict[int, Fade] = {}
    self._people_fades: dict[int, Fade] = {}

  def update(self) -> None:
    if not ui_state.radar_scope:
      self._alpha.x = 0.0
      return
    sm = ui_state.sm
    self._feed = shared_radar_feed(sm, ui_state.started_frame, self._dt)
    cs = sm['carState']
    fresh = sm.recv_frame['carState'] >= ui_state.started_frame
    self._blindspots[0].update(1.0 if fresh and cs.leftBlindspot else 0.0)
    self._blindspots[1].update(1.0 if fresh and cs.rightBlindspot else 0.0)
    self._alpha.update(1.0 if self._feed.available else 0.0)
    self._track_fades = update_fades(self._track_fades, {t.track_id: t for t in self._feed.tracks}, self._dt,
                                     lambda t: (t.braking, t.merging))
    self._people_fades = update_fades(self._people_fades, {p.trackId: p for p in camera_people(sm)}, self._dt)
    if self._alpha.x > 0.01:
      self._update_lanes(sm)

  def _update_lanes(self, sm) -> None:
    if sm.recv_frame['modelV2'] < ui_state.started_frame:
      self._lane_y, self._lanes_model = None, None
      return
    model = sm['modelV2']
    if model is self._lanes_model:  # lane lines only change with a new model frame
      return
    self._lanes_model = model
    self._lane_y = None
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
    self._draw_rings(frame, ui_state.is_metric, now, alpha)
    self._draw_lanes(frame, alpha)
    self._draw_blindspots(frame, now, alpha)
    self._draw_tracks(frame, now, alpha)
    self._draw_ego(frame, alpha)

  def _ring_dashes(self, frame: ScopeFrame, is_metric: bool) -> list[tuple[rl.Vector2, rl.Vector2]]:
    """The dashes of the range rings, which only move when the panel does, so they are built once instead of every frame."""
    key = (frame, is_metric)
    if key != self._rings_key:
      dashes = []
      for radius in ring_radii(is_metric):
        pts = [rl.Vector2(*frame.project(d, y)) for d, y in ring_points(radius)]
        dashes += [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
      self._rings_key, self._rings = key, dashes
    return self._rings

  def _draw_rings(self, frame: ScopeFrame, is_metric: bool, now: float, alpha: float) -> None:
    color = rgba(WHITE, 0.2 * alpha)
    for a, b in self._ring_dashes(frame, is_metric):
      rl.draw_line_ex(a, b, RING_WIDTH, color)

    # a faint ping sweeping outward from the bumper
    phase = (now % PING_PERIOD) / PING_PERIOD
    if phase < 0.85:
      pts = [frame.project(d, y) for d, y in ring_points(phase / 0.85 * MAX_RANGE * 0.98)]
      draw_polyline(pts, 4.0, rgba(WHITE, 0.14 * (1 - phase / 0.85) * alpha))

  def _draw_lanes(self, frame: ScopeFrame, alpha: float) -> None:
    """Each lane line in one call per unbroken run, and our lane's shading as one triangle strip, instead of a call per segment."""
    if self._lane_y is None:
      return
    visible = np.isfinite(self._lane_y) & (np.abs(np.nan_to_num(self._lane_y, nan=np.inf)) <= MAX_LATERAL)
    xs, ys = frame.project_arrays(LANE_DISTANCES[None, :], np.nan_to_num(self._lane_y))
    ys = np.broadcast_to(ys, xs.shape)

    # our lane, shaded
    both = visible[1] & visible[2]
    fill = rgba(CLASS_RGB[TrackClass.MOVING], 0.1 * alpha)
    for lo, hi in runs(both):
      strip = np.empty((2 * (hi - lo), 2))
      strip[0::2, 0], strip[0::2, 1] = xs[1, lo:hi], ys[1, lo:hi]
      strip[1::2, 0], strip[1::2, 1] = xs[2, lo:hi], ys[2, lo:hi]
      pts = strip.tolist()
      rl.draw_triangle_strip(pts, len(pts), fill)

    for i in (0, 3, 1, 2):
      bright = i in (1, 2)
      color = rgba(WHITE, (0.85 if bright else 0.3) * self._lane_alpha[i] * alpha)
      for lo, hi in runs(visible[i]):
        draw_polyline(np.column_stack((xs[i, lo:hi], ys[i, lo:hi])).tolist(), LANE_WIDTH_OURS if bright else LANE_WIDTH_OTHER, color)

  def _draw_blindspots(self, frame: ScopeFrame, now: float, alpha: float) -> None:
    """The car's blind spot monitor: the lane beside us lit from behind the car to a little ahead of the bumper. Most radars that
    give the scope only the car ahead still have blind spot sensors, and this is where those cars are."""
    pulse = 0.5 + 0.5 * math.sin(now * 2 * math.pi / 0.9)
    rgb = ZONE_RGB[GapZone.CLOSE]
    for side, level in zip((-1.0, 1.0), (f.x for f in self._blindspots), strict=True):
      if level < 0.01:
        continue
      inner, outer = frame.cx + side * BLINDSPOT_LATERAL[0] * frame.px_per_m, frame.cx + side * BLINDSPOT_LATERAL[1] * frame.px_per_m
      top = frame.origin_y - float(frame.range_frac(BLINDSPOT_AHEAD)) * frame.height
      zone = rl.Rectangle(min(inner, outer), top, abs(outer - inner), frame.origin_y + EGO_H - top)
      r = roundness(10, zone.width, zone.height)
      rl.draw_rectangle_rounded(zone, r, 8, rgba(rgb, (0.3 + 0.25 * pulse) * level * alpha))
      rl.draw_rectangle_rounded_lines_ex(zone, r, 8, 3, rgba(rgb, 0.9 * level * alpha))

  def _draw_tracks(self, frame: ScopeFrame, now: float, alpha: float) -> None:
    feed = self._feed
    lead_one, lead_two = feed.lead_one, feed.lead_two
    fades = [f for f in self._track_fades.values() if frame.visible(f.item.d_rel, f.item.y_right)]
    # stationary clutter first so moving traffic and threats land on top
    fades.sort(key=lambda f: (f.item.cls != TrackClass.STATIONARY, f.item.ttc < math.inf))
    for f in fades:
      self._draw_dot(frame, f.item, now, alpha * f.alpha.x, f.braking.x, f.merging.x)
    for f in self._people_fades.values():
      p = f.item
      if frame.visible(p.dRel, -p.yRel):
        draw_person(*frame.project(p.dRel, -p.yRel), p.kind == KIND_CYCLIST, alpha * f.alpha.x)

    for lead, ring in ((lead_two, "dashed"), (lead_one, "solid")):
      if lead is not None and frame.visible(lead.d_rel, lead.y_right):
        self._draw_lead_ring(frame, lead, ring, now, alpha)
    # vision-only leads have no dot to ring, so mark where the camera sees them
    for lead, ring in zip(feed.vision_leads, ("solid", "dashed"), strict=True):
      if lead is not None and frame.visible(lead.d_rel, lead.y_right):
        self._draw_lead_ring(frame, lead, ring, now, alpha * 0.8)

  @staticmethod
  def _draw_dot(frame: ScopeFrame, t: RadarTrack, now: float, alpha: float, braking: float = 0.0, merging: float = 0.0) -> None:
    x, y = frame.project(t.d_rel, t.y_right)
    zone = display_zone(t)
    rgb = ZONE_RGB[zone] if zone is not None else CLASS_RGB[t.cls]
    stationary = t.cls == TrackClass.STATIONARY
    a = (0.5 if stationary else 0.95) * alpha
    radius = DOT_STATIONARY if stationary else DOT_MOVING

    if zone is not None:
      pulse = 0.5 + 0.5 * math.sin(now * 2 * math.pi / (0.6 if zone == GapZone.CRITICAL else 1.2))
      rl.draw_circle_v(rl.Vector2(x, y), DOT_THREAT + 5 + 4 * pulse, rgba(rgb, (0.16 + 0.1 * pulse) * alpha))
      radius = DOT_THREAT

    if not stationary and abs(t.v_rel) >= MIN_TICK_SPEED:
      # points up when it's pulling away, down when we're gaining on it
      length = min(max(abs(t.v_rel) * VELOCITY_TICK, 10.0), MAX_TICK)
      sign = -1.0 if t.v_rel > 0 else 1.0
      start = y + sign * radius * 0.6
      end = y + sign * (radius + length)
      rl.draw_line_ex(rl.Vector2(x, start), rl.Vector2(x, end), TICK_WIDTH, rgba(rgb, 0.8 * alpha))
      rl.draw_circle_v(rl.Vector2(x, end), TICK_WIDTH / 2, rgba(rgb, 0.8 * alpha))

    if t.camera:
      # a car the cameras found: a rounded square, so it reads apart from the radar's dots
      box = rl.Rectangle(x - radius, y - radius * 1.25, 2 * radius, 2.5 * radius)
      rl.draw_rectangle_rounded(box, 0.45, 8, rgba(rgb, a))
      # its outline turns brake red, so it reads even under a lead ring
      outline = tuple(w + (b - w) * braking for w, b in zip(WHITE, BRAKE_RGB, strict=True))
      rl.draw_rectangle_rounded_lines_ex(box, 0.45, 8, 2.5 + 1.5 * braking, rgba(outline, (0.55 + 0.45 * braking) * alpha))
      if braking > 0.01:
        draw_brake_lamps(x, y + radius * 1.25, radius, alpha * braking)
    elif t.cls == TrackClass.ONCOMING and zone is None:
      rl.draw_ring(rl.Vector2(x, y), radius - 5.0, radius, 0, 360, 24, rgba(rgb, a))
    else:
      rl.draw_circle_v(rl.Vector2(x, y), radius, rgba(rgb, a))
      if not stationary:
        rl.draw_circle_v(rl.Vector2(x - radius * 0.25, y - radius * 0.25), radius * 0.35, rgba(WHITE, 0.35 * alpha))
      if braking > 0.01:
        rl.draw_ring(rl.Vector2(x, y), radius + 1.0, radius + 4.0, 0, 360, 24, rgba(BRAKE_RGB, alpha * braking))
        draw_brake_lamps(x, y + radius * 0.9, radius, alpha * braking)
    if merging > 0.01:
      # a chevron on the side toward our lane, pointing in and nudging that way
      side = -1.0 if x > frame.cx else 1.0
      nudge = 4.0 * (0.5 - 0.5 * math.cos(now * 2 * math.pi / 0.8))
      tip = x + side * (radius + 16 + nudge)
      draw_triangle((tip, y), (tip - side * 12, y - 10), (tip - side * 12, y + 10), rgba(ZONE_RGB[GapZone.CLOSE], alpha * merging))

  @staticmethod
  def _draw_lead_ring(frame: ScopeFrame, lead: RadarTrack, style: str, now: float, alpha: float) -> None:
    center = rl.Vector2(*frame.project(lead.d_rel, lead.y_right))
    if style == "solid":
      pulse = 0.5 + 0.5 * math.sin(now * 2 * math.pi / 1.6)
      radius = LEAD_RING_RADIUS + 2.0 * pulse
      rl.draw_ring(center, radius - LEAD_RING_WIDTH, radius, 0, 360, 36, rgba(WHITE, alpha))
      rl.draw_ring(center, radius + 3.0, radius + 9.0, 0, 360, 36, rgba(WHITE, (0.12 + 0.08 * pulse) * alpha))
    else:
      radius = LEAD_RING_RADIUS - 3.0
      for k in range(6):
        start = k * 60.0 + (now * 30.0) % 60.0
        rl.draw_ring(center, radius - LEAD_RING_WIDTH + 1.0, radius, start, start + 36.0, 8, rgba(WHITE, 0.85 * alpha))

  def _draw_ego(self, frame: ScopeFrame, alpha: float) -> None:
    cx, top = frame.cx, frame.origin_y
    rl.draw_circle_gradient(rl.Vector2(cx, top + EGO_H * 0.55), 56, rgba(WHITE, 0.16 * alpha), rgba(WHITE, 0.0))
    body = rl.Rectangle(cx - EGO_W / 2, top, EGO_W, EGO_H)
    rl.draw_rectangle_rounded(body, 0.5, 10, rgba(WHITE, alpha))
    shade = rgba(INK, 0.5 * alpha)
    rl.draw_rectangle_rounded(rl.Rectangle(cx - EGO_W * 0.36, top + EGO_H * 0.2, EGO_W * 0.72, EGO_H * 0.2), 0.5, 6, shade)
    rl.draw_rectangle_rounded(rl.Rectangle(cx - EGO_W * 0.32, top + EGO_H * 0.68, EGO_W * 0.64, EGO_H * 0.14), 0.5, 6, rgba(INK, 0.35 * alpha))
    # nose marker, the origin radar distances are measured from
    draw_triangle((cx - 8, top - 3), (cx + 8, top - 3), (cx, top - 15), rgba(WHITE, 0.9 * alpha))


def draw_brake_lamps(x: float, y: float, radius: float, alpha: float) -> None:
  """Two red lamps across the back of a car (the side toward us), glowing steadily, as brake lights do."""
  for dx in (-radius * 0.65, radius * 0.65):
    c = rl.Vector2(x + dx, y)
    rl.draw_circle_v(c, 8.0, rgba(BRAKE_RGB, 0.28 * alpha))
    rl.draw_circle_v(c, 4.5, rgba(BRAKE_RGB, alpha))


def draw_person(x: float, y: float, cyclist: bool, alpha: float) -> None:
  ink = rgba(PERSON_RGB, alpha)
  rl.draw_circle_v(rl.Vector2(x, y - 13), 5.0, ink)
  rl.draw_line_ex(rl.Vector2(x, y - 8), rl.Vector2(x, y + 4), 5.0, ink)
  if cyclist:
    rl.draw_ring(rl.Vector2(x, y + 10), 4.0, 7.0, 0, 360, 16, ink)
  else:
    rl.draw_line_ex(rl.Vector2(x, y + 3), rl.Vector2(x - 5, y + 13), 4.0, ink)
    rl.draw_line_ex(rl.Vector2(x, y + 3), rl.Vector2(x + 5, y + 13), 4.0, ink)
