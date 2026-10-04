"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Track brackets: corner brackets projected onto radar tracks in the camera view, so you can see what the radar sees.
"""
import math
import time
from collections.abc import Callable

import numpy as np
import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, INK, WHITE, ZONE_RGB, rgba, draw_text, \
  draw_delta_arrow, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone, format_distance, speed_value
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.projection import RoadProjector
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_tracks import BOARD_TTC_MAX, CLASS_RGB, RadarFeed, RadarTrack, TrackClass, Lane, \
  display_zone, in_threat_corridor, rank_threats, NO_TRACK
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

BOX_WIDTH, BOX_HEIGHT = 1.8, 1.3     # m, a car's rear as seen from behind
MAX_TRACKS = 8
MAX_LATERAL = 8.0                    # m
MAX_DISTANCE = 100.0                 # m
MAX_TAGS = 3
MIN_BOX_W, MIN_BOX_H = 34.0, 26.0    # px, so a distant track is still a visible mark
MAX_ARM_X, MAX_ARM_Y = 64.0, 48.0    # px, so a close bracket doesn't swallow the view
MARGIN = 14.0

NEAR_TAG_DISTANCE = 30.0             # m, closer than this a track is tagged even when it isn't closing
CLOSING_MIN_TAG = 0.5                # m/s, slower than this the tag shows distance instead of closing speed


def select_tracks(tracks: list[RadarTrack], skip_id: int = NO_TRACK, limit: int = MAX_TRACKS, max_threats: int = MAX_TAGS) -> list[RadarTrack]:
  """The tracks that get brackets: the nearest ones within reach, except the lead (it has its own lock) and with the
  most threatening ones always included, even when the nearest few are all guardrail. Returned nearest first."""
  candidates = [t for t in tracks if 0.0 < t.d_rel < MAX_DISTANCE and abs(t.y_rel) <= MAX_LATERAL and t.track_id != skip_id]
  chosen = rank_threats([t for t in candidates if display_zone(t) is not None])[:max_threats]
  taken = {id(t) for t in chosen}
  for t in sorted(candidates, key=lambda t: t.d_rel):
    if len(chosen) >= limit:
      break
    if id(t) not in taken:
      chosen.append(t)
  return sorted(chosen, key=lambda t: t.d_rel)


def tagged_tracks(tracks: list[RadarTrack], count: int = MAX_TAGS) -> set[int]:
  """Track ids of the most threatening tracks, the ones that get a tag: closing on us, or right next to us. Stationary tracks
  never do, and neither does traffic pulling away in the distance."""
  candidates = [t for t in tracks if in_threat_corridor(t) and (t.ttc <= BOARD_TTC_MAX or t.d_rel <= NEAR_TAG_DISTANCE)]
  return {t.track_id for t in rank_threats(candidates)[:count]}


def project_box(to_screen: Callable[[float, float, float], tuple[float, float] | None], d_rel: float, y_right: float) -> np.ndarray | None:
  """Screen box (left, top, right, bottom) around a BOX_WIDTH x BOX_HEIGHT target standing on the road, or None
  when part of it falls outside the view."""
  half = BOX_WIDTH / 2
  corners = [c for dy in (-half, half) for h in (0.0, BOX_HEIGHT) if (c := to_screen(d_rel, y_right + dy, h)) is not None]
  if len(corners) < 4:
    return None
  xs, ys = [c[0] for c in corners], [c[1] for c in corners]
  return pad_box(min(xs), min(ys), max(xs), max(ys))


def pad_box(left: float, top: float, right: float, bottom: float) -> np.ndarray:
  """Grows a box that projected tiny up to a minimum size, keeping it standing on the same spot."""
  cx = (left + right) / 2
  w, h = max(right - left, MIN_BOX_W), max(bottom - top, MIN_BOX_H)
  return np.array([cx - w / 2, bottom - h, cx + w / 2, bottom])


class _Bracket:
  """One track's on-screen state: the smoothed box, color and fade, which keep animating after the track is gone."""

  def __init__(self, dt: float):
    self.box = FirstOrderFilter(np.zeros(4), 0.08, dt, initialized=False)
    self.color = FirstOrderFilter(np.zeros(3), 0.15, dt, initialized=False)
    self.alpha = FirstOrderFilter(0.0, 0.1, dt)
    self.track: RadarTrack | None = None
    self.zone: GapZone | None = None
    self.tagged = False


class TrackBrackets:
  """Corner brackets on up to eight radar tracks, 1.8 m x 1.3 m at the track's distance. Closing threats are colored by time to
  collision, same-direction traffic is cyan, oncoming pink, stationary things dim gray. The three most threatening carry a tag
  with their closing speed. The lead is skipped, it has its own lock."""

  def __init__(self):
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi_bold = gui_app.font(FontWeight.SEMI_BOLD)
    self._dt = 1 / gui_app.target_fps
    self._feed = RadarFeed(self._dt)
    self._selected: list[RadarTrack] = []
    self._tagged: set[int] = set()
    self._brackets: dict[int, _Bracket] = {}

  def update(self) -> None:
    if not ui_state.radar_track_brackets:
      self._selected = []
      self._tagged = set()
      self._brackets.clear()
      return
    self._feed.update(ui_state.sm, ui_state.started_frame, time.monotonic())
    self._selected = select_tracks(self._feed.tracks, self._feed.lead_ids[0])
    self._tagged = tagged_tracks(self._selected)

  def render(self, rect: rl.Rectangle, projector: RoadProjector) -> None:
    if not ui_state.radar_track_brackets:
      return
    if not projector.ready:
      return

    live = set()
    for t in self._selected:
      box = project_box(projector.to_screen, t.d_rel, t.y_right)
      if box is None:
        continue
      b = self._brackets.get(t.track_id)
      if b is None:
        b = self._brackets[t.track_id] = _Bracket(self._dt)
      zone = display_zone(t)
      rgb = ZONE_RGB[zone] if zone is not None else CLASS_RGB[t.cls]
      b.box.update(box)
      b.color.update(np.array(rgb, dtype=float))
      b.alpha.update(1.0)
      b.track, b.zone, b.tagged = t, zone, t.track_id in self._tagged
      live.add(t.track_id)

    for track_id in list(self._brackets):
      if track_id not in live:
        b = self._brackets[track_id]
        if b.alpha.update(0.0) < 0.02:
          del self._brackets[track_id]

    now = rl.get_time()
    # far to near, so close brackets sit on top
    for b in sorted(self._brackets.values(), key=lambda b: -b.track.d_rel if b.track else 0.0):
      self._draw(rect, b, now)

  def _draw(self, rect: rl.Rectangle, b: _Bracket, now: float) -> None:
    t = b.track
    if t is None or b.alpha.x < 0.02:
      return
    left, top, right, bottom = b.box.x
    stationary = t.cls == TrackClass.STATIONARY and b.zone is None
    alpha = b.alpha.x * (0.45 if stationary and t.lane != Lane.CENTER else (0.65 if stationary else 0.95))

    if b.zone == GapZone.CRITICAL:
      scale = 1.0 + 0.06 * (0.5 + 0.5 * math.sin(now * 2 * math.pi / 0.8))
      cx, cy = (left + right) / 2, (top + bottom) / 2
      left, right = cx - (cx - left) * scale, cx + (right - cx) * scale
      top, bottom = cy - (cy - top) * scale, cy + (bottom - cy) * scale

    self._draw_corners(left, top, right, bottom, b.color.x, alpha, thick=b.zone is not None)
    if b.tagged:
      self._draw_tag(rect, (left + right) / 2, bottom, t, b.zone, b.color.x, alpha)

  @staticmethod
  def _draw_corners(left: float, top: float, right: float, bottom: float, rgb, alpha: float, thick: bool) -> None:
    w, h = right - left, bottom - top
    arm_x, arm_y = float(np.clip(w * 0.3, 8.0, MAX_ARM_X)), float(np.clip(h * 0.36, 8.0, MAX_ARM_Y))
    t = float(np.clip(w * (0.05 if thick else 0.036), 2.2, 6.5))
    corners = (
      ((left, top), (1, 1)),
      ((right, top), (-1, 1)),
      ((left, bottom), (1, -1)),
      ((right, bottom), (-1, -1)),
    )
    for thickness, a in ((t * 2.4, 0.2 * alpha), (t, alpha)):
      color = rgba(rgb, a)
      for (x, y), (dx, dy) in corners:
        corner = rl.Vector2(x, y)
        rl.draw_line_ex(corner, rl.Vector2(x + dx * arm_x, y), thickness, color)
        rl.draw_line_ex(corner, rl.Vector2(x, y + dy * arm_y), thickness, color)
        rl.draw_circle_v(corner, thickness / 2, color)

  def _draw_tag(self, rect: rl.Rectangle, cx: float, box_bottom: float, t: RadarTrack, zone: GapZone | None, rgb, alpha: float) -> None:
    closing = t.closing_speed >= CLOSING_MIN_TAG
    if closing:
      value, unit = str(speed_value(t.closing_speed, ui_state.is_metric)), tr("km/h") if ui_state.is_metric else tr("mph")
    else:
      value, unit = format_distance(t.d_rel, ui_state.is_metric)

    num_size, unit_size = 25, 17
    num_sz = measure_text_cached(self._font_bold, value, num_size)
    unit_sz = measure_text_cached(self._font_semi_bold, unit, unit_size)
    arrow = 13.0 if closing else 0.0
    pad_x, pad_y, gap = 11.0, 4.0, 5.0
    content_w = (arrow + gap if arrow else 0.0) + num_sz.x + gap + unit_sz.x
    tag_w, tag_h = content_w + 2 * pad_x, num_sz.y + 2 * pad_y
    x = float(np.clip(cx - tag_w / 2, rect.x + MARGIN, rect.x + rect.width - MARGIN - tag_w))
    y = min(box_bottom + 8.0, rect.y + rect.height - MARGIN - tag_h)

    tag = rl.Rectangle(x, y, tag_w, tag_h)
    radius = roundness(tag_h / 2, tag_w, tag_h)
    if zone is not None:
      fill, ink, muted = rgba(rgb, alpha), rgba(INK, alpha), rgba(INK, 0.8 * alpha)
      rl.draw_rectangle_rounded(tag, radius, 12, fill)
    else:
      fill, ink, muted = rgba(BLACK, 0.7 * alpha), rgba(WHITE, alpha), rgba(WHITE, 0.65 * alpha)
      rl.draw_rectangle_rounded(tag, radius, 12, fill)
      rl.draw_rectangle_rounded_lines_ex(tag, radius, 12, 2.0, rgba(rgb, 0.8 * alpha))

    tx = x + pad_x
    if closing:
      draw_delta_arrow(-1, tx, y + tag_h / 2, arrow, ink)
      tx += arrow + gap
    draw_text(self._font_bold, value, num_size, tx, y + pad_y, ink)
    tx += num_sz.x + gap
    draw_text(self._font_semi_bold, unit, unit_size, tx, y + pad_y + num_sz.y * 0.8 - unit_sz.y * 0.8, muted)
