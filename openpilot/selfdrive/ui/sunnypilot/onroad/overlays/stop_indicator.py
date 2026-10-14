"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import datetime
import time

import pyray as rl

from openpilot.cereal import custom
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, WHITE, ZONE_RGB, rgba, draw_text, draw_text_centered, \
  draw_triangle, draw_value_with_unit, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import FT_PER_M, GapZone
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.detection import DetectionEvent, EventKind, StopPhase, get_detection_feed
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wrap_text import wrap_text

DEC_BLENDED = custom.LongitudinalPlanSP.DynamicExperimentalControl.DynamicExperimentalControlState.blended

STOP_RGB = ZONE_RGB[GapZone.CRITICAL]
GO_RGB = ZONE_RGB[GapZone.GOOD]
WARN_RGB = ZONE_RGB[GapZone.CAUTION]
MUTED_RGB = ZONE_RGB[GapZone.NONE]

# right of the speed, clear of the turn signal, the wheel button and the threat board
X_FROM_CENTER = 240
TOP = 40
MAX_WIDTH = 420
WHEEL_CLEARANCE = 270
THREAT_BOARD_CLEARANCE = 450
PAD = 22
ICON_R = 34
PULSE_TIME = 0.8  # s

LOG_GAP = 14
LOG_ROW_H = 42
LOG_SHOW = 60.0   # s an event stays in the log
LOG_FRESH = 1.2   # s a new row stays highlighted


def column(rect: rl.Rectangle) -> tuple[float, float]:
  """x and width of the column the card and the log share."""
  x = rect.x + rect.width / 2 + X_FROM_CENTER
  right = rect.x + rect.width - (THREAT_BOARD_CLEARANCE if ui_state.threat_board else WHEEL_CLEARANCE)
  return x, max(260.0, min(MAX_WIDTH, right - x))


def format_distance(meters: float, is_metric: bool) -> tuple[str, str]:
  if is_metric:
    return f"{max(meters, 0.0):.0f}", "m"
  return f"{max(meters, 0.0) * FT_PER_M:.0f}", "ft"


# bottom edge of the card this frame, so the log can sit right under it
_card_bottom: float | None = None


def update_feed() -> None:
  # wall time is only for showing when something happened; all timing uses the monotonic clock
  get_detection_feed().update(ui_state.sm, ui_state.started_frame, datetime.datetime.now().timestamp(), time.monotonic())


class StopCard:
  """The moment the model starts planning a stop, how far away it is, why, and whether openpilot will act on it."""

  def __init__(self):
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi = gui_app.font(FontWeight.SEMI_BOLD)
    self._alpha = FirstOrderFilter(0.0, 0.12, 1 / gui_app.target_fps)
    self._distance = 0.0
    self._phase = StopPhase.IDLE

  def update(self) -> None:
    if not ui_state.stop_detection:
      self._alpha.x = 0.0
      return
    update_feed()
    stop = get_detection_feed().stop
    if stop.phase != StopPhase.IDLE:
      self._phase = stop.phase  # keep the last phase while fading out
    if stop.stop_x is not None:
      self._distance = stop.stop_x  # already smoothed by the detector; the stop line shows the same number
    self._alpha.update(1.0 if stop.phase != StopPhase.IDLE else 0.0)

  def _lines(self, now: float) -> list[tuple[str, tuple]]:
    """(text, color) lines under the big number."""
    stop = get_detection_feed().stop
    sm = ui_state.sm
    lines: list[tuple[str, tuple]] = []
    if self._phase in (StopPhase.APPROACH, StopPhase.STOPPED):
      if stop.car_ahead:
        lines.append((tr("Car ahead"), WHITE))
      else:
        lines.append((tr("No car ahead"), WHITE))
        lines.append((tr("Likely a light or sign"), MUTED_RGB))
    elif self._phase == StopPhase.GO:
      lines.append((tr("Traffic is moving") if stop.car_ahead else tr("Model wants to go"), WHITE))
    elif self._phase == StopPhase.CLEARED:
      lines.append((tr("No longer planning a stop"), WHITE))

    if stop.onset is not None and self._phase in (StopPhase.APPROACH, StopPhase.STOPPED):
      value, unit = format_distance(stop.onset.distance or 0.0, ui_state.is_metric)
      lines.append((tr("Seen {:.1f}s ago, {} {} out").format(now - stop.onset.mono_time, value, unit), MUTED_RGB))

    if self._phase == StopPhase.APPROACH and not stop.car_ahead:
      if not sm['carControl'].longActive:
        lines.append((tr("Not controlling speed"), WARN_RGB))
      elif not planner_follows_model(sm):
        lines.append((tr("Chill mode won't stop"), WARN_RGB))
    return lines

  def render(self, rect: rl.Rectangle) -> None:
    global _card_bottom
    _card_bottom = None
    alpha = self._alpha.x
    if not ui_state.stop_detection or alpha < 0.01:
      return

    now = time.monotonic()
    stop = get_detection_feed().stop
    x, w = column(rect)
    y = rect.y + TOP
    text_x = x + PAD + 2 * ICON_R + 18

    color = {StopPhase.GO: GO_RGB, StopPhase.CLEARED: MUTED_RGB}.get(self._phase, STOP_RGB)
    title = {StopPhase.APPROACH: tr("STOP AHEAD"), StopPhase.STOPPED: tr("STOPPED"), StopPhase.GO: tr("GO"),
             StopPhase.CLEARED: tr("STOP CLEARED")}.get(self._phase, "")

    big: tuple[str, str] | None = None
    if self._phase == StopPhase.APPROACH:
      big = format_distance(self._distance, ui_state.is_metric)
    elif self._phase == StopPhase.STOPPED:
      waited = int(now - stop.phase_since)
      big = (f"{waited // 60}:{waited % 60:02d}", "")

    line_size, line_gap = 27, 5
    wrapped = [(line, rgb) for text, rgb in self._lines(now) for line in wrap_text(self._font_semi, text, line_size, int(w - 2 * PAD))]
    head_h = max(2 * ICON_R, 44 + (78 if big else 0))
    h = PAD + head_h + (12 + len(wrapped) * (line_size + line_gap) if wrapped else 0) + PAD

    # a red flash on the border marks the exact moment a stop was first seen
    pulse = 0.0
    if stop.onset is not None and self._phase == StopPhase.APPROACH:
      pulse = max(0.0, 1.0 - (now - stop.phase_since) / PULSE_TIME)
    card = rl.Rectangle(x, y, w, h)
    rl.draw_rectangle_rounded(card, roundness(24, w, h), 12, rgba(BLACK, 0.62 * alpha))
    rl.draw_rectangle_rounded_lines_ex(card, roundness(24, w, h), 12, 3 + 5 * pulse, rgba(color, (0.45 + 0.55 * pulse) * alpha))

    icon = rl.Vector2(x + PAD + ICON_R, y + PAD + ICON_R)
    if pulse > 0:
      rl.draw_ring(icon, ICON_R + 4 + (1 - pulse) * 40, ICON_R + 10 + (1 - pulse) * 40, 0, 360, 48, rgba(color, 0.6 * pulse * alpha))
    self._draw_icon(icon, color, alpha)

    draw_text(self._font_bold, title, 36, text_x, y + PAD, rgba(color, alpha), spacing=1.5)
    if big is not None:
      draw_value_with_unit(self._font_bold, big[0], 76, self._font_semi, big[1], 30, text_x, y + PAD + 40, rgba(WHITE, alpha),
                           rgba(WHITE, 0.7 * alpha), gap=8)

    ly = y + PAD + head_h + 12
    for text, rgb in wrapped:
      draw_text(self._font_semi, text, line_size, x + PAD, ly, rgba(rgb, alpha))
      ly += line_size + line_gap
    _card_bottom = y + h

  def _draw_icon(self, c: rl.Vector2, color, alpha: float) -> None:
    rl.draw_circle_v(c, ICON_R, rgba(color, alpha))
    ink = rgba(BLACK if self._phase == StopPhase.GO else WHITE, alpha)
    if self._phase == StopPhase.GO:
      draw_triangle((c.x, c.y - 20), (c.x - 16, c.y - 2), (c.x + 16, c.y - 2), ink)
      rl.draw_rectangle_rec(rl.Rectangle(c.x - 6, c.y - 4, 12, 22), ink)
    elif self._phase == StopPhase.CLEARED:
      draw_text_centered(self._font_bold, "✓", 40, c.x, c.y, ink)
    else:
      rl.draw_rectangle_rounded(rl.Rectangle(c.x - 20, c.y - 6, 40, 12), 0.5, 6, ink)


def planner_follows_model(sm) -> bool:
  """True when the longitudinal planner includes the model's own plan (experimental mode, or DEC in blended)."""
  if not sm['selfdriveState'].experimentalMode:
    return False
  dec = sm['longitudinalPlanSP'].dec
  return not dec.active or dec.state == DEC_BLENDED


EVENT_STYLE = {
  EventKind.STOP: (tr_noop("STOP"), STOP_RGB),
  EventKind.CLEARED: (tr_noop("CLEARED"), MUTED_RGB),
  EventKind.GO: (tr_noop("GO"), GO_RGB),
  EventKind.GREEN_LIGHT: (tr_noop("GREEN"), GO_RGB),
  EventKind.LEAD: (tr_noop("LEAD"), ZONE_RGB[GapZone.OPEN]),
  EventKind.LEAD_LOST: (tr_noop("LEAD LOST"), MUTED_RGB),
  EventKind.CUT_IN: (tr_noop("CUT-IN"), ZONE_RGB[GapZone.CLOSE]),
  EventKind.LEAD_DEPART: (tr_noop("DEPARTING"), GO_RGB),
  EventKind.CURVE: (tr_noop("CURVE"), WARN_RGB),
  EventKind.HARD_BRAKE: (tr_noop("HARD BRAKE"), STOP_RGB),
}


def event_detail(event: DetectionEvent, is_metric: bool) -> str:
  parts = []
  if event.distance is not None:
    parts.append(" ".join(format_distance(event.distance, is_metric)))
  if event.kind == EventKind.CURVE and event.value is not None:
    parts.insert(0, f"{event.value / 9.81:.2f} g")
  if event.kind in (EventKind.STOP, EventKind.GO) and event.car_ahead is not None:
    parts.append(tr("car") if event.car_ahead else tr("no car"))
  if event.kind in (EventKind.LEAD, EventKind.CUT_IN) and event.radar is not None:
    parts.append(tr("radar") if event.radar else tr("vision"))
  return " • ".join(parts)


def clock(wall: float) -> str:
  t = datetime.datetime.fromtimestamp(wall)
  return f"{t:%H:%M:%S}.{t.microsecond // 100000}"


class DetectionLogPanel:
  """Timestamped list of what the model and planner picked up, newest first."""

  def __init__(self):
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi = gui_app.font(FontWeight.SEMI_BOLD)
    self._alpha = FirstOrderFilter(0.0, 0.2, 1 / gui_app.target_fps)
    self._top = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps, initialized=False)

  def update(self) -> None:
    if not ui_state.detection_log:
      self._alpha.x = 0.0
      return
    update_feed()
    recent = time.monotonic() - get_detection_feed().last_event_mono < LOG_SHOW
    self._alpha.update(1.0 if recent else 0.0)

  def render(self, rect: rl.Rectangle) -> None:
    alpha = self._alpha.x
    if not ui_state.detection_log or alpha < 0.01:
      return
    now = time.monotonic()
    events = [e for e in get_detection_feed().log if now - e.mono_time < LOG_SHOW]
    if not events:
      return

    x, w = column(rect)
    target = _card_bottom + LOG_GAP if _card_bottom is not None else rect.y + TOP
    if not self._top.initialized or target > self._top.x:
      self._top.x, self._top.initialized = target, True  # get out from under the card at once, ease back up
    y = self._top.update(target)
    h = 50 + len(events) * LOG_ROW_H + 10
    panel = rl.Rectangle(x, y, w, h)
    rl.draw_rectangle_rounded(panel, roundness(20, w, h), 12, rgba(BLACK, 0.55 * alpha))
    draw_text(self._font_bold, tr("DETECTIONS"), 20, x + 16, y + 14, rgba(WHITE, 0.6 * alpha), spacing=2)

    ry = y + 50
    for event in events:
      age = now - event.mono_time
      row_alpha = alpha * (1.0 if age < 10 else max(0.35, 1.0 - (age - 10) / (LOG_SHOW - 10)))
      label, rgb = EVENT_STYLE[event.kind]
      if age < LOG_FRESH:
        fresh = 1.0 - age / LOG_FRESH
        rl.draw_rectangle_rounded(rl.Rectangle(x + 8, ry - 3, w - 16, LOG_ROW_H - 2), 0.4, 8, rgba(rgb, 0.25 * fresh * alpha))
      tx = x + 16
      tx += draw_text(self._font_semi, clock(event.wall_time), 21, tx, ry + 5, rgba(WHITE, 0.55 * row_alpha)).x + 12
      rl.draw_circle_v(rl.Vector2(tx + 7, ry + 18), 7, rgba(rgb, row_alpha))
      tx += 22
      tx += draw_text(self._font_bold, tr(label), 23, tx, ry + 3, rgba(rgb, row_alpha)).x + 10
      detail = event_detail(event, ui_state.is_metric)
      room = x + w - 16 - tx
      while detail and measure_text_cached(self._font_semi, detail, 21).x > room:
        detail = detail.rsplit(" • ", 1)[0] if " • " in detail else ""
      if detail:
        dw = measure_text_cached(self._font_semi, detail, 21).x
        draw_text(self._font_semi, detail, 21, x + w - 16 - dw, ry + 5, rgba(WHITE, 0.8 * row_alpha))
      ry += LOG_ROW_H
