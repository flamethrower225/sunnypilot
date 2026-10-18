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
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, TEXT_MIN, WHITE, ZONE_RGB, rgba, draw_text, \
  draw_text_centered, draw_triangle, draw_value_with_unit, measure_value_with_unit, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import FT_PER_M, GapZone
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.detection import NOTIFICATION_KINDS, DetectionEvent, EventKind, StopPhase, \
  get_detection_feed
from openpilot.selfdrive.ui.onroad.hud_renderer import UI_CONFIG
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.light_view import LIGHT_REASON, LIGHT_RGB, LightState, current_light, current_stop_sign, \
  draw_signal_head, draw_stop_sign
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_scope import scope_rect
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

# left of the speed, just right of the speed limit sign and clear of the left turn signal
SIGN_LEFT = 60 + 30 - 6  # the speed limit sign sits this far right of the MAX box, which starts 60 px in (speed_limit.py)
VIENNA_OVERHANG = 9      # the round metric sign is 18 px wider than its box
SIGN_GAP = 24
TURN_SIGNAL_CLEARANCE = 250  # from the center: the left turn signal starts 80 + 150 px left of it, plus a gap
TOP = 40
MIN_WIDTH = 260
MAX_WIDTH = 420
PAD = 22
ICON_R = 34
PULSE_TIME = 0.8  # s

BIG_SIZE = 76
LINE_SIZE = TEXT_MIN
LINE_GAP = 6

LOG_GAP = 14
LOG_PAD = 16
LOG_SHOW = 60.0          # s an event stays in the log while stopped
LOG_SHOW_MOVING = 8.0    # s the newest event stays up while driving
LOG_FRESH = 1.2          # s a new event stays highlighted
LOG_ROWS_STOPPED = 3
LOG_LABEL_MOVING = 48    # px, the one event shown while driving is as large as an alert
LOG_DOT = 10.0           # px radius
STOPPED_SPEED = 0.5      # m/s, below this the log shows its full list


def column(rect: rl.Rectangle) -> tuple[float, float]:
  """x and width of the column the card and the log share: right of the speed limit sign, left of the left turn signal."""
  sign_w = UI_CONFIG.set_speed_width_metric if ui_state.is_metric else UI_CONFIG.set_speed_width_imperial
  sign_right = rect.x + SIGN_LEFT + 2 * sign_w + (VIENNA_OVERHANG if ui_state.is_metric else 0)
  x = sign_right + SIGN_GAP
  if ui_state.radar_scope:  # the scope sits under the sign and grows right when the rocket fuel bar takes the left edge
    scope = scope_rect(rect, ui_state.rocket_fuel)
    x = max(x, scope.x + scope.width + SIGN_GAP)
  right = rect.x + rect.width / 2 - TURN_SIGNAL_CLEARANCE
  return x, max(MIN_WIDTH, min(MAX_WIDTH, right - x))


def format_distance(meters: float, is_metric: bool) -> tuple[str, str]:
  if is_metric:
    return f"{max(meters, 0.0):.0f}", "m"
  return f"{max(meters, 0.0) * FT_PER_M:.0f}", "ft"


# bottom edge of the card this frame, so the log can sit right under it
_card_bottom: float | None = None


def card_bottom() -> float | None:
  return _card_bottom


def set_card_bottom(y: float | None) -> None:
  """For the traffic light chip, which takes the card's place in the column while there's no stop."""
  global _card_bottom
  _card_bottom = y


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

  def _lines(self) -> list[tuple[str, tuple]]:
    """(text, color) lines under the big number: a few words each, the reason first, then anything you have to do."""
    stop = get_detection_feed().stop
    sm = ui_state.sm
    lines: list[tuple[str, tuple]] = []
    light = current_light()
    if self._phase in (StopPhase.APPROACH, StopPhase.STOPPED):
      if stop.car_ahead:
        lines.append((tr("Car ahead"), WHITE))
      elif light is not None:   # the cameras can tell which
        reason = tr("Flashing red") if light.flashing and light.state == LightState.red else tr(LIGHT_REASON[light.state])
        lines.append((reason, LIGHT_RGB[light.state]))
      elif current_stop_sign() is not None:
        lines.append((tr("Stop sign"), STOP_RGB))
      else:
        lines.append((tr("Light or sign"), WHITE))
    elif self._phase == StopPhase.GO:
      lines.append((tr("Lead moving") if stop.car_ahead else tr("Model says go"), WHITE))

    if self._phase == StopPhase.APPROACH and not stop.car_ahead:
      if not sm['carControl'].longActive:
        lines.append((tr("You must brake"), WARN_RGB))
      elif not planner_follows_model(sm):
        lines.append((tr("Chill: you brake"), WARN_RGB))

    # a fixed number, not a running timer, so it doesn't pull the eye
    if stop.onset is not None and self._phase == StopPhase.APPROACH:
      value, unit = format_distance(stop.onset.distance or 0.0, ui_state.is_metric)
      lines.append((tr("Seen {} {} out").format(value, unit), MUTED_RGB))
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

    line_size, line_gap = LINE_SIZE, LINE_GAP
    wrapped = [(line, rgb) for text, rgb in self._lines() for line in wrap_text(self._font_semi, text, line_size, int(w - 2 * PAD))]
    big_h = measure_value_with_unit(self._font_bold, big[0], BIG_SIZE, self._font_semi, big[1], TEXT_MIN, gap=8).y if big else 0.0

    # the title goes beside the icon, or on its own row when the column is too narrow for it
    title_w = measure_text_cached(self._font_bold, title, TEXT_MIN, 1.5).x
    if text_x + title_w <= x + w - PAD:
      title_pos = (text_x, y + PAD)
      head_h = max(2 * ICON_R, TEXT_MIN + 8 + big_h * 0.9)
      icon_y, big_y = y + PAD + ICON_R, y + PAD + TEXT_MIN + 4
    else:
      title_pos = (x + PAD, y + PAD)
      row_h = max(2 * ICON_R, big_h * 0.9)
      head_h = TEXT_MIN + 12 + row_h
      icon_y = y + PAD + TEXT_MIN + 12 + row_h / 2
      big_y = icon_y - big_h / 2
    h = PAD + head_h + (12 + len(wrapped) * (line_size + line_gap) if wrapped else 0) + PAD

    # a red flash on the border marks the exact moment a stop was first seen
    pulse = 0.0
    if stop.onset is not None and self._phase == StopPhase.APPROACH:
      pulse = max(0.0, 1.0 - (now - stop.phase_since) / PULSE_TIME)
    card = rl.Rectangle(x, y, w, h)
    rl.draw_rectangle_rounded(card, roundness(24, w, h), 12, rgba(BLACK, 0.62 * alpha))
    rl.draw_rectangle_rounded_lines_ex(card, roundness(24, w, h), 12, 3 + 5 * pulse, rgba(color, (0.45 + 0.55 * pulse) * alpha))

    icon = rl.Vector2(x + PAD + ICON_R, icon_y)
    if pulse > 0:
      rl.draw_ring(icon, ICON_R + 4 + (1 - pulse) * 40, ICON_R + 10 + (1 - pulse) * 40, 0, 360, 48, rgba(color, 0.6 * pulse * alpha))
    self._draw_icon(icon, color, alpha)

    draw_text(self._font_bold, title, TEXT_MIN, title_pos[0], title_pos[1], rgba(color, alpha), spacing=1.5)
    if big is not None:
      draw_value_with_unit(self._font_bold, big[0], BIG_SIZE, self._font_semi, big[1], TEXT_MIN, text_x, big_y, rgba(WHITE, alpha),
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
    elif (light := current_light()) is not None and not get_detection_feed().stop.car_ahead:
      draw_signal_head(c.x, c.y, 0.72, light.state, alpha, light.arrow)
    elif current_stop_sign() is not None and not get_detection_feed().stop.car_ahead:
      draw_stop_sign(c.x, c.y, ICON_R, self._font_bold, alpha)
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
  EventKind.STOPPED_TRAFFIC: (tr_noop("STOPPED CAR"), STOP_RGB),
  EventKind.SHARP_CURVE: (tr_noop("SHARP CURVE"), ZONE_RGB[GapZone.CLOSE]),
  EventKind.SLOW_LEAD: (tr_noop("SLOW LEAD"), ZONE_RGB[GapZone.OPEN]),
  EventKind.OVER_LIMIT: (tr_noop("OVER LIMIT"), WARN_RGB),
  EventKind.SPEED_LIMIT: (tr_noop("SPEED LIMIT"), WHITE),
  EventKind.BREAK: (tr_noop("BREAK"), GO_RGB),
  EventKind.RED_LIGHT: (tr_noop("RED LIGHT"), STOP_RGB),
  EventKind.YELLOW_LIGHT: (tr_noop("YELLOW LIGHT"), WARN_RGB),
  EventKind.LIGHT_GREEN: (tr_noop("LIGHT GREEN"), GO_RGB),
  EventKind.PEDESTRIAN: (tr_noop("PEDESTRIAN"), STOP_RGB),
  EventKind.MERGING: (tr_noop("MERGING"), ZONE_RGB[GapZone.CLOSE]),
  EventKind.STOP_SIGN: (tr_noop("STOP SIGN"), STOP_RGB),
  EventKind.BRAKING_AHEAD: (tr_noop("BRAKING AHEAD"), ZONE_RGB[GapZone.CLOSE]),
  EventKind.FLASHING_SIGNAL: (tr_noop("FLASHING"), WARN_RGB),
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
  if event.note:
    parts.insert(0, event.note)
  return " • ".join(parts)


def clock(wall: float) -> str:
  t = datetime.datetime.fromtimestamp(wall)
  return f"{t:%H:%M:%S}.{t.microsecond // 100000}"


def is_stopped(sm) -> bool:
  return sm['carState'].vEgo < STOPPED_SPEED


def age_text(seconds: float) -> str:
  """How long ago, in whole minutes, so the idle line doesn't tick every second."""
  if seconds < 60:
    return tr("<1m")
  if seconds < 3600:
    return tr("{}m").format(int(seconds // 60))
  return tr("1h+")


class DetectionLogPanel:
  """What the model and planner picked up. While driving only the newest event shows, large, for a few seconds; once
  stopped the last few come back with their timestamps, when there is time to read them. In between it shrinks to one dim
  line with the last event and how long ago it was, so it's plain the log is on."""

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
    self._alpha.update(1.0)

  def render(self, rect: rl.Rectangle) -> None:
    alpha = self._alpha.x
    if not ui_state.detection_log or alpha < 0.01:
      return
    now = time.monotonic()
    log = list(get_detection_feed().log)
    recent = [e for e in log if now - e.mono_time < LOG_SHOW]
    stopped = is_stopped(ui_state.sm)

    x, w = column(rect)
    target = _card_bottom + LOG_GAP if _card_bottom is not None else rect.y + TOP
    if not self._top.initialized or target > self._top.x:
      self._top.x, self._top.initialized = target, True  # get out from under the card at once, ease back up
    y = self._top.update(target)
    if stopped and recent:
      self._draw_list(x, y, w, recent[:LOG_ROWS_STOPPED], now, alpha)
    else:
      # while driving, a notification is already on its own card, so the log shows what else came up
      moving = [e for e in log if not (ui_state.drive_notifications and e.kind in NOTIFICATION_KINDS)] if not stopped else log
      if not stopped and moving and now - moving[0].mono_time < LOG_SHOW_MOVING:
        self._draw_latest(x, y, w, moving[0], now, alpha)
      else:
        self._draw_idle(x, y, w, moving[0] if moving else None, now, alpha)

  def _draw_idle(self, x: float, y: float, w: float, event: DetectionEvent | None, now: float, alpha: float) -> None:
    if event is not None:
      label, rgb = EVENT_STYLE[event.kind]
      text, age = tr(label), age_text(now - event.mono_time)
    else:
      rgb, text, age = MUTED_RGB, tr("Nothing yet"), ""
    h = LINE_SIZE + 2 * 12
    panel = rl.Rectangle(x, y, w, h)
    rl.draw_rectangle_rounded(panel, roundness(h / 2, w, h), 12, rgba(BLACK, 0.45 * alpha))
    cy = y + h / 2
    rl.draw_circle_v(rl.Vector2(x + LOG_PAD + LOG_DOT, cy), LOG_DOT, rgba(rgb, 0.6 * alpha))
    tx = x + LOG_PAD + 2 * LOG_DOT + 14
    ty = cy - measure_text_cached(self._font_semi, text, LINE_SIZE).y / 2
    text_w = draw_text(self._font_semi, text, LINE_SIZE, tx, ty, rgba(WHITE, 0.6 * alpha)).x
    # how long ago, right aligned, left out when the label leaves no room
    age_w = measure_text_cached(self._font_semi, age, LINE_SIZE).x
    if age and tx + text_w + 16 + age_w <= x + w - LOG_PAD:
      draw_text(self._font_semi, age, LINE_SIZE, x + w - LOG_PAD - age_w, ty, rgba(WHITE, 0.45 * alpha))

  def _fit_detail(self, detail: str, room: float) -> str:
    """Drops trailing parts of the detail until it fits."""
    while detail and measure_text_cached(self._font_semi, detail, LINE_SIZE).x > room:
      detail = detail.rsplit(" • ", 1)[0] if " • " in detail else ""
    return detail

  def _draw_latest(self, x: float, y: float, w: float, event: DetectionEvent, now: float, alpha: float) -> None:
    label, rgb = EVENT_STYLE[event.kind]
    detail = self._fit_detail(event_detail(event, ui_state.is_metric), w - 2 * LOG_PAD)
    h = LOG_PAD + LOG_LABEL_MOVING + (LINE_GAP + LINE_SIZE if detail else 0) + LOG_PAD
    panel = rl.Rectangle(x, y, w, h)
    fresh = max(0.0, 1.0 - (now - event.mono_time) / LOG_FRESH)
    rl.draw_rectangle_rounded(panel, roundness(20, w, h), 12, rgba(BLACK, 0.6 * alpha))
    rl.draw_rectangle_rounded(panel, roundness(20, w, h), 12, rgba(rgb, 0.3 * fresh * alpha))
    rl.draw_rectangle_rounded_lines_ex(panel, roundness(20, w, h), 12, 3, rgba(rgb, 0.6 * alpha))

    cy = y + LOG_PAD + LOG_LABEL_MOVING / 2
    rl.draw_circle_v(rl.Vector2(x + LOG_PAD + LOG_DOT, cy), LOG_DOT, rgba(rgb, alpha))
    # as large as the column allows, never below the text floor
    label_x = x + LOG_PAD + 2 * LOG_DOT + 14
    size = LOG_LABEL_MOVING
    while size > TEXT_MIN and measure_text_cached(self._font_bold, tr(label), size).x > x + w - LOG_PAD - label_x:
      size -= 2
    label_sz = measure_text_cached(self._font_bold, tr(label), size)
    draw_text(self._font_bold, tr(label), size, label_x, cy - label_sz.y / 2, rgba(rgb, alpha))
    if detail:
      draw_text(self._font_semi, detail, LINE_SIZE, x + LOG_PAD, y + LOG_PAD + LOG_LABEL_MOVING + LINE_GAP, rgba(WHITE, 0.85 * alpha))

  def _draw_list(self, x: float, y: float, w: float, events: list[DetectionEvent], now: float, alpha: float) -> None:
    rows = []
    for event in events:
      label, rgb = EVENT_STYLE[event.kind]
      rows.append((event, tr(label), rgb, self._fit_detail(event_detail(event, ui_state.is_metric), w - 2 * LOG_PAD - 2 * LOG_DOT - 14)))
    row_h = [LINE_SIZE + 8 + (LINE_SIZE + 4 if detail else 0) + 10 for *_, detail in rows]
    h = LOG_PAD + sum(row_h) + LOG_PAD - 10
    panel = rl.Rectangle(x, y, w, h)
    rl.draw_rectangle_rounded(panel, roundness(20, w, h), 12, rgba(BLACK, 0.6 * alpha))

    ry = y + LOG_PAD
    for (event, label, rgb, detail), rh in zip(rows, row_h, strict=True):
      age = now - event.mono_time
      row_alpha = alpha * (1.0 if age < 10 else max(0.35, 1.0 - (age - 10) / (LOG_SHOW - 10)))
      if age < LOG_FRESH:
        rl.draw_rectangle_rounded(rl.Rectangle(x + 8, ry - 4, w - 16, rh - 2), 0.3, 8, rgba(rgb, 0.25 * (1.0 - age / LOG_FRESH) * alpha))
      rl.draw_circle_v(rl.Vector2(x + LOG_PAD + LOG_DOT, ry + LINE_SIZE / 2 + 2), LOG_DOT, rgba(rgb, row_alpha))
      tx = x + LOG_PAD + 2 * LOG_DOT + 14
      label_w = draw_text(self._font_bold, label, LINE_SIZE, tx, ry, rgba(rgb, row_alpha)).x
      # the time it happened, right aligned, shortened or left out when the label leaves no room
      room = x + w - LOG_PAD - (tx + label_w + 12)
      stamp = next((c for c in (clock(event.wall_time), clock(event.wall_time)[:-2]) if measure_text_cached(self._font_semi, c, LINE_SIZE).x <= room), "")
      if stamp:
        sw = measure_text_cached(self._font_semi, stamp, LINE_SIZE).x
        draw_text(self._font_semi, stamp, LINE_SIZE, x + w - LOG_PAD - sw, ry, rgba(WHITE, 0.6 * row_alpha))
      if detail:
        draw_text(self._font_semi, detail, LINE_SIZE, tx, ry + LINE_SIZE + 4, rgba(WHITE, 0.8 * row_alpha))
      ry += rh
