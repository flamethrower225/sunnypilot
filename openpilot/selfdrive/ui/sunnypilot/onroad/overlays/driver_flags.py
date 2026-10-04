"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
from dataclasses import dataclass

import pyray as rl

from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.monitoring.policy import DRIVER_MONITOR_SETTINGS
from openpilot.selfdrive.ui.onroad.driver_state import BTN_SIZE
from openpilot.selfdrive.ui import UI_BORDER_SIZE
from openpilot.selfdrive.ui.sunnypilot.onroad.developer_ui import get_bottom_dev_ui_offset
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, INK, WHITE, ZONE_RGB, rgba, draw_text, draw_text_centered, \
  draw_arc, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.lib.text_measure import measure_text_cached

_SETTINGS = DRIVER_MONITOR_SETTINGS()
PHONE_THRESH = _SETTINGS._PHONE_THRESH   # dmonitoring policy: phone_prob above this counts as on the phone
FACE_THRESH = _SETTINGS._FACE_THRESHOLD  # the policy only judges the driver while a face is found, so the flags do too
# The policy has no sleepProb rule. The model's output is a probability, so call it drowsy past even odds.
SLEEP_THRESH = 0.5

ON_DELAY = 0.3          # s above threshold before a chip appears
OFF_DELAY = 1.0         # s below threshold before it goes
FADE_RC = 0.12
GATE_RC = 0.15

CHIP_HEIGHT = 38.0
CHIP_GAP = 8.0
CHIP_TEXT_SIZE = 24.0
CHIP_ICON = 28.0
MAX_ROW_WIDTH = 390.0   # keeps the row clear of the lead gap badge
ICON_CLEARANCE = 8.0   # between the chips and the top of the driver monitoring icon

READOUT_HEIGHT = 44.0
READOUT_GAP = 10.0      # between the driver monitoring icon and the attention readout


@dataclass(frozen=True)
class FlagSpec:
  key: str
  label: str
  rgb: tuple[int, int, int]
  threshold: float


# Most severe first, so it sits at the outer edge of the row
FLAGS = (
  FlagSpec("sleep", tr_noop("DROWSY"), ZONE_RGB[GapZone.CRITICAL], SLEEP_THRESH),
  FlagSpec("phone", tr_noop("PHONE"), ZONE_RGB[GapZone.CLOSE], PHONE_THRESH),
)


class Debounce:
  """Switches on after the input has been on for on_s and off after it has been off for off_s. Time is passed in so it can be tested."""

  def __init__(self, on_s: float, off_s: float):
    self.on_s = on_s
    self.off_s = off_s
    self.state = False
    self._pending_since: float | None = None

  def reset(self) -> None:
    self.state = False
    self._pending_since = None

  def update(self, raw: bool, now: float) -> bool:
    if raw == self.state:
      self._pending_since = None
    else:
      if self._pending_since is None:
        self._pending_since = now
      if now - self._pending_since >= (self.on_s if raw else self.off_s):
        self.state = raw
        self._pending_since = None
    return self.state


def flag_inputs(driver_data) -> dict[str, bool]:
  """Which flags the model is raising right now, before debouncing. Needs a face, like the monitoring policy does."""
  face = driver_data.faceProb > FACE_THRESH
  return {
    "phone": face and driver_data.phoneProb > PHONE_THRESH,
    "sleep": face and driver_data.sleepProb > SLEEP_THRESH,
  }


def awareness_percent(dm_state) -> int:
  """Attention left before the next alert, from whichever monitoring policy is in charge."""
  if dm_state.activePolicy == log.DriverMonitoringState.MonitoringPolicy.vision:
    return int(dm_state.visionPolicyState.awarenessPercent)
  return int(dm_state.wheeltouchPolicyState.awarenessPercent)


def awareness_color(percent: int) -> tuple[int, int, int]:
  if percent > 66:
    return ZONE_RGB[GapZone.GOOD]
  if percent > 33:
    return ZONE_RGB[GapZone.CAUTION]
  return ZONE_RGB[GapZone.CRITICAL]


def fresh(service: str) -> bool:
  sm = ui_state.sm
  return sm.recv_frame[service] >= ui_state.started_frame and sm.valid[service] and sm.alive[service]


class DriverFlags:
  """Chips above the driver monitoring icon when the camera sees the driver on a phone or drowsy, with the attention left beside it."""

  def __init__(self):
    dt = 1 / gui_app.target_fps
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi_bold = gui_app.font(FontWeight.SEMI_BOLD)
    self._debounce = {f.key: Debounce(ON_DELAY, OFF_DELAY) for f in FLAGS}
    self._alpha = {f.key: FirstOrderFilter(0.0, FADE_RC, dt) for f in FLAGS}
    self._gate = FirstOrderFilter(0.0, GATE_RC, dt)
    self._awareness = 100
    self._rhd = False

  def _reset(self) -> None:
    for f in FLAGS:
      self._debounce[f.key].reset()
      self._alpha[f.key].x = 0.0
    self._gate.x = 0.0

  def update(self) -> None:
    if not ui_state.driver_flags:
      self._reset()
      return

    sm = ui_state.sm
    # same visibility as the driver monitoring icon the chips sit on: gone while an alert covers the bottom of the screen
    visible = fresh("driverStateV2") and fresh("driverMonitoringState") and sm["selfdriveState"].alertSize == log.SelfdriveState.AlertSize.none
    self._gate.update(1.0 if visible else 0.0)

    raw = dict.fromkeys(self._debounce, False)
    if visible:
      dm_state = sm["driverMonitoringState"]
      self._rhd = bool(dm_state.isRHD)
      driver_state = sm["driverStateV2"]
      raw = flag_inputs(driver_state.rightDriverData if self._rhd else driver_state.leftDriverData)
      self._awareness = awareness_percent(dm_state)

    now = rl.get_time()
    for f in FLAGS:
      on = self._debounce[f.key].update(raw[f.key], now)
      self._alpha[f.key].update(1.0 if on else 0.0)

  def render(self, rect: rl.Rectangle) -> None:
    if not ui_state.driver_flags:
      return
    gate = self._gate.x
    if gate < 0.01:
      return
    chips = [(f, self._alpha[f.key].x * gate) for f in FLAGS if self._alpha[f.key].x * gate >= 0.01]
    if not chips:
      return

    # driver_state.py puts the icon here, bottom left or mirrored for a right hand drive car
    offset = UI_BORDER_SIZE + BTN_SIZE // 2
    icon_x = rect.x + (rect.width - offset if self._rhd else offset)
    icon_y = rect.y + rect.height - offset - get_bottom_dev_ui_offset()
    direction = -1.0 if self._rhd else 1.0
    edge = icon_x - direction * BTN_SIZE / 2

    self._draw_chips(chips, edge, direction, icon_y - BTN_SIZE / 2 - ICON_CLEARANCE)
    strongest = max(a for _, a in chips)
    self._draw_readout(icon_x + direction * (BTN_SIZE / 2 + READOUT_GAP), icon_y, direction, strongest)

  def _draw_chips(self, chips, edge: float, direction: float, bottom: float) -> None:
    """One row along the bottom, most severe at the outer edge. A column of two chips would reach into the radar scope on the left.
    A fading chip gives up its slot as it goes, so the other one glides over instead of jumping."""
    x = edge
    for f, alpha in chips:
      w = self._chip_width(f)
      if abs(x - edge) + w > MAX_ROW_WIDTH:
        break
      slide = -direction * (1.0 - alpha) * 14
      left = x + slide if direction > 0 else x - w + slide
      self._draw_chip(f, left, bottom - CHIP_HEIGHT, w, alpha)
      x += direction * (w + CHIP_GAP) * alpha

  def _chip_width(self, flag: FlagSpec) -> float:
    return 8 + CHIP_ICON + 9 + measure_text_cached(self._font_bold, tr(flag.label), int(CHIP_TEXT_SIZE), 1.0).x + 16

  def _draw_chip(self, flag: FlagSpec, x: float, y: float, w: float, alpha: float) -> None:
    box = rl.Rectangle(x, y, w, CHIP_HEIGHT)
    round_box = roundness(CHIP_HEIGHT / 2, w, CHIP_HEIGHT)
    breathe = 0.5 + 0.5 * math.sin(rl.get_time() * 2 * math.pi / 1.4) if flag.key == "sleep" else 1.0
    rl.draw_rectangle_rounded(box, round_box, 20, rgba(BLACK, 0.68 * alpha))
    rl.draw_rectangle_rounded(box, round_box, 20, rgba(flag.rgb, 0.1 * alpha))
    rl.draw_rectangle_rounded_lines_ex(box, round_box, 20, 2.5, rgba(flag.rgb, (0.7 + 0.3 * breathe) * alpha))

    cx, cy = x + 8 + CHIP_ICON / 2, y + CHIP_HEIGHT / 2
    rl.draw_circle_v(rl.Vector2(cx, cy), CHIP_ICON / 2, rgba(flag.rgb, alpha))
    if flag.key == "phone":
      self._draw_phone_glyph(cx, cy, alpha)
    else:
      self._draw_closed_eye_glyph(cx, cy, alpha)
    draw_text(self._font_bold, tr(flag.label), CHIP_TEXT_SIZE, x + 8 + CHIP_ICON + 9,
              cy - measure_text_cached(self._font_bold, tr(flag.label), int(CHIP_TEXT_SIZE), 1.0).y / 2, rgba(WHITE, alpha), spacing=1.0)

  @staticmethod
  def _draw_phone_glyph(cx: float, cy: float, alpha: float) -> None:
    body = rl.Rectangle(cx - 5, cy - 8.5, 10, 17)
    rl.draw_rectangle_rounded(body, 0.35, 6, rgba(INK, 0.92 * alpha))
    screen = rl.Rectangle(cx - 3, cy - 6, 6, 10)
    rl.draw_rectangle_rounded(screen, 0.2, 4, rgba(WHITE, 0.85 * alpha))

  @staticmethod
  def _draw_closed_eye_glyph(cx: float, cy: float, alpha: float) -> None:
    color = rgba(INK, 0.92 * alpha)
    draw_arc(rl.Vector2(cx, cy - 5.5), 8.5, 2.6, 28, 152, color, round_caps=True)
    for angle in (50, 90, 130):
      rad = math.radians(angle)
      start = rl.Vector2(cx + 8.5 * math.cos(rad), cy - 5.5 + 8.5 * math.sin(rad))
      end = rl.Vector2(cx + 12.5 * math.cos(rad), cy - 5.5 + 12.5 * math.sin(rad))
      rl.draw_line_ex(start, end, 2.0, color)

  def _draw_readout(self, x: float, cy: float, direction: float, alpha: float) -> None:
    """Attention left, beside the icon. While a chip is up it shows how close the driver is to an alert."""
    percent = max(0, min(100, self._awareness))
    color = awareness_color(percent)
    value = f"{percent}%"
    value_size, caption_size = 26.0, 13.0
    caption = tr("ATTENTION")
    value_sz = measure_text_cached(self._font_bold, value, int(value_size))
    caption_sz = measure_text_cached(self._font_semi_bold, caption, int(caption_size), 1.0)
    w = max(value_sz.x, caption_sz.x) + 24
    left = x if direction > 0 else x - w
    box = rl.Rectangle(left, cy - READOUT_HEIGHT / 2, w, READOUT_HEIGHT)
    round_box = roundness(10.0, w, READOUT_HEIGHT)
    rl.draw_rectangle_rounded(box, round_box, 16, rgba(BLACK, 0.62 * alpha))
    rl.draw_rectangle_rounded_lines_ex(box, round_box, 16, 1.5, rgba(WHITE, 0.28 * alpha))
    draw_text_centered(self._font_semi_bold, caption, caption_size, left + w / 2, box.y + 11, rgba(WHITE, 0.7 * alpha), spacing=1.0)
    draw_text_centered(self._font_bold, value, value_size, left + w / 2, box.y + 29, rgba(color, alpha))
