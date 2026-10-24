"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The traffic light state scened reads from the cameras, for the HUD: a chip in the stop column with a signal head showing the lit lamp
and the distance, which hands its place to the stop card while that is up (the card then names the light itself).
"""
import math
import time
from dataclasses import dataclass

import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, TEXT_MIN, WHITE, ZONE_RGB, draw_text, rgba, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays import stop_indicator
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.light_view import HEAD_W, LIGHT_RGB, LIGHT_WORD, LightView, SignView, current_light, \
  current_stop_sign, draw_signal_head, draw_stop_sign, flash_level
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.lib.text_measure import measure_text_cached

CHIP_H = 96.0
CHIP_PAD = 16.0
LIGHT_RED, LIGHT_GREEN = 1, 3
STOPPED_SPEED = 0.5    # m/s, waiting at a red below this: the chip counts the wait instead of the distance
WAIT_SHOW = 2.0        # s stopped before the wait shows, so rolling up to a light doesn't flash a timer
FLASHING_WORD = tr_noop("FLASHING")

FADE_RC = 0.15         # s, the chip coming and going
SWAP_RC = 0.1          # s, one state handing over to the next (red to green, a light to a stop sign)
DISTANCE_RC = 0.3      # s, the distance gliding between passes
DISTANCE_SNAP = 12.0   # m, a jump this big is a different light: snap to it
PULSE_TIME = 0.7       # s a state change rings out from the chip
PULSE_REACH = 26.0     # px
WORD_SLIDE = 14.0      # px the word moves as one state hands over to the next


def distance_text(distance: float) -> tuple[str, str]:
  if not math.isfinite(distance):
    return "", ""
  return stop_indicator.format_distance(distance, ui_state.is_metric)


def wait_text(seconds: float) -> str:
  seconds = int(max(seconds, 0.0))
  return f"{seconds // 60}:{seconds % 60:02d}"


@dataclass(frozen=True)
class Face:
  """What the chip shows, apart from the numbers: when it changes, the old face fades into the new."""
  kind: str          # "light" or "sign"
  state: int = 0
  arrow: int = 0
  flashing: bool = False

  @property
  def rgb(self) -> tuple:
    return LIGHT_RGB.get(self.state, WHITE) if self.kind == "light" else ZONE_RGB[GapZone.CRITICAL]

  @property
  def word(self) -> str:
    if self.kind == "sign":
      return tr("STOP")
    return tr(FLASHING_WORD) if self.flashing else tr(LIGHT_WORD[self.state])


def face_of(view: LightView | SignView) -> Face:
  if isinstance(view, LightView):
    return Face("light", view.state, view.arrow, view.flashing)
  return Face("sign")


def mix(a: tuple, b: tuple, t: float) -> tuple:
  return tuple(x + (y - x) * t for x, y in zip(a, b, strict=True))


class TrafficLightChip:
  """The light ahead (its lamp, arrow and flashing as read), how long we've waited at it while red, or else a stop sign ahead. A
  change of state cross-fades and rings out once; the distance glides between scened's passes."""

  def __init__(self):
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi = gui_app.font(FontWeight.SEMI_BOLD)
    dt = 1 / gui_app.target_fps
    self._alpha = FirstOrderFilter(0.0, FADE_RC, dt)
    self._swap = FirstOrderFilter(1.0, SWAP_RC, dt)        # 0 just after a change of face, 1 settled
    self._distance = FirstOrderFilter(math.nan, DISTANCE_RC, dt)
    self._wait_mix = FirstOrderFilter(0.0, FADE_RC, dt)
    self._view: LightView | SignView | None = None
    self._face: Face | None = None
    self._prev_face: Face | None = None
    self._pulse_at = -math.inf
    self._wait_since: float | None = None

  def update(self) -> None:
    now = time.monotonic()
    view: LightView | SignView | None = None
    if ui_state.started:
      view = current_light() or current_stop_sign()
    if view is not None:
      face = face_of(view)
      if self._face is None or self._alpha.x < 0.02:   # coming up from nothing: no hand-over
        self._prev_face, self._swap.x = None, 1.0
        self._distance.x = view.distance
      elif face != self._face:
        self._prev_face, self._swap.x = self._face, 0.0
        if face.kind != self._face.kind or face.state != self._face.state:
          self._pulse_at = now
      self._face, self._view = face, view
      d = view.distance
      if math.isfinite(d) and (not math.isfinite(self._distance.x) or abs(d - self._distance.x) > max(DISTANCE_SNAP, 0.25 * d)):
        self._distance.x = d
      else:
        self._distance.update(d)
    self._swap.update(1.0)

    stopped_at_red = isinstance(view, LightView) and view.state == LIGHT_RED and not view.flashing and \
      ui_state.sm['carState'].vEgo < STOPPED_SPEED
    if not stopped_at_red:
      self._wait_since = None
    elif self._wait_since is None:
      self._wait_since = now
    self._wait_mix.update(1.0 if self.waited is not None and self.waited >= WAIT_SHOW else 0.0)
    self._alpha.update(1.0 if view is not None else 0.0)

  @property
  def waited(self) -> float | None:
    """Seconds stopped at the red light ahead, or None when not waiting at one."""
    return None if self._wait_since is None else time.monotonic() - self._wait_since

  def render(self, rect: rl.Rectangle) -> None:
    # the stop card has the column while it's up, and names the light itself: the chip fades out under it as it fades in
    card = stop_indicator.card_bottom()
    alpha = self._alpha.x * (1.0 - stop_indicator.card_alpha() if card is not None else 1.0)
    face = self._face
    if alpha < 0.01 or face is None:
      return
    now = time.monotonic()
    x, w = stop_indicator.column(rect)
    y = rect.y + stop_indicator.TOP
    chip = rl.Rectangle(x, y, w, CHIP_H)
    swap = self._swap.x
    prev = self._prev_face if swap < 0.99 else None
    rgb = mix(prev.rgb, face.rgb, swap) if prev is not None else face.rgb
    r = roundness(22, w, CHIP_H)

    pulse = (now - self._pulse_at) / PULSE_TIME
    if 0.0 <= pulse < 1.0:
      grow = PULSE_REACH * pulse
      ring = rl.Rectangle(x - grow, y - grow, w + 2 * grow, CHIP_H + 2 * grow)
      rl.draw_rectangle_rounded_lines_ex(ring, roundness(22 + grow, ring.width, ring.height), 12, 4.0,
                                         rgba(face.rgb, 0.7 * (1.0 - pulse) ** 2 * alpha))
    rl.draw_rectangle_rounded(chip, r, 12, rgba(BLACK, 0.62 * alpha))
    rl.draw_rectangle_rounded_lines_ex(chip, r, 12, 3, rgba(rgb, 0.7 * alpha))

    # the old word lifts out as the new one rises into place, rather than the two sitting on top of each other
    for f, a, dy in ((prev, (1.0 - swap) ** 2 * alpha, -WORD_SLIDE * swap), (face, swap * alpha if prev is not None else alpha,
                                                                             WORD_SLIDE * (1.0 - swap) if prev is not None else 0.0)):
      if f is not None and a > 0.01:
        self._draw_face(f, x, y, a, now, dy)

    tx = x + CHIP_PAD + HEAD_W + 18
    word_w = measure_text_cached(self._font_bold, face.word, self._word_size(face)).x
    room = x + w - CHIP_PAD - (tx + word_w + 20)
    value, unit = distance_text(self._distance.x)
    waited = self.waited
    for text, a in ((f"{value} {unit}".strip(), 1.0 - self._wait_mix.x), (wait_text(waited) if waited is not None else "", self._wait_mix.x)):
      if text and a > 0.01:
        sz = measure_text_cached(self._font_semi, text, TEXT_MIN)
        if sz.x <= room:
          draw_text(self._font_semi, text, TEXT_MIN, x + w - CHIP_PAD - sz.x, y + (CHIP_H - sz.y) / 2, rgba(WHITE, 0.8 * a * alpha))
    stop_indicator.set_card_bottom(max(card, y + CHIP_H) if card is not None else y + CHIP_H)

  @staticmethod
  def _word_size(face: Face) -> int:
    return 40 if face.flashing else 48   # FLASHING is a longer word

  def _draw_face(self, face: Face, x: float, y: float, alpha: float, now: float, dy: float = 0.0) -> None:
    icon_cx, cy = x + CHIP_PAD + HEAD_W / 2, y + CHIP_H / 2
    if face.kind == "light":
      draw_signal_head(icon_cx, cy, 1.0, face.state, alpha, face.arrow, flash_level(now) if face.flashing else 1.0)
    else:
      draw_stop_sign(icon_cx, cy, HEAD_W * 0.95, self._font_bold, alpha)
    size = self._word_size(face)
    word_sz = measure_text_cached(self._font_bold, face.word, size)
    draw_text(self._font_bold, face.word, size, x + CHIP_PAD + HEAD_W + 18, y + (CHIP_H - word_sz.y) / 2 + dy, rgba(face.rgb, alpha))
