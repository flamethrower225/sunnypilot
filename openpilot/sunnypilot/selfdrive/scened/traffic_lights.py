"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Traffic light state. The detector finds the housings; this reads which lamp is lit, keeps the lights that are over our road, and
turns flickery per-frame readings into a state that only changes when the evidence does.

Reading a lamp: lit lamps are the saturated, bright pixels of the box. Their hue votes red, yellow or green. At night a lamp burns out
to white in the middle, so only its saturated halo votes. On a vertical housing the lit lamp's height (red on top, green at the
bottom) adds its own vote, which settles the red-or-amber question an overexposed lamp leaves open.

Which lights count: above the horizon (a red tail light never is, a signal head always is), within reach, and over the lane the
model plans to drive. Mast arms carry one head per lane and they agree except for arrows, so the relevant heads are pooled.
"""
import math
from collections import deque
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

# lamp reading
GLOW_SATURATION = 0.45
GLOW_VALUE = 0.3
GLOW_ABOVE_MEDIAN = 0.12     # brighter than the housing around it by this much
MIN_GLOW_PIXELS = 4
MIN_GLOW_FRACTION = 0.015
HUE_RED = (20.0, 315.0)      # degrees: red is below the first or above the second
HUE_YELLOW = (20.0, 68.0)
HUE_GREEN = (110.0, 196.0)   # LED greens read as blue-green on these cameras; sky starts around 200
HALO_FILLS = 0.3             # a box this full of saturated light is a lamp flaring at night
HOUSING_MAX_VALUE = 0.44     # median brightness of a lit head by day: most of the box is its dark housing
VERTICAL_ASPECT = 1.6        # a box this much taller than wide is a vertical three-lamp head
LAMP_POSITION = {1: 0.2, 2: 0.5, 3: 0.8}   # where each lamp sits down a vertical head (by LightColor value)
POSITION_WEIGHT = 0.6

# arrows
ARROW_MIN_PX = 7             # a lit lamp narrower than this can't show its shape
ARROW_MIN_AREA = 12
ARROW_ASPECT = 1.3           # a sideways arrow is at least this much wider than tall (a ball is about 1)
ARROW_HEAD_RATIO = 1.25      # the head end of an arrow is this much taller than the tail end
BALL_BESIDE_ARROW = 0.3      # weight of a round lamp when another head shows an arrow for our movement

# which lights count
MIN_ELEVATION = math.radians(1.0)    # signal heads are mounted high; tail lights and reflections are below the horizon
MAX_DISTANCE = 130.0                 # m
CORRIDOR = 5.0                       # m either side of the planned path
HEAD_WIDTH = 0.36                    # m, a signal head with its backplate cut off
HEAD_HEIGHT_ABOVE_ROAD = 5.2         # m, typical for mast arm heads

# the state filter
EVIDENCE_TAU = 0.8                   # s, evidence fades with this time constant
STATE_ON = 0.9                       # evidence needed before a state shows (about two clear readings)
STATE_MARGIN = 1.6                   # and how far ahead of the runner-up it has to be
LOST_AFTER = 1.5                     # s without a relevant light before the state goes unknown

# flashing signals
FLASH_WINDOW = 4.0                   # s of looks to judge from
FLASH_MIN_LOOKS = 6
FLASH_TOGGLES = 4                    # on-off changes in the window: about two flashes
FLASH_DARK_SHARE = (0.2, 0.8)        # share of looks that saw the lamp dark
FLASH_DARK_GAP = 1.5                 # s after the last sighting that seeing nothing counts as dark
FLASH_CONFIDENCE = 0.7

# stop signs
STOP_SIGN_WIDTH = 0.76               # m, a 30 inch octagon
SIGN_MIN_DISTANCE, SIGN_MAX_DISTANCE = 3.0, 80.0
SIGN_CORRIDOR = 9.0                  # m either side of the planned path
SIGN_EVIDENCE_TAU = 1.2              # s
SIGN_ON = 0.8
SIGN_LOST_AFTER = 2.5                # s
SIGN_PASSED = 1.5                    # m, closer than this we're at it
SIGN_RED_SHARE = 0.38                # share of the box that has to be saturated
SIGN_HUE = (-32.0, 7.5)              # degrees: its median hue, red to magenta; orange signs start around +10


class LightColor(IntEnum):
  UNKNOWN = 0
  RED = 1
  YELLOW = 2
  GREEN = 3


class Arrow(IntEnum):
  NONE = 0       # a round lamp
  LEFT = 1
  RIGHT = 2
  STRAIGHT = 3
  OTHER = 4      # an arrow, but which way can't be told (kept for the message; read_arrow calls those balls)


def largest_blob(mask: np.ndarray) -> np.ndarray:
  """The largest 4-connected component of a small boolean mask (a lamp crop is at most 48 px a side)."""
  seen = np.zeros_like(mask)
  best: list[tuple[int, int]] = []
  h, w = mask.shape
  for sy, sx in zip(*np.nonzero(mask), strict=True):
    if seen[sy, sx]:
      continue
    stack, blob = [(sy, sx)], []
    seen[sy, sx] = True
    while stack:
      y, x = stack.pop()
      blob.append((y, x))
      for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
          seen[ny, nx] = True
          stack.append((ny, nx))
    if len(blob) > len(best):
      best = blob
  out = np.zeros_like(mask)
  if best:
    ys, xs = zip(*best, strict=True)
    out[list(ys), list(xs)] = True
  return out


def read_arrow(lamp: np.ndarray) -> Arrow:
  """Whether the lit lamp is a ball or an arrow, from the shape of its glow. An arrow pointing sideways is wider than tall with its
  head, the tall end, on the side it points to; one pointing up is taller than wide with the head on top. Lamps under ARROW_MIN_PX
  are too small to tell. A stretched glow with no head at either end is a ball half hidden by its visor or smeared by its halo,
  which real heads show all the time, so it reads as a ball: counting it as an unknown arrow would let it be outvoted."""
  blob = largest_blob(lamp)
  ys, xs = np.nonzero(blob)
  if ys.size < ARROW_MIN_AREA:
    return Arrow.NONE
  w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
  if max(w, h) < ARROW_MIN_PX:
    return Arrow.NONE
  aspect = w / h
  if aspect >= ARROW_ASPECT:
    cols = blob[:, xs.min():xs.max() + 1].sum(axis=0).astype(float)
    third = max(len(cols) // 3, 1)
    left, right = cols[:third].mean(), cols[-third:].mean()
    if left > ARROW_HEAD_RATIO * right:
      return Arrow.LEFT
    if right > ARROW_HEAD_RATIO * left:
      return Arrow.RIGHT
    return Arrow.NONE
  if aspect <= 1 / ARROW_ASPECT:
    rows = blob[ys.min():ys.max() + 1, :].sum(axis=1).astype(float)
    third = max(len(rows) // 3, 1)
    return Arrow.STRAIGHT if rows[:third].mean() > ARROW_HEAD_RATIO * rows[-third:].mean() else Arrow.NONE
  return Arrow.NONE


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Hue in degrees, saturation and value 0-1, from RGB 0-255."""
  c = rgb.astype(np.float32) / 255.0
  r, g, b = c[..., 0], c[..., 1], c[..., 2]
  mx, mn = c.max(-1), c.min(-1)
  d = mx - mn
  safe = np.where(d > 1e-6, d, 1.0)
  h = np.where(mx == r, ((g - b) / safe) % 6.0, np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0))
  h = np.where(d > 1e-6, h * 60.0, 0.0)
  s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
  return h, s, mx


@dataclass(frozen=True)
class LampReading:
  color: LightColor
  confidence: float        # 0-1
  glow_fraction: float     # share of the box that is lit
  arrow: Arrow = Arrow.NONE


def read_lamp(rgb: np.ndarray) -> LampReading:
  """Which lamp of a signal head is lit, from a crop of its box (RGB 0-255)."""
  if rgb.size == 0 or min(rgb.shape[:2]) < 2:
    return LampReading(LightColor.UNKNOWN, 0.0, 0.0)
  h, s, v = rgb_to_hsv(rgb)
  glow = (s > GLOW_SATURATION) & (v > max(GLOW_VALUE, float(np.median(v)) + GLOW_ABOVE_MEDIAN))
  halo = (s > GLOW_SATURATION) & (v > GLOW_VALUE)
  flaring = halo.mean() > HALO_FILLS
  if flaring:
    # at night a lamp's flare fills the whole box and burns out to a white (often blue-tinged) core, which the brightest pixels
    # would read as green; the saturated flare around it holds the real color
    glow = halo
  elif float(np.median(v)) > HOUSING_MAX_VALUE:
    # by day a signal head is mostly dark housing; a box this bright is sky, a sign or a pole the detector mistook for one
    return LampReading(LightColor.UNKNOWN, 0.0, float(glow.mean()))
  count = int(glow.sum())
  fraction = count / glow.size
  if count < MIN_GLOW_PIXELS or fraction < MIN_GLOW_FRACTION:
    return LampReading(LightColor.UNKNOWN, 0.0, fraction)

  hue = h[glow]
  weight = v[glow]
  votes = {
    LightColor.RED: float(weight[(hue < HUE_RED[0]) | (hue > HUE_RED[1])].sum()),
    LightColor.YELLOW: float(weight[(hue >= HUE_YELLOW[0]) & (hue < HUE_YELLOW[1])].sum()),
    LightColor.GREEN: float(weight[(hue >= HUE_GREEN[0]) & (hue <= HUE_GREEN[1])].sum()),
  }
  total = sum(votes.values())
  if total <= 0:
    return LampReading(LightColor.UNKNOWN, 0.0, fraction)
  shares = {c: votes[c] / total for c in votes}

  rows, _ = np.nonzero(glow)
  height, width = glow.shape
  if height >= VERTICAL_ASPECT * width:
    position = float(rows.mean()) / height
    for c, expected in LAMP_POSITION.items():
      shares[LightColor(c)] += POSITION_WEIGHT * math.exp(-((position - expected) / 0.18) ** 2)
    norm = sum(shares.values())
    shares = {c: p / norm for c, p in shares.items()}

  color = max(shares, key=lambda c: shares[c])
  ranked = sorted(shares.values(), reverse=True)
  confidence = ranked[0] - 0.5 * ranked[1]
  arrow = Arrow.NONE
  if not flaring:  # a flaring lamp's shape is the flare's
    lo, hi = {LightColor.RED: (HUE_RED[1], HUE_RED[0] + 360.0), LightColor.YELLOW: HUE_YELLOW, LightColor.GREEN: HUE_GREEN}[color]
    hue_all = np.where(h < HUE_RED[0], h + 360.0, h) if color == LightColor.RED else h
    arrow = read_arrow(glow & (hue_all >= lo) & (hue_all <= hi))
  return LampReading(color, float(np.clip(confidence, 0.0, 1.0)), fraction, arrow)


@dataclass(frozen=True)
class LightObservation:
  """One signal head seen in one frame."""
  color: LightColor
  confidence: float
  bearing: float       # rad, right positive
  elevation: float     # rad above the horizon
  distance: float      # m from the camera
  lateral: float       # m right of straight ahead, at that distance
  arrow: Arrow = Arrow.NONE


def arrow_weight(arrow: Arrow, turn: int, ours_shown: bool = False) -> float:
  """How much a head counts toward our light, given where the planned path goes there (-1 left, 0 straight, 1 right). A round lamp
  is for everyone; an arrow only for the movement it points to, so a red left arrow never stops us going straight. ours_shown is
  whether some head shows an arrow for our movement: then that arrow rules it, and a round green beside it is for the others."""
  if arrow == Arrow.NONE:
    return BALL_BESIDE_ARROW if ours_shown else 1.0
  if arrow == Arrow.OTHER:
    return 0.3
  wanted = {-1: Arrow.LEFT, 0: Arrow.STRAIGHT, 1: Arrow.RIGHT}[turn]
  return 1.3 if arrow == wanted else 0.0


def estimate_distance(box_width_px: float, focal: float, elevation: float, camera_height: float) -> float:
  """Distance (m) to a signal head from two guesses: its apparent width against a standard head, and its angle above the horizon
  against a standard mounting height. The size guess is noisy on small boxes, the height guess on short poles; their average holds
  up better than either."""
  by_size = focal * HEAD_WIDTH / max(box_width_px, 1.0)
  if elevation > math.radians(2.0):
    by_height = (HEAD_HEIGHT_ABOVE_ROAD - camera_height) / math.tan(elevation)
    return 0.5 * (by_size + by_height)
  return by_size


def relevant(obs: LightObservation, path_lateral: float) -> bool:
  """Over our road and within reach. path_lateral is where the planned path is (m right) at the light's distance."""
  return obs.elevation >= MIN_ELEVATION and obs.distance <= MAX_DISTANCE and abs(obs.lateral - path_lateral) <= CORRIDOR


@dataclass(frozen=True)
class TrafficLightReading:
  state: LightColor = LightColor.UNKNOWN
  confidence: float = 0.0
  distance: float = math.nan    # m from the camera to the nearest relevant head
  bearing: float = 0.0
  since: float = 0.0            # monotonic s the state started
  count: int = 0                # relevant heads in the last frame that had any
  flashing: bool = False        # the lamp goes on and off: flashing red is a stop sign, flashing yellow is caution
  arrow: Arrow = Arrow.NONE     # the state comes from an arrow for our movement, not a round lamp


class TrafficLightFilter:
  """Pools the relevant heads of each frame into evidence per color, fading over EVIDENCE_TAU. A state shows once its evidence is
  both large enough and well ahead of the next color, so one odd reading never flips it, and it goes unknown when the lights are
  out of sight for LOST_AFTER. Heads count by arrow_weight for the way the planned path goes.

  It also keeps the last FLASH_WINDOW of what each look saw: a lit color, or heads seen dark (or gone) right after being lit. One
  color switching on and off over and over, with nothing else lit, is a flashing signal."""

  def __init__(self):
    self.reset()

  def reset(self) -> None:
    self.evidence = dict.fromkeys((LightColor.RED, LightColor.YELLOW, LightColor.GREEN), 0.0)
    self.reading = TrafficLightReading()
    self._last_t: float | None = None
    self._last_seen = -math.inf
    self._distance = math.nan
    self._looks: deque[tuple[float, LightColor]] = deque()
    self._arrow_votes: dict[LightColor, dict[Arrow, float]] = {}

  @property
  def last_seen(self) -> float:
    """Monotonic s a relevant head was last in view."""
    return self._last_seen

  @property
  def updated_at(self) -> float | None:
    """Monotonic s of the last update, which the reading's distance is as of."""
    return self._last_t

  def update(self, now: float, observations: list[LightObservation], turn: int = 0, v_ego: float = 0.0) -> TrafficLightReading:
    if self._last_t is not None:
      if math.isfinite(self._distance):
        self._distance = max(self._distance - v_ego * max(now - self._last_t, 0.0), 0.0)   # it came closer since the last look
      decay = math.exp(-max(now - self._last_t, 0.0) / EVIDENCE_TAU)
      for c in self.evidence:
        self.evidence[c] *= decay
      for votes in self._arrow_votes.values():
        for a in votes:
          votes[a] *= decay
    self._last_t = now

    lit = [o for o in observations if o.color != LightColor.UNKNOWN]
    ours = {-1: Arrow.LEFT, 0: Arrow.STRAIGHT, 1: Arrow.RIGHT}[turn]
    ours_shown = any(o.arrow == ours for o in lit)
    weighted = [(o, arrow_weight(o.arrow, turn, ours_shown)) for o in lit]
    weighted = [(o, w) for o, w in weighted if w > 0]
    if observations:
      self._last_seen = now
      nearest = min(observations, key=lambda o: o.distance)
      self._distance = nearest.distance if math.isnan(self._distance) else 0.6 * self._distance + 0.4 * nearest.distance
    for o, w in weighted:
      gain = w * o.confidence / max(1.0, 0.5 * len(weighted))  # several heads agreeing count, but not linearly
      self.evidence[o.color] += gain
      votes = self._arrow_votes.setdefault(o.color, {})
      votes[o.arrow] = votes.get(o.arrow, 0.0) + gain

    self._remember_look(now, weighted, bool(observations))
    flashing = self._flashing_color(now)

    ranked = sorted(self.evidence.items(), key=lambda kv: -kv[1])
    (best, top), (_, second) = ranked[0], ranked[1]
    state = self.reading.state
    if flashing is not None:
      state = flashing
    elif now - self._last_seen > LOST_AFTER:
      state, self._distance = LightColor.UNKNOWN, math.nan
    elif top >= STATE_ON and top >= STATE_MARGIN * second:
      state = best
    since = self.reading.since if state == self.reading.state and (flashing is not None) == self.reading.flashing else now
    confidence = top / (top + second + 1e-6) if state != LightColor.UNKNOWN else 0.0
    if flashing is not None:
      confidence = max(confidence, FLASH_CONFIDENCE)
    votes = self._arrow_votes.get(state, {})
    arrow = max(votes, key=lambda a: votes[a]) if votes and state != LightColor.UNKNOWN else Arrow.NONE
    bearing = min(observations, key=lambda o: o.distance).bearing if observations else self.reading.bearing
    self.reading = TrafficLightReading(state, float(confidence), float(self._distance), float(bearing), since,
                                       len(observations) if observations else self.reading.count, flashing is not None, arrow)
    return self.reading

  def _remember_look(self, now: float, weighted: list, heads_seen: bool) -> None:
    if weighted:
      totals: dict[LightColor, float] = {}
      for o, w in weighted:
        totals[o.color] = totals.get(o.color, 0.0) + w * o.confidence
      look = max(totals, key=lambda c: totals[c])
    elif heads_seen or now - self._last_seen <= FLASH_DARK_GAP:
      look = LightColor.UNKNOWN   # dark: housings in view but nothing lit, or the lamp just went out
    else:
      return
    self._looks.append((now, look))
    while self._looks and self._looks[0][0] < now - FLASH_WINDOW:
      self._looks.popleft()

  def _flashing_color(self, now: float) -> LightColor | None:
    if len(self._looks) < FLASH_MIN_LOOKS or now - self._looks[0][0] < FLASH_WINDOW * 0.7:
      return None
    lit = {c for _, c in self._looks if c != LightColor.UNKNOWN}
    if len(lit) != 1:
      return None
    (color,) = lit
    seq = [c for _, c in self._looks]
    toggles = sum(1 for a, b in zip(seq, seq[1:], strict=False) if a != b)
    dark_share = seq.count(LightColor.UNKNOWN) / len(seq)
    if toggles >= FLASH_TOGGLES and FLASH_DARK_SHARE[0] <= dark_share <= FLASH_DARK_SHARE[1] and color in (LightColor.RED, LightColor.YELLOW):
      return color
    return None


@dataclass(frozen=True)
class SignObservation:
  confidence: float
  distance: float      # m from the camera
  lateral: float       # m right


@dataclass(frozen=True)
class StopSignReading:
  present: bool = False
  confidence: float = 0.0
  distance: float = math.nan   # m from the camera


def stop_sign_face(rgb: np.ndarray) -> bool:
  """Whether a box the detector calls a stop sign shows a stop sign's face: mostly saturated red, a red that leans magenta rather
  than orange. That turns away the backs of stop signs (gray), orange and yellow diamonds, and DO NOT ENTER (a little red on
  white), all of which the detector also likes."""
  if rgb.size == 0:
    return False
  h, s, v = rgb_to_hsv(rgb)
  sat = (s > 0.4) & (v > 0.25)
  if sat.mean() < SIGN_RED_SHARE:
    return False
  hue = np.where(h > 180.0, h - 360.0, h)[sat]
  return bool(SIGN_HUE[0] <= float(np.median(hue)) <= SIGN_HUE[1])


def sign_distance(box_width_px: float, focal: float) -> float:
  return focal * STOP_SIGN_WIDTH / max(box_width_px, 1.0)


def sign_relevant(obs: SignObservation, path_lateral: float) -> bool:
  """A stop sign for us stands at the side of our road (either side: one-way streets put one on the left) within reach."""
  return SIGN_MIN_DISTANCE <= obs.distance <= SIGN_MAX_DISTANCE and abs(obs.lateral - path_lateral) <= SIGN_CORRIDOR


class StopSignFilter:
  """A stop sign is there once it has been seen a couple of times, and gone once it hasn't for a while or we've reached it."""

  def __init__(self):
    self.reset()

  def reset(self) -> None:
    self.evidence = 0.0
    self.reading = StopSignReading()
    self._last_t: float | None = None
    self._last_seen = -math.inf
    self._distance = math.nan

  @property
  def updated_at(self) -> float | None:
    return self._last_t

  def update(self, now: float, observations: list[SignObservation], v_ego: float = 0.0) -> StopSignReading:
    if self._last_t is not None:
      dt = max(now - self._last_t, 0.0)
      self.evidence *= math.exp(-dt / SIGN_EVIDENCE_TAU)
      if math.isfinite(self._distance):
        self._distance -= v_ego * dt   # it comes closer between looks
    self._last_t = now
    if observations:
      nearest = min(observations, key=lambda o: o.distance)
      self.evidence += max(o.confidence for o in observations)
      self._last_seen = now
      self._distance = nearest.distance if not math.isfinite(self._distance) else 0.5 * self._distance + 0.5 * nearest.distance
    # once there it stays until out of sight for SIGN_LOST_AFTER, though its evidence fades between looks
    present = (self.evidence >= SIGN_ON or self.reading.present) and now - self._last_seen <= SIGN_LOST_AFTER and \
      self._distance > SIGN_PASSED
    if not present and now - self._last_seen > SIGN_LOST_AFTER:
      self._distance = math.nan
    self.reading = StopSignReading(present, float(min(self.evidence, 1.0)) if present else 0.0,
                                   float(self._distance) if present else math.nan)
    return self.reading
