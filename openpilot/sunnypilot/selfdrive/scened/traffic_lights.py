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


class LightColor(IntEnum):
  UNKNOWN = 0
  RED = 1
  YELLOW = 2
  GREEN = 3


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


def read_lamp(rgb: np.ndarray) -> LampReading:
  """Which lamp of a signal head is lit, from a crop of its box (RGB 0-255)."""
  if rgb.size == 0 or min(rgb.shape[:2]) < 2:
    return LampReading(LightColor.UNKNOWN, 0.0, 0.0)
  h, s, v = rgb_to_hsv(rgb)
  glow = (s > GLOW_SATURATION) & (v > max(GLOW_VALUE, float(np.median(v)) + GLOW_ABOVE_MEDIAN))
  halo = (s > GLOW_SATURATION) & (v > GLOW_VALUE)
  if halo.mean() > HALO_FILLS:
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
  return LampReading(color, float(np.clip(confidence, 0.0, 1.0)), fraction)


@dataclass(frozen=True)
class LightObservation:
  """One signal head seen in one frame."""
  color: LightColor
  confidence: float
  bearing: float       # rad, right positive
  elevation: float     # rad above the horizon
  distance: float      # m from the camera
  lateral: float       # m right of straight ahead, at that distance


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


class TrafficLightFilter:
  """Pools the relevant heads of each frame into evidence per color, fading over EVIDENCE_TAU. A state shows once its evidence is
  both large enough and well ahead of the next color, so one odd reading never flips it, and it goes unknown when the lights are
  out of sight for LOST_AFTER."""

  def __init__(self):
    self.reset()

  def reset(self) -> None:
    self.evidence = dict.fromkeys((LightColor.RED, LightColor.YELLOW, LightColor.GREEN), 0.0)
    self.reading = TrafficLightReading()
    self._last_t: float | None = None
    self._last_seen = -math.inf
    self._distance = math.nan

  def update(self, now: float, observations: list[LightObservation]) -> TrafficLightReading:
    if self._last_t is not None:
      decay = math.exp(-max(now - self._last_t, 0.0) / EVIDENCE_TAU)
      for c in self.evidence:
        self.evidence[c] *= decay
    self._last_t = now

    known = [o for o in observations if o.color != LightColor.UNKNOWN]
    if observations:
      self._last_seen = now
      nearest = min(observations, key=lambda o: o.distance)
      self._distance = nearest.distance if math.isnan(self._distance) else 0.6 * self._distance + 0.4 * nearest.distance
    for o in known:
      self.evidence[o.color] += o.confidence / max(1.0, 0.5 * len(known))  # several heads agreeing count, but not linearly

    ranked = sorted(self.evidence.items(), key=lambda kv: -kv[1])
    (best, top), (_, second) = ranked[0], ranked[1]
    state = self.reading.state
    if now - self._last_seen > LOST_AFTER:
      state, self._distance = LightColor.UNKNOWN, math.nan
    elif top >= STATE_ON and top >= STATE_MARGIN * second:
      state = best
    since = self.reading.since if state == self.reading.state else now
    confidence = top / (top + second + 1e-6) if state != LightColor.UNKNOWN else 0.0
    bearing = min(observations, key=lambda o: o.distance).bearing if observations else self.reading.bearing
    self.reading = TrafficLightReading(state, float(confidence), float(self._distance), float(bearing), since,
                                       len(observations) if observations else self.reading.count)
    return self.reading
