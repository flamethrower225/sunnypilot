"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np
import pyray as rl

from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import BLACK, INK, WHITE, ZONE_RGB, rgba, draw_text, \
  draw_text_centered, roundness
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import GapZone
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

PANEL_WIDTH = 340.0
PANEL_HEIGHT = 47.0
PANEL_TOP = 305.0       # from the top of the HUD content rect, just under the speed unit
PANEL_RADIUS = 12.0
PAD_X = 10.0
CELL_GAP = 5.0
CELL_TOP = 21.0
CELL_HEIGHT = 22.0
CAPTION_Y = 4.0
CAPTION_SIZE = 15.0
LABEL_SIZE = 15.0
MAX_CELLS = 8
DEFAULT_STEP = 2.0      # s between horizons when the model doesn't say

# Risk ramp: the same numbers modeld uses to turn its disengage score into red/yellow/green (ModelConstants.RYG_*),
# so a cell turns yellow where the model itself would call that slice yellow, and red where it would call it red.
RISK_STOPS = (
  (0.5 * ModelConstants.RYG_GREEN, ZONE_RGB[GapZone.GOOD]),
  (2.0 * ModelConstants.RYG_GREEN, ZONE_RGB[GapZone.CAUTION]),
  (ModelConstants.RYG_YELLOW, ZONE_RGB[GapZone.CLOSE]),
  (2.0 * ModelConstants.RYG_YELLOW, ZONE_RGB[GapZone.CRITICAL]),
)

FADE_IN_RC = 0.12
FADE_OUT_RC = 0.25
RISK_RC = 0.3


def combine_disengage(brake, gas, steer) -> np.ndarray:
  """Cumulative probability of any takeover within each horizon: 1 - (1 - brake)(1 - gas)(1 - steer).

  This is the formula modeld itself uses for its red/yellow/green confidence (fill_model_msg.py), so the meter agrees with the model.
  The stock confidence ball (mici/onroad/confidence_ball.py) only multiplies brake and steer and ignores gas, because a gas press no
  longer disengages openpilot. Here gas stays in because the model still predicts it and it means the driver wants something other
  than what the plan is doing. The three terms are treated as independent.

  An empty list (a custom model that doesn't output that head) is skipped instead of blanking the meter, and the result is cut to
  the shortest list that is present. Returns an empty array when there is nothing to show. Cumulative values can't fall with time,
  so a noisy net is clamped with a running max.
  """
  present = [np.asarray(p, dtype=float) for p in (brake, gas, steer) if len(p) > 0]
  if not present:
    return np.zeros(0)
  n = min(len(p) for p in present)
  survive = np.ones(n)
  for p in present:
    survive *= 1.0 - np.clip(np.nan_to_num(p[:n], nan=0.0), 0.0, 1.0)
  return np.maximum.accumulate(1.0 - survive)


def slice_hazard(cumulative) -> np.ndarray:
  """Chance of a takeover inside each 2 s slice given none before it. This is modeld's 'independent disengage prob', and what the
  meter colors, so a cell lights up in the slice where the risk is, instead of every later cell turning red as the total grows."""
  cumulative = np.asarray(cumulative, dtype=float)
  if cumulative.size == 0:
    return cumulative
  previous = np.concatenate(([0.0], cumulative[:-1]))
  return np.clip((cumulative - previous) / np.maximum(1.0 - previous, 1e-6), 0.0, 1.0)


def risk_color(risk: float) -> np.ndarray:
  """Green to yellow to orange to red along RISK_STOPS."""
  xs = [s[0] for s in RISK_STOPS]
  return np.array([np.interp(risk, xs, [s[1][c] for s in RISK_STOPS]) for c in range(3)])


def horizon_labels(t, n: int) -> list[str]:
  """'2s', '4s', ... from the model's own horizon times, falling back to a 2 s step."""
  times = [float(v) for v in t][:n]
  times += [(i + 1) * DEFAULT_STEP for i in range(len(times), n)]
  return [f"{round(v):d}s" for v in times]


def confidence_color(confidence) -> tuple[int, int, int]:
  if confidence == log.ModelDataV2.ConfidenceClass.green:
    return ZONE_RGB[GapZone.GOOD]
  if confidence == log.ModelDataV2.ConfidenceClass.yellow:
    return ZONE_RGB[GapZone.CAUTION]
  return ZONE_RGB[GapZone.CRITICAL]


def percent_text(p: float) -> str:
  return "<1%" if 0.0 < p < 0.005 else f"{p * 100:.0f}%"


def model_fresh() -> bool:
  sm = ui_state.sm
  return sm.recv_frame["modelV2"] >= ui_state.started_frame and sm.valid["modelV2"] and sm.alive["modelV2"]


class DisengageMeter:
  """Five-cell strip under the speed: how likely a takeover is in each 2 s slice of the next 10 s, green to red."""

  def __init__(self):
    dt = 1 / gui_app.target_fps
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._alpha = FirstOrderFilter(0.0, FADE_OUT_RC, dt)
    self._risk = FirstOrderFilter(np.zeros(0), RISK_RC, dt, initialized=False)
    self._total = FirstOrderFilter(0.0, RISK_RC, dt, initialized=False)
    self._dot = FirstOrderFilter(np.array(ZONE_RGB[GapZone.GOOD], dtype=float), 0.2, dt)
    self._labels: list[str] = []
    self._shown = False

  def _reset(self) -> None:
    self._alpha.x = 0.0
    self._risk.initialized = False
    self._total.initialized = False
    self._shown = False

  def update(self) -> None:
    if not ui_state.disengage_horizon:
      self._reset()
      return

    cumulative = np.zeros(0)
    if model_fresh():
      meta = ui_state.sm["modelV2"].meta
      dp = meta.disengagePredictions
      cumulative = combine_disengage(dp.brakeDisengageProbs, dp.gasDisengageProbs, dp.steerOverrideProbs)[:MAX_CELLS]
      if cumulative.size:
        hazard = slice_hazard(cumulative)
        if self._risk.initialized and self._risk.x.shape != hazard.shape:
          self._risk.initialized = False
        self._risk.update(hazard)
        self._total.update(float(cumulative[-1]))
        self._labels = horizon_labels(dp.t, cumulative.size)
        self._dot.update(np.array(confidence_color(ui_state.sm["modelV2"].confidence), dtype=float))

    self._shown = cumulative.size > 0
    self._alpha.update_alpha(FADE_IN_RC if self._shown else FADE_OUT_RC)
    self._alpha.update(1.0 if self._shown else 0.0)

  def render(self, rect: rl.Rectangle) -> None:
    if not ui_state.disengage_horizon:
      return
    alpha = self._alpha.x
    if alpha < 0.01 or not self._risk.initialized:
      return

    w, h = PANEL_WIDTH, PANEL_HEIGHT
    x0, y0 = rect.x + (rect.width - w) / 2, rect.y + PANEL_TOP
    panel = rl.Rectangle(x0, y0, w, h)
    rl.draw_rectangle_rounded(panel, roundness(PANEL_RADIUS, w, h), 16, rgba(BLACK, 0.62 * alpha))
    rl.draw_rectangle_rounded_lines_ex(panel, roundness(PANEL_RADIUS, w, h), 16, 1.5, rgba(WHITE, 0.28 * alpha))

    self._draw_caption(x0, y0, w, alpha)
    self._draw_cells(x0, y0 + CELL_TOP, w, alpha)

  def _draw_caption(self, x0: float, y0: float, w: float, alpha: float) -> None:
    y = y0 + CAPTION_Y
    draw_text(self._font_bold, tr("DISENGAGE RISK"), CAPTION_SIZE, x0 + PAD_X + 2, y, rgba(WHITE, 0.72 * alpha), spacing=1.2)

    # model confidence dot with the total chance of a takeover inside the full horizon
    total = percent_text(float(self._total.x))
    total_sz = measure_text_cached(self._font_bold, total, int(CAPTION_SIZE))
    right = x0 + w - PAD_X - 2
    draw_text(self._font_bold, total, CAPTION_SIZE, right - total_sz.x, y, rgba(WHITE, 0.9 * alpha))
    radius = 5.0
    dot_center = rl.Vector2(right - total_sz.x - 10 - radius, y + total_sz.y / 2)
    rl.draw_circle_v(dot_center, radius + 2.5, rgba(BLACK, 0.5 * alpha))
    rl.draw_circle_v(dot_center, radius, rgba(self._dot.x, alpha))

  def _draw_cells(self, x0: float, y: float, w: float, alpha: float) -> None:
    risks = self._risk.x
    n = len(risks)
    cell_w = (w - 2 * PAD_X - (n - 1) * CELL_GAP) / n
    for i, risk in enumerate(risks):
      color = risk_color(float(risk))
      cell = rl.Rectangle(x0 + PAD_X + i * (cell_w + CELL_GAP), y, cell_w, CELL_HEIGHT)
      round_cell = roundness(6.0, cell_w, CELL_HEIGHT)
      rl.draw_rectangle_rounded(cell, round_cell, 8, rgba(WHITE, 0.1 * alpha))
      rl.draw_rectangle_rounded(cell, round_cell, 8, rgba(color, 0.94 * alpha))
      label = self._labels[i] if i < len(self._labels) else ""
      draw_text_centered(self._font_bold, label, LABEL_SIZE, cell.x + cell_w / 2, cell.y + CELL_HEIGHT / 2, rgba(INK, 0.9 * alpha))
