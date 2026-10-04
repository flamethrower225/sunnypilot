"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
from dataclasses import dataclass

import numpy as np
import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import WHITE, draw_text, draw_triangle, rgba
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.projection import RoadProjector
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr

# Lead wake: when the lead car is braking hard, red chevrons on the road show where it is predicted to be 2, 4 and 6 s from
# now (relative to us), so you see the gap about to close before it does.

BRAKE_ACCEL = -1.5        # m/s², the lead is braking when its acceleration is below this
MODEL_LEAD_PROB = 0.5     # the model's lead has to be at least this likely to be trusted
RADAR_TO_CAMERA = 1.52    # m, modelV2 lead x is measured from the camera, radarState dRel from the front bumper
WAKE_TIMES = (2.0, 4.0, 6.0)  # s, the points of leadsV3 (t = 0, 2, 4, ...) to draw
WAKE_ALPHA = (1.0, 0.85, 0.68)  # chevron opacity at each of the WAKE_TIMES
MIN_WAKE_X = 3.0          # m, closer than this is under the hood
HOLD_TIME = 0.8           # s, the wake stays up this long after the lead stops braking, then fades

RED = (232, 54, 60)       # lead chevron red is (201, 34, 49), a little hotter here so it reads on its own
GLOW = (255, 118, 88)
PULSE_PERIOD = 1.1        # s
LABEL_SPACING = 24.0      # px, minimum vertical gap between two time labels


@dataclass(frozen=True)
class WakePoint:
  t: float   # s from now
  x: float   # m ahead of the front bumper
  y: float   # m to the right of the car's centerline


# ---- pure math ----

def lead_braking(a_lead_k: float, model_prob: float, model_a0: float) -> bool:
  """Radar-based lead acceleration, or the model's, when the model is sure it sees the lead."""
  return a_lead_k < BRAKE_ACCEL or (model_prob > MODEL_LEAD_PROB and model_a0 < BRAKE_ACCEL)


def model_wake_points(lead_t, lead_x, lead_y, times=WAKE_TIMES) -> list[WakePoint]:
  """Predicted lead positions from modelV2.leadsV3[0]. Its y is right positive like the path, so no sign flip (radarState
  yRel is the one that is left positive, see radard.py: yRel = -lead.y[0]). Its x is from the camera."""
  if len(lead_t) < 2 or len(lead_x) != len(lead_t) or len(lead_y) != len(lead_t):
    return []
  return [WakePoint(t, float(np.interp(t, lead_t, lead_x)) - RADAR_TO_CAMERA, float(np.interp(t, lead_t, lead_y))) for t in times]


def kinematic_wake_points(d_rel: float, y_rel: float, v_ego: float, v_lead: float, a_lead: float, times=WAKE_TIMES) -> list[WakePoint]:
  """Same thing from radarState when the model doesn't see the lead: the lead holds a_lead until it has stopped and we hold
  our speed. yRel is left positive, hence the sign flip."""
  pts = []
  for t in times:
    if a_lead < -1e-3 and v_lead > 0.0:
      t_run = min(t, v_lead / -a_lead)
      travel = v_lead * t_run + 0.5 * a_lead * t_run ** 2
    else:
      travel = v_lead * t + 0.5 * a_lead * t ** 2
    pts.append(WakePoint(t, d_rel + travel - v_ego * t, -y_rel))
  return pts


def visible_wake_points(points: list[WakePoint]) -> list[WakePoint]:
  return [p for p in points if p.x > MIN_WAKE_X]


class HoldTimer:
  """True while active, and for hold seconds after that."""

  def __init__(self, hold: float, dt: float):
    self._hold = hold
    self._dt = dt
    self._elapsed = math.inf

  def update(self, active: bool) -> bool:
    self._elapsed = 0.0 if active else self._elapsed + self._dt
    return self._elapsed <= self._hold + 1e-6

  def reset(self) -> None:
    self._elapsed = math.inf


class LeadWake:
  """Fading red chevrons at the lead's predicted positions while it brakes."""

  def __init__(self):
    dt = 1 / gui_app.target_fps
    self._font = gui_app.font(FontWeight.SEMI_BOLD)
    self._hold = HoldTimer(HOLD_TIME, dt)
    self._alpha = FirstOrderFilter(0.0, 0.15, dt)
    self._points: list[WakePoint] = []

  def _reset(self) -> None:
    self._alpha.x = 0.0
    self._hold.reset()
    self._points = []

  def update(self) -> None:
    if not ui_state.lead_braking_wake:
      self._reset()
      return

    sm = ui_state.sm
    lead = sm['radarState'].leadOne
    present = bool(sm.valid['radarState'] and sm.recv_frame['radarState'] >= ui_state.started_frame and lead.present)

    braking = False
    if present:
      model_prob, model_a0, model_lead = 0.0, 0.0, None
      if sm.recv_frame['modelV2'] >= ui_state.started_frame and len(sm['modelV2'].leadsV3):
        model_lead = sm['modelV2'].leadsV3[0]
        model_prob = model_lead.prob
        model_a0 = model_lead.a[0] if len(model_lead.a) else 0.0
      braking = lead_braking(lead.aLeadK, model_prob, model_a0)

      if braking:
        points = []
        if model_lead is not None and model_prob > MODEL_LEAD_PROB:
          points = model_wake_points(list(model_lead.t), list(model_lead.x), list(model_lead.y))
        if not points:
          points = kinematic_wake_points(lead.dRel, lead.yRel, sm['carState'].vEgo, lead.vLeadK, lead.aLeadK)
        self._points = visible_wake_points(points)

    # once it stops braking the last prediction stays up for the hold time, then fades
    self._alpha.update(1.0 if self._hold.update(braking) and self._points else 0.0)

  def render(self, rect: rl.Rectangle, projector: RoadProjector) -> None:
    if not ui_state.lead_braking_wake or self._alpha.x < 0.01 or not self._points or not projector.ready:
      return

    alpha = self._alpha.x
    now = rl.get_time()

    # farthest first so nearer chevrons overlap the ones behind them
    label_ys: list[float] = []
    for i, pt in sorted(enumerate(self._points), key=lambda ip: -ip[1].x):
      p = projector.to_screen(pt.x, pt.y, 0.0)
      if p is None:
        continue
      reveal = float(np.clip(alpha * 1.8 - i * 0.35, 0.0, 1.0))  # chevrons come in one after the other, nearest the lead first
      pulse = 0.78 + 0.22 * math.sin(now * 2 * math.pi / PULSE_PERIOD - i * 1.1)
      # chevrons that are close on screen would stack their time labels, so only the farthest of them gets one
      show_label = all(abs(p[1] - y) > LABEL_SPACING for y in label_ys)
      if show_label:
        label_ys.append(p[1])
      self._draw_chevron(rect, pt, p, float(np.interp(pt.t, WAKE_TIMES, WAKE_ALPHA)) * pulse * reveal, show_label)

  def _draw_chevron(self, rect: rl.Rectangle, pt: WakePoint, screen: tuple[float, float], opacity: float, show_label: bool) -> None:
    """Same shape and scaling as the lead chevron in ModelRenderer._update_lead_vehicle: apex at the road point, base toward us."""
    if opacity < 0.01:
      return
    sz = float(np.clip((25 * 30) / (pt.x / 3 + 30), 15.0, 30.0)) * 2.35 * 0.95
    x = float(np.clip(screen[0], rect.x, rect.x + rect.width - sz / 2))
    y = min(screen[1], rect.y + rect.height - sz * 0.6)
    g_xo, g_yo = sz / 5, sz / 10

    glow = [(x + sz * 1.35 + g_xo, y + sz + g_yo), (x, y - g_yo), (x - sz * 1.35 - g_xo, y + sz + g_yo)]
    chevron = [(x + sz * 1.25, y + sz), (x, y), (x - sz * 1.25, y + sz)]
    draw_triangle(glow[0], glow[1], glow[2], rgba(GLOW, 0.35 * opacity))
    draw_triangle(chevron[0], chevron[1], chevron[2], rgba(RED, opacity))
    for a, b in zip(chevron, chevron[1:] + chevron[:1], strict=True):  # a light rim keeps it apart from the path underneath
      rl.draw_line_ex(rl.Vector2(*a), rl.Vector2(*b), 2.0, rgba((255, 214, 208), 0.7 * opacity))

    if show_label:
      draw_text(self._font, tr("{} s").format(int(pt.t)), 21.0, x + sz * 1.25 + 10.0, y + sz * 0.45, rgba(WHITE, 0.85 * opacity))
