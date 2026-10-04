"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Lead gap badge: a large, color-coded readout of the time gap to the lead car and the lead's speed.
Dial, Strip and Block are screen-space and drawn by the HUD, along with the optional gap history graph.
Lock follows the lead and Paint is drawn on the road, both by the model renderer.
"""
import pyray as rl

from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.developer_ui import get_bottom_dev_ui_offset
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.block import SignalBlock
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import ZoneColor
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.dial import GapDial
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.history import GapHistory, SPARKLINE_GAP, SPARKLINE_HEIGHT, draw_sparkline
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.lock import LeadLock
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import LeadGapBadgeStyle, LeadGapMetrics, LeadGapReading
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.road_paint import RoadPaint
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.strip import FollowStrip
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight

__all__ = ["LeadGapBadgeStyle", "LeadGapBadgeState", "LeadGapBadgeRenderer", "LeadLock", "RoadPaint"]

BOTTOM_MARGIN = 40
TORQUE_BAR_CLEARANCE = 120
STANDALONE_SPARKLINE_WIDTH = 440


class LeadGapBadgeState:
  """Reads leadOne each frame and tracks the smoothed values, zone color and fade."""

  def __init__(self):
    dt = 1 / gui_app.target_fps
    self.metrics = LeadGapMetrics(dt)
    self.zone_color = ZoneColor(dt)
    self._alpha = FirstOrderFilter(0.0, 0.15, dt)
    self.reading = LeadGapReading()

  def update(self, sm, visible: bool = True) -> LeadGapReading:
    radar_ok = sm.valid['radarState'] and sm.recv_frame['radarState'] >= ui_state.started_frame
    lead = sm['radarState'].leadOne
    self.reading = self.metrics.update(radar_ok and lead.present, lead.dRel, lead.vRel, lead.vLeadK, sm['carState'].vEgo,
                                       sm['selfdriveState'].personality)
    if self.reading.present:
      # hold the last color while fading out
      self.zone_color.update(self.reading.zone)
    self._alpha.update(1.0 if self.reading.present and visible else 0.0)
    return self.reading

  @property
  def alpha(self) -> float:
    return self._alpha.x


def alert_showing(sm) -> bool:
  return sm.recv_frame['selfdriveState'] >= ui_state.started_frame and \
    sm['selfdriveState'].alertSize != log.SelfdriveState.AlertSize.none


class LeadGapBadgeRenderer:
  """Dial, Strip and Block styles, plus the gap history graph for every style. They sit at the bottom center,
  so they step aside while an alert is up."""

  def __init__(self):
    self._state = LeadGapBadgeState()
    self._history = GapHistory(1 / gui_app.target_fps)
    self._dial = GapDial()
    self._strip = FollowStrip()
    self._block = SignalBlock()
    self._font = gui_app.font(FontWeight.SEMI_BOLD)
    self._style = LeadGapBadgeStyle.OFF

  def update(self) -> None:
    self._style = ui_state.lead_gap_badge
    if self._style == LeadGapBadgeStyle.OFF:
      self._history.clear()
      return
    sm = ui_state.sm
    reading = self._state.update(sm, visible=not alert_showing(sm))
    if ui_state.lead_gap_history:
      self._history.update(reading)
    else:
      self._history.clear()

  @staticmethod
  def bottom_offset() -> float:
    offset = BOTTOM_MARGIN + get_bottom_dev_ui_offset()
    if ui_state.torque_bar:
      offset += TORQUE_BAR_CLEARANCE
    return offset

  def _badge_width(self, rect: rl.Rectangle) -> float:
    if self._style == LeadGapBadgeStyle.DIAL:
      return self._dial.size()
    if self._style == LeadGapBadgeStyle.STRIP:
      return self._strip.width(rect)
    if self._style == LeadGapBadgeStyle.BLOCK:
      return self._block.width(rect)
    return STANDALONE_SPARKLINE_WIDTH

  def render(self, rect: rl.Rectangle) -> None:
    if self._style == LeadGapBadgeStyle.OFF or self._state.alpha < 0.01:
      return

    reading, alpha = self._state.reading, self._state.alpha
    offset = self.bottom_offset()
    if ui_state.lead_gap_history:
      w = self._badge_width(rect)
      target = None if reading.low_speed else reading.target_gap
      draw_sparkline(self._font, self._history, rect.x + (rect.width - w) / 2, rect.y + rect.height - offset - SPARKLINE_HEIGHT, w,
                     target, alpha)
      offset += SPARKLINE_HEIGHT + SPARKLINE_GAP

    renderers = {LeadGapBadgeStyle.DIAL: self._dial, LeadGapBadgeStyle.STRIP: self._strip, LeadGapBadgeStyle.BLOCK: self._block}
    if self._style in renderers:
      renderers[self._style].render(rect, offset, reading, self._state.zone_color.rgb, alpha, ui_state.is_metric)
