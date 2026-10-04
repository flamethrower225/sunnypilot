"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Lead gap badge: a large, color-coded readout of the time gap to the lead car and the lead's speed.
Dial and Strip are screen-space and drawn by the HUD; Lock follows the lead and is drawn by the model renderer.
"""
import pyray as rl

from openpilot.cereal import log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.developer_ui import get_bottom_dev_ui_offset
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.common import ZoneColor
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.dial import GapDial
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.lock import LeadLock
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.metrics import LeadGapBadgeStyle, LeadGapMetrics, LeadGapReading
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.strip import FollowStrip
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app

__all__ = ["LeadGapBadgeStyle", "LeadGapBadgeState", "LeadGapBadgeRenderer", "LeadLock"]

BOTTOM_MARGIN = 40
TORQUE_BAR_CLEARANCE = 120


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
  """Dial and Strip styles. They sit at the bottom center, so they step aside while an alert is up."""

  def __init__(self):
    self._state = LeadGapBadgeState()
    self._dial = GapDial()
    self._strip = FollowStrip()
    self._style = LeadGapBadgeStyle.OFF

  def update(self) -> None:
    self._style = ui_state.lead_gap_badge
    if self._style not in (LeadGapBadgeStyle.DIAL, LeadGapBadgeStyle.STRIP):
      return
    sm = ui_state.sm
    self._state.update(sm, visible=not alert_showing(sm))

  @staticmethod
  def bottom_offset() -> float:
    offset = BOTTOM_MARGIN + get_bottom_dev_ui_offset()
    if ui_state.torque_bar:
      offset += TORQUE_BAR_CLEARANCE
    return offset

  def render(self, rect: rl.Rectangle) -> None:
    if self._style not in (LeadGapBadgeStyle.DIAL, LeadGapBadgeStyle.STRIP) or self._state.alpha < 0.01:
      return

    renderer = self._dial if self._style == LeadGapBadgeStyle.DIAL else self._strip
    renderer.render(rect, self.bottom_offset(), self._state.reading, self._state.zone_color.rgb, self._state.alpha, ui_state.is_metric)
