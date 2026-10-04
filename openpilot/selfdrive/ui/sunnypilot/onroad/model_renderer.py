"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np
import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.selfdrive.ui.sunnypilot.onroad.chevron_metrics import ChevronMetrics
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge import LeadGapBadgeState, LeadGapBadgeStyle, LeadLock
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_gap_badge.lock import LEAD_HALF_WIDTH, LEAD_HEIGHT
from openpilot.selfdrive.ui.sunnypilot.onroad.rainbow_path import RainbowPath
from openpilot.selfdrive.ui.sunnypilot.ui_state import MADSState
from openpilot.system.ui.lib.application import gui_app


class ModelRendererSP:
  def __init__(self):
    self.rainbow_path = RainbowPath()
    self.chevron_metrics = ChevronMetrics()
    self._width_filter = FirstOrderFilter(0.9, 0.1, 1 / gui_app.target_fps)
    self.lead_lock = LeadLock()
    self._lead_lock_state = LeadGapBadgeState()

  @property
  def _lateral_active(self) -> bool:
    sm = ui_state.sm
    if sm.valid["selfdriveStateSP"]:
      mads = sm["selfdriveStateSP"].mads
      if mads.available:
        return mads.enabled and mads.state != MADSState.paused
    return ui_state.status in (UIStatus.ENGAGED, UIStatus.LAT_ONLY)

  def _get_path_half_width(self) -> float:
    target = 0.9 if self._lateral_active else 0.40
    return self._width_filter.update(target)

  @property
  def lead_lock_enabled(self) -> bool:
    return ui_state.lead_gap_badge == LeadGapBadgeStyle.LOCK

  def _update_lead_lock(self, radar_state, path_x_array: np.ndarray) -> None:
    """Projects a car-sized box around leadOne for the lead lock badge."""
    lead = radar_state.leadOne if radar_state is not None else None
    if not self.lead_lock_enabled or lead is None or not lead.present:
      self.lead_lock.set_box(None)
      return

    idx = self._get_path_length_idx(path_x_array, lead.dRel)
    z = self._path.raw_points[idx, 2] if idx < len(self._path.raw_points) else 0.0
    ground = z + self._path_offset_z
    y = -lead.yRel + self._camera_offset
    corners = [
      self._map_to_screen(lead.dRel, y - LEAD_HALF_WIDTH, ground),
      self._map_to_screen(lead.dRel, y + LEAD_HALF_WIDTH, ground),
      self._map_to_screen(lead.dRel, y, ground - LEAD_HEIGHT),
    ]
    if any(c is None for c in corners):
      self.lead_lock.set_box(None)
      return

    (x0, y0), (x1, y1), (_, top) = corners
    self.lead_lock.set_box(LeadLock.pad_box(min(x0, x1), top, max(x0, x1), (y0 + y1) / 2, self._rect))

  def _draw_lead_lock(self, sm, draw_second_lead: bool) -> None:
    """Lead lock replaces the chevron and chevron metrics for leadOne; leadTwo keeps its chevron."""
    reading = self._lead_lock_state.update(sm)
    if draw_second_lead:
      lead = self._lead_vehicles[1]
      if lead.glow and lead.chevron:
        rl.draw_triangle_fan(lead.glow, len(lead.glow), rl.Color(218, 202, 37, 255))
        rl.draw_triangle_fan(lead.chevron, len(lead.chevron), rl.Color(201, 34, 49, lead.fill_alpha))

    if self._lead_lock_state.alpha > 0.01:
      self.lead_lock.render(self._rect, reading, self._lead_lock_state.zone_color.rgb, self._lead_lock_state.alpha, ui_state.is_metric)
