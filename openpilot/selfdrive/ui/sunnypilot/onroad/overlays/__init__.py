"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Optional HUD overlays, each behind its own toggle in Settings > Visuals.
HudOverlays are screen-space and drawn by the HUD; RoadOverlays are projected into the camera view by the model renderer.
Every overlay checks its own toggle in update() and render().
"""
import pyray as rl

from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.disengage_meter import DisengageMeter
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.driver_flags import DriverFlags
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.hard_brake_flash import HardBrakeFlash
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.projection import RoadProjector
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.radar_scope import RadarScope
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.threat_board import ThreatBoard
from openpilot.selfdrive.ui.sunnypilot.onroad.overlays.track_brackets import TrackBrackets

# drawn in this order, warnings last so they sit on top
HUD_OVERLAY_TYPES: list[type] = [RadarScope, ThreatBoard, DisengageMeter, DriverFlags, HardBrakeFlash]
ROAD_OVERLAY_TYPES: list[type] = [TrackBrackets]


class HudOverlays:
  def __init__(self):
    self._overlays = [cls() for cls in HUD_OVERLAY_TYPES]

  def update(self) -> None:
    for overlay in self._overlays:
      overlay.update()

  def render(self, rect: rl.Rectangle) -> None:
    for overlay in self._overlays:
      overlay.render(rect)


class RoadOverlays:
  def __init__(self):
    self._overlays = [cls() for cls in ROAD_OVERLAY_TYPES]

  def render(self, rect: rl.Rectangle, projector: RoadProjector) -> None:
    for overlay in self._overlays:
      overlay.update()
      overlay.render(rect, projector)
