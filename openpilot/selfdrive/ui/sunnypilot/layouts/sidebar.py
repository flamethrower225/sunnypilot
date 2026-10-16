"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

import pyray as rl
import time
from dataclasses import dataclass
from openpilot.cereal import log
from openpilot.selfdrive.ui.ui_state import ui_state, ChestnutState
from openpilot.sunnypilot.sunnylink.api import UNREGISTERED_SUNNYLINK_DONGLE_ID
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr_noop


PING_TIMEOUT_NS = 80_000_000_000  # 80 seconds in nanoseconds
METRIC_HEIGHT = 126
METRIC_MARGIN = 30
METRIC_START_Y = 300
HOME_BTN = rl.Rectangle(60, 860, 180, 180)

CHESTNUT_ICON_WIDTH = 180
CHESTNUT_ICON_HEIGHT = 133

# Car battery, from the voltage the panda measures on the car's 12 V supply
BATTERY_LOW = 11.8          # V, openpilot stops charging the device below this (power_monitoring.VBATT_PAUSE_CHARGING)
BATTERY_WEAK = 12.2         # V, a resting battery this low is about half charged
BATTERY_CHARGING = 13.2     # V, above this the alternator is charging it
BATTERY_HIGH = 15.0         # V, a charging system above this is overcharging
BATTERY_METER_RANGE = (11.0, 15.0)  # V from an empty to a full meter
BATTERY_METER_SPACE = 8     # px the text moves up to make room for the meter
BATTERY_METER_HEIGHT = 8


# Color scheme
class Colors:
  WHITE = rl.WHITE
  WHITE_DIM = rl.Color(255, 255, 255, 85)
  GRAY = rl.Color(84, 84, 84, 255)

  # Status colors
  GOOD = rl.WHITE
  WARNING = rl.Color(218, 202, 37, 255)
  DANGER = rl.Color(201, 34, 49, 255)
  PROGRESS = rl.Color(0, 134, 233, 255)
  DISABLED = rl.Color(128, 128, 128, 255)

  # UI elements
  METRIC_BORDER = rl.Color(255, 255, 255, 85)
  BUTTON_NORMAL = rl.WHITE
  BUTTON_PRESSED = rl.Color(255, 255, 255, 166)


@dataclass(slots=True)
class MetricData:
  label: str
  value: str
  color: rl.Color

  def update(self, label: str, value: str, color: rl.Color):
    self.label = label
    self.value = value
    self.color = color


@dataclass(slots=True)
class BatteryMetricData(MetricData):
  fraction: float | None = None  # how full the meter is, None without a reading


def battery_reading(voltage: float | None) -> tuple[str, str, rl.Color, float | None]:
  """Label, value, color and meter fill for the car battery metric. voltage is in volts, None without a reading."""
  if voltage is None:
    return tr_noop("BATTERY"), "--", Colors.DISABLED, None
  label = tr_noop("CHARGING") if voltage >= BATTERY_CHARGING else tr_noop("BATTERY")
  if voltage < BATTERY_LOW:
    color = Colors.DANGER
  elif voltage < BATTERY_WEAK or voltage > BATTERY_HIGH:
    color = Colors.WARNING
  else:
    color = Colors.GOOD
  lo, hi = BATTERY_METER_RANGE
  return label, f"{voltage:.1f}V", color, min(max((voltage - lo) / (hi - lo), 0.0), 1.0)


def car_battery_voltage(sm) -> float | None:
  """Live car battery voltage in volts from the panda, None when there is no panda or no fresh reading."""
  if not (sm.alive['peripheralState'] and sm.valid['peripheralState']):
    return None
  state = sm['peripheralState']
  if state.pandaType == log.PandaState.PandaType.unknown or state.voltage <= 0:
    return None
  return state.voltage / 1000.0


class SidebarSP:
  def __init__(self):
    self._sunnylink_status = MetricData(tr_noop("SUNNYLINK"), tr_noop("OFFLINE"), Colors.WARNING)
    self._battery_status = BatteryMetricData(tr_noop("BATTERY"), "--", Colors.DISABLED)
    self._chestnut_green_img = gui_app.texture("icons_mici/chestnut_green.png", CHESTNUT_ICON_WIDTH, CHESTNUT_ICON_HEIGHT)
    self._chestnut_default_img = gui_app.texture("icons_mici/chestnut.png", CHESTNUT_ICON_WIDTH, CHESTNUT_ICON_HEIGHT)
    self._chestnut_orange_img = gui_app.texture("icons_mici/chestnut_orange.png", CHESTNUT_ICON_WIDTH, CHESTNUT_ICON_HEIGHT)

  def _update_sunnylink_status(self):
    if not ui_state.params.get_bool("SunnylinkEnabled"):
      self._sunnylink_status.update(tr_noop("SUNNYLINK"), tr_noop("DISABLED"), Colors.DISABLED)
      return

    last_ping = ui_state.params.get("LastSunnylinkPingTime") or 0
    dongle_id = ui_state.params.get("SunnylinkDongleId")

    is_online = last_ping and (time.monotonic_ns() - last_ping) < PING_TIMEOUT_NS
    is_temp_fault = ui_state.params.get_bool("SunnylinkTempFault")
    is_registering = not is_temp_fault and dongle_id in (None, "", UNREGISTERED_SUNNYLINK_DONGLE_ID)

    # Determine status/color pair based on priority
    if last_ping:
      status, color = (tr_noop("ONLINE"), Colors.GOOD) if is_online else (tr_noop("ERROR"), Colors.DANGER)
    elif is_temp_fault:
      status, color = (tr_noop("FAULT"), Colors.WARNING)
    elif is_registering:
      status, color = (tr_noop("REGIST..."), Colors.PROGRESS)
    else:
      status, color = (tr_noop("OFFLINE"), Colors.DANGER)

    self._sunnylink_status.update(tr_noop("SUNNYLINK"), status, color)

  def _update_battery_status(self):
    label, value, color, fraction = battery_reading(car_battery_voltage(ui_state.sm))
    self._battery_status.update(label, value, color)
    self._battery_status.fraction = fraction

  @staticmethod
  def _draw_battery_meter(metric_rect: rl.Rectangle, metric: BatteryMetricData):
    """A bar along the bottom of the battery metric, filled from 11 V to 15 V in the metric's color."""
    if metric.fraction is None:
      return
    x, w = metric_rect.x + 34, metric_rect.width - 34 - 18
    y = metric_rect.y + metric_rect.height - 20
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, BATTERY_METER_HEIGHT), 1.0, 8, Colors.GRAY)
    if metric.fraction > 0:
      fill = max(BATTERY_METER_HEIGHT, w * metric.fraction)
      rl.draw_rectangle_rounded(rl.Rectangle(x, y, fill, BATTERY_METER_HEIGHT), 1.0, 8, metric.color)

  def _get_home_icon(self, default_img: rl.Texture) -> tuple[rl.Texture, rl.Vector2, float]:
    default_pos = rl.Vector2(HOME_BTN.x, HOME_BTN.y)
    state = ui_state.chestnut_state
    if state == ChestnutState.DISCONNECTED:
      return default_img, default_pos, 1.0

    if state == ChestnutState.LOADING:
      icon = self._chestnut_default_img
      opacity = 0.35 + 0.65 * (0.5 - 0.5 * math.cos(rl.get_time() * 6.0))
    elif state in (ChestnutState.UNCOMPILED, ChestnutState.FAILED):
      icon, opacity = self._chestnut_orange_img, 1.0
    else:
      icon, opacity = self._chestnut_green_img, 1.0

    x = HOME_BTN.x + (HOME_BTN.width - icon.width) / 2
    y = HOME_BTN.y + (HOME_BTN.height - icon.height) / 2
    return icon, rl.Vector2(x, y), opacity

  def _draw_metrics_w_sunnylink(self, rect: rl.Rectangle, _temp, _panda, _battery):
    metrics = [_temp, _panda, _battery, self._sunnylink_status]
    start_y = int(rect.y) + METRIC_START_Y
    available_height = max(0, int(HOME_BTN.y) - METRIC_MARGIN - METRIC_HEIGHT - start_y)
    spacing = available_height / max(1, len(metrics) - 1)

    return metrics, start_y, spacing
