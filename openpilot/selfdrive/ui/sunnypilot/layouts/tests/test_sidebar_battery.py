"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

from openpilot.cereal import log
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.sunnypilot.layouts.sidebar import BATTERY_METER_RANGE, Colors, battery_reading, car_battery_voltage

PandaType = log.PandaState.PandaType


class FakeSM:
  def __init__(self, voltage_mv=12600, panda_type=PandaType.tres, alive=True, valid=True):
    self.alive = {"peripheralState": alive}
    self.valid = {"peripheralState": valid}
    self._state = SimpleNamespace(voltage=voltage_mv, pandaType=panda_type)

  def __getitem__(self, service):
    assert service == "peripheralState"
    return self._state


class TestBatteryReading(OpenpilotTestCase):
  @parameterized.expand([
    ("dead", 10.9, "BATTERY", "10.9V", Colors.DANGER),
    ("below_cutoff", 11.79, "BATTERY", "11.8V", Colors.DANGER),
    ("at_cutoff", 11.8, "BATTERY", "11.8V", Colors.WARNING),
    ("half_charged", 12.1, "BATTERY", "12.1V", Colors.WARNING),
    ("resting", 12.6, "BATTERY", "12.6V", Colors.GOOD),
    ("charging", 14.2, "CHARGING", "14.2V", Colors.GOOD),
    ("charging_edge", 13.2, "CHARGING", "13.2V", Colors.GOOD),
    ("overcharging", 15.4, "CHARGING", "15.4V", Colors.WARNING),
  ])
  def test_thresholds(self, _, voltage, label, value, color):
    got_label, got_value, got_color, fraction = battery_reading(voltage)
    assert (got_label, got_value, got_color) == (label, value, color)
    assert fraction is not None and 0.0 <= fraction <= 1.0

  def test_no_reading(self):
    label, value, color, fraction = battery_reading(None)
    assert (label, value, color, fraction) == ("BATTERY", "--", Colors.DISABLED, None)

  def test_meter_fills_across_its_range(self):
    lo, hi = BATTERY_METER_RANGE
    assert battery_reading(lo)[3] == 0.0 and battery_reading(hi)[3] == 1.0
    assert battery_reading(lo - 2)[3] == 0.0 and battery_reading(hi + 2)[3] == 1.0
    middle = battery_reading((lo + hi) / 2)[3]
    assert middle is not None and abs(middle - 0.5) < 1e-9


class TestCarBatteryVoltage(OpenpilotTestCase):
  def test_reads_millivolts_as_volts(self):
    assert car_battery_voltage(FakeSM(voltage_mv=12634)) == 12.634

  @parameterized.expand([
    ("not_alive", {"alive": False}),
    ("invalid", {"valid": False}),
    ("no_panda", {"panda_type": PandaType.unknown}),
    ("no_voltage", {"voltage_mv": 0}),
  ])
  def test_no_reading(self, _, kwargs):
    assert car_battery_voltage(FakeSM(**kwargs)) is None
