from opendbc.car.structs import car
from openpilot.cereal import custom
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.car.cruise import VCruiseHelper, V_CRUISE_INITIAL_EXPERIMENTAL_MODE
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.common import Mode as SpeedLimitMode

ButtonEvent = car.CarState.ButtonEvent
ButtonType = car.CarState.ButtonEvent.Type


class TestSpeedLimitInitialCruise(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()
    self.params.put("SpeedLimitMode", int(SpeedLimitMode.assist), block=True)
    # openpilot longitudinal: set speed is managed by openpilot, not the PCM
    self.v_cruise_helper = VCruiseHelper(car.CarParams(pcmCruise=False), custom.CarParamsSP(pcmCruiseSpeed=True))

  def teardown_method(self):
    self.params.remove("SpeedLimitMode")

  def enable(self, v_ego, experimental_mode, dynamic_experimental_control):
    self.v_cruise_helper.initialize_v_cruise(car.CarState(vEgo=v_ego), experimental_mode, dynamic_experimental_control)

  def set_speed_limit(self, speed_limit_final_last: float, valid: bool = True) -> None:
    LP_SP = custom.LongitudinalPlanSP.new_message()
    LP_SP.speedLimit.resolver.speedLimitValid = valid
    LP_SP.speedLimit.resolver.speedLimitFinalLast = speed_limit_final_last
    self.v_cruise_helper.update_speed_limit_assist(False, LP_SP)
    # the minimum set speed is computed in update_v_cruise before engagement
    self.v_cruise_helper.update_v_cruise(car.CarState(cruiseState={"available": True}), enabled=False, is_metric=False)

  def test_engage_at_speed_limit_with_offset(self):
    # 45 mph limit + 5 mph offset, engaging below it in Experimental Mode
    self.set_speed_limit(50 * CV.MPH_TO_MS)
    self.enable(30 * CV.MPH_TO_MS, True, False)
    assert round(self.v_cruise_helper.v_cruise_kph * CV.KPH_TO_MPH) == 50

  def test_engage_above_speed_limit(self):
    self.set_speed_limit(50 * CV.MPH_TO_MS)
    self.enable(70 * CV.MPH_TO_MS, False, False)
    assert round(self.v_cruise_helper.v_cruise_kph * CV.KPH_TO_MPH) == 50

  def test_no_speed_limit_falls_back(self):
    self.set_speed_limit(0., valid=False)
    self.enable(30 * CV.MPH_TO_MS, True, False)
    assert self.v_cruise_helper.v_cruise_kph == V_CRUISE_INITIAL_EXPERIMENTAL_MODE

  def test_sla_off_falls_back(self):
    self.params.put("SpeedLimitMode", int(SpeedLimitMode.warning), block=True)
    self.v_cruise_helper.read_custom_set_speed_params()
    self.set_speed_limit(50 * CV.MPH_TO_MS)
    self.enable(30 * CV.MPH_TO_MS, True, False)
    assert self.v_cruise_helper.v_cruise_kph == V_CRUISE_INITIAL_EXPERIMENTAL_MODE

  def test_resume_keeps_last_set_speed(self):
    self.set_speed_limit(50 * CV.MPH_TO_MS)
    self.enable(30 * CV.MPH_TO_MS, True, False)
    self.v_cruise_helper.v_cruise_kph = 100.
    self.v_cruise_helper.update_v_cruise(car.CarState(cruiseState={"available": True}), enabled=False, is_metric=False)
    self.v_cruise_helper.initialize_v_cruise(car.CarState(buttonEvents=[ButtonEvent(type=ButtonType.resumeCruise)]), True, False)
    assert self.v_cruise_helper.v_cruise_kph == 100.
