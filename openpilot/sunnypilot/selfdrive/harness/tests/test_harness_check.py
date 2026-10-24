"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import json

import pytest

from openpilot.sunnypilot.selfdrive.harness.harness_check import (FINAL_S, GIVE_UP_S, LFA, LFA_ALT, LKAS, LKAS_ALT, MIN_FRAMES, HarnessSession,
                                                                   HarnessTally, Stamp, Verdict, alert_text, boot_id, classify, decision_from,
                                                                   is_final, settled)

OTHER = 0x1A0          # some ordinary car message
FD_LEN, CLASSIC_LEN = 32, 8
NOW = 1_800_000_000.0
MONO, BOOT = 500.0, "boot-a"
STAMP = Stamp(NOW, MONO, BOOT)
HDA2_CAR = decision_from("KIA_SPORTAGE_HEV_2026", "hyundai", True)
NON_HDA2_CAR = decision_from("HYUNDAI_TUCSON_2025", "hyundai", False)


def feed(tally: HarnessTally, bus: int, address: int, frames: int, length: int = FD_LEN, hz: float = 100.0, start_s: float = 0.0) -> None:
  for i in range(frames):
    tally.ingest(int((start_s + i / hz) * 1e9), bus, address, length)


def car_traffic(tally: HarnessTally, buses=(0, 1, 2)) -> None:
  """Ordinary CAN-FD traffic on each bus, so the tally has a sensible shape."""
  for bus in buses:
    feed(tally, bus, OTHER, 100)


def check(tally: HarnessTally, decision=None) -> dict:
  return classify(tally, decision, STAMP)


class TestVerdicts:
  def test_lkas_on_the_camera_bus_is_correct_hda2_wiring(self):
    t = HarnessTally()
    car_traffic(t)
    feed(t, 2, LKAS, 200)
    r = check(t)
    assert r["verdict"] == Verdict.OK_HDA2
    assert r["camera_bus"] == 2
    assert "No pin swap needed" in r["action"]

  def test_lkas_alt_on_the_camera_bus_also_counts_as_hda2(self):
    t = HarnessTally()
    car_traffic(t)
    feed(t, 2, LKAS_ALT, 200)
    assert check(t)["verdict"] == Verdict.OK_HDA2

  def test_lkas_on_another_bus_needs_a_pin_swap(self):
    t = HarnessTally()
    car_traffic(t)
    feed(t, 0, LKAS, 200)
    r = check(t)
    assert r["verdict"] == Verdict.SWAP_NEEDED
    assert "bus 0" in r["title"]
    assert "bus 2" in r["detail"]

  def test_lkas_on_two_wrong_buses_names_both(self):
    t = HarnessTally()
    car_traffic(t)
    feed(t, 0, LKAS, 200)
    feed(t, 1, LKAS_ALT, 200)
    assert "buses 0 and 1" in check(t)["title"]

  def test_miswired_hda2_with_lfa_landing_on_bus_2_is_still_a_swap(self):
    # the wrong pair on bus 2 can carry the ADAS ECU's LFA while the camera's LKAS sits on bus 1: openpilot would call it non-HDA2
    t = HarnessTally()
    car_traffic(t)
    feed(t, 1, LKAS, 200)
    feed(t, 2, LFA, 200)
    assert check(t)["verdict"] == Verdict.SWAP_NEEDED

  def test_lfa_on_the_camera_bus_is_the_non_hda2_layout(self):
    t = HarnessTally()
    car_traffic(t)
    feed(t, 2, LFA, 200)
    r = check(t)
    assert r["verdict"] == Verdict.NON_HDA2
    assert "non-HDA2" in r["detail"]

  def test_lfa_alt_on_the_camera_bus_is_the_non_hda2_layout(self):
    t = HarnessTally()
    car_traffic(t)
    feed(t, 2, LFA_ALT, 200)
    assert check(t)["verdict"] == Verdict.NON_HDA2

  def test_lfa_only_on_another_bus_means_a_swap_is_likely(self):
    t = HarnessTally()
    car_traffic(t)
    feed(t, 0, LFA, 200)
    r = check(t)
    assert r["verdict"] == Verdict.SWAP_LIKELY
    assert "bus 2" in r["title"] and "bus 0" in r["detail"]

  def test_no_steering_messages_is_inconclusive(self):
    t = HarnessTally()
    car_traffic(t)
    r = check(t)
    assert r["verdict"] == Verdict.INCONCLUSIVE
    assert "No steering messages" in r["title"]

  def test_no_traffic_at_all_is_inconclusive(self):
    r = check(HarnessTally())
    assert r["verdict"] == Verdict.INCONCLUSIVE
    assert r["buses"] == [] and r["seconds"] == 0.0

  def test_classic_can_car_is_inconclusive_even_with_matching_ids(self):
    t = HarnessTally()
    for bus in (0, 1, 2):
      feed(t, bus, OTHER, 100, length=CLASSIC_LEN)
    feed(t, 0, LKAS, 200, length=CLASSIC_LEN)
    r = check(t)
    assert r["verdict"] == Verdict.INCONCLUSIVE
    assert r["can_fd"] is False
    assert "not CAN-FD" in r["detail"]


class TestCounting:
  def test_frames_below_min_frames_do_not_count(self):
    t = HarnessTally()
    car_traffic(t)
    feed(t, 0, LKAS, MIN_FRAMES - 1)
    feed(t, 2, LKAS, MIN_FRAMES)
    assert check(t)["verdict"] == Verdict.OK_HDA2

  def test_a_stray_frame_on_a_wrong_bus_does_not_trigger_a_swap(self):
    t = HarnessTally()
    car_traffic(t)
    feed(t, 1, LKAS, 3)
    assert check(t)["verdict"] == Verdict.INCONCLUSIVE

  def test_sent_and_blocked_frames_are_ignored(self):
    t = HarnessTally()
    car_traffic(t)
    feed(t, 128 + 0, LKAS, 200)   # our own TX echoed back on bus 0
    feed(t, 192 + 1, LKAS, 200)   # blocked on bus 1
    r = check(t)
    assert r["verdict"] == Verdict.INCONCLUSIVE
    assert {b["bus"] for b in r["buses"]} == {0, 1, 2}

  def test_second_panda_moves_the_camera_bus_to_6(self):
    t = HarnessTally()
    car_traffic(t, buses=(0, 1, 2, 4, 5, 6))
    feed(t, 6, LKAS, 200)
    r = check(t)
    assert r["camera_bus"] == 6
    assert r["verdict"] == Verdict.OK_HDA2

  def test_second_panda_with_lkas_on_bus_2_is_a_swap(self):
    t = HarnessTally()
    car_traffic(t, buses=(0, 1, 2, 4, 5, 6))
    feed(t, 2, LKAS, 200)
    assert check(t)["verdict"] == Verdict.SWAP_NEEDED

  def test_seconds_come_from_can_timestamps(self):
    t = HarnessTally()
    feed(t, 0, OTHER, 101, hz=100.0, start_s=5.0)
    assert t.seconds == pytest.approx(1.0)

  def test_per_bus_breakdown_and_rates(self):
    t = HarnessTally()
    car_traffic(t)
    feed(t, 2, LKAS, 100, hz=100.0)   # 100 frames over the tally's ~1 s
    r = check(t)
    bus2 = next(b for b in r["buses"] if b["bus"] == 2)
    assert bus2["frames"] == 200 and bus2["ids"] == 2
    assert bus2["steering"]["LKAS"]["frames"] == 100
    assert bus2["steering"]["LKAS"]["hz"] == pytest.approx(100.0, rel=0.01)
    assert next(b for b in r["buses"] if b["bus"] == 0)["steering"] == {}

  def test_rate_uses_the_messages_own_time_span(self):
    # the camera starts sending 10 s into the check: its rate must still read 100 Hz, not a third of that
    t = HarnessTally()
    feed(t, 0, OTHER, 1501, hz=100.0)
    feed(t, 2, LKAS, 500, hz=100.0, start_s=10.0)
    lkas = next(b for b in check(t)["buses"] if b["bus"] == 2)["steering"]["LKAS"]
    assert lkas["hz"] == pytest.approx(100.0, rel=0.01)


class TestDecision:
  def test_unrecognized_car(self):
    assert decision_from("MOCK", "mock", False) == {"recognized": False, "car": "", "hda2": None}
    assert decision_from("", "", False)["recognized"] is False

  def test_hda2_flag_only_reported_for_hyundai(self):
    assert decision_from("KIA_SPORTAGE_HEV_2026", "hyundai", True)["hda2"] is True
    assert decision_from("TOYOTA_RAV4", "toyota", True)["hda2"] is None

  def test_decision_agreeing_with_the_wiring(self):
    t = HarnessTally()
    car_traffic(t)
    feed(t, 2, LKAS, 200)
    r = check(t, HDA2_CAR)
    assert r["decision"] == HDA2_CAR and r["disagrees"] is False

  def test_decision_disagreeing_with_the_wiring(self):
    t = HarnessTally()
    car_traffic(t)
    feed(t, 2, LKAS, 200)
    assert check(t, NON_HDA2_CAR)["disagrees"] is True

  def test_missing_decision_defaults_to_unrecognized(self):
    t = HarnessTally()
    car_traffic(t)
    r = check(t, None)
    assert r["decision"]["recognized"] is False and r["disagrees"] is False


class TestTiming:
  def test_definite_verdicts_are_final_at_final_s(self):
    assert not is_final(FINAL_S - 0.1, Verdict.OK_HDA2)
    assert is_final(FINAL_S, Verdict.OK_HDA2)
    assert is_final(FINAL_S, Verdict.SWAP_NEEDED)

  def test_inconclusive_keeps_listening_until_give_up(self):
    assert not is_final(FINAL_S + 10, Verdict.INCONCLUSIVE)
    assert is_final(GIVE_UP_S, Verdict.INCONCLUSIVE)


class TestAlert:
  DRIVE_START = MONO - 5

  def _result(self, verdict, mono=MONO, boot=BOOT, final=True):
    return {"verdict": str(verdict), "title": "Pin swap needed: steering on bus 0", "t": NOW, "mono": mono, "boot": boot, "final": final}

  def test_swap_verdicts_alert(self):
    assert alert_text(self._result(Verdict.SWAP_NEEDED), self.DRIVE_START, BOOT) == "Pin swap needed: steering on bus 0"
    assert alert_text(self._result(Verdict.SWAP_LIKELY), self.DRIVE_START, BOOT) is not None

  def test_other_verdicts_do_not_alert(self):
    for v in (Verdict.OK_HDA2, Verdict.NON_HDA2, Verdict.INCONCLUSIVE):
      assert alert_text(self._result(v), self.DRIVE_START, BOOT) is None

  def test_a_provisional_swap_does_not_alert(self):
    # LFA from the ADAS ECU can show up before the camera's LKAS in the first seconds: only the settled verdict may alarm
    assert alert_text(self._result(Verdict.SWAP_LIKELY, final=False), self.DRIVE_START, BOOT) is None

  def test_a_result_from_an_earlier_drive_does_not_alert(self):
    assert alert_text(self._result(Verdict.SWAP_NEEDED, mono=100.0), self.DRIVE_START, BOOT) is None

  def test_a_result_from_before_a_reboot_does_not_alert_even_with_a_later_monotonic_time(self):
    # monotonic time restarts at boot, so an old result can carry a bigger number than this drive's start
    assert alert_text(self._result(Verdict.SWAP_NEEDED, mono=9_000.0, boot="boot-old"), self.DRIVE_START, BOOT) is None

  def test_missing_or_malformed_result(self):
    assert alert_text(None, self.DRIVE_START, BOOT) is None
    assert alert_text({}, self.DRIVE_START, BOOT) is None
    assert alert_text({"verdict": "swap_needed", "final": True}, self.DRIVE_START, BOOT) is None

  def test_settled_once_this_drives_result_is_final(self):
    assert settled(self._result(Verdict.OK_HDA2), self.DRIVE_START, BOOT)
    assert not settled(self._result(Verdict.OK_HDA2, final=False), self.DRIVE_START, BOOT)
    assert not settled(self._result(Verdict.OK_HDA2, mono=100.0), self.DRIVE_START, BOOT)
    assert not settled(None, self.DRIVE_START, BOOT) and not settled({}, self.DRIVE_START, BOOT)

  def test_boot_id_is_a_string(self):
    assert isinstance(boot_id(), str)   # "" where the kernel doesn't provide one (PC, Windows)


class TestSession:
  def test_writes_every_provisional_period_of_can_time(self):
    s = HarnessSession()
    feed(s.tally, 2, LKAS, 150, hz=100.0)           # 1.49 s of CAN time
    assert not s.due()
    feed(s.tally, 2, LKAS, 60, hz=100.0, start_s=1.5)
    assert s.due()
    r = s.result(HDA2_CAR, STAMP)
    assert r["verdict"] == Verdict.OK_HDA2 and r["final"] is False
    assert not s.due()                                # nothing new since the write

  def test_final_result_stops_the_session(self):
    s = HarnessSession()
    car_traffic(s.tally)
    feed(s.tally, 2, LKAS, int(FINAL_S * 100) + 1, hz=100.0)
    r = s.result(HDA2_CAR, STAMP)
    assert r["final"] is True
    s.finish(r)
    assert s.final and not s.due()

  def test_a_failed_write_is_retried_next_period(self):
    s = HarnessSession()
    feed(s.tally, 2, LKAS, int(FINAL_S * 100) + 1, hz=100.0)
    r = s.result(HDA2_CAR, STAMP)                       # params.put failed, so finish() is never called
    assert r["final"] and not s.final
    feed(s.tally, 2, LKAS, 300, hz=100.0, start_s=FINAL_S + 0.1)
    assert s.due()

  def test_changed_reports_verdict_transitions(self):
    s = HarnessSession()
    car_traffic(s.tally)
    feed(s.tally, 0, LFA, 150, hz=100.0)
    feed(s.tally, 0, OTHER, 250, hz=100.0)
    first = s.result(None, STAMP)
    assert first["verdict"] == Verdict.SWAP_LIKELY and s.changed
    feed(s.tally, 0, OTHER, 250, hz=100.0, start_s=2.5)
    s.result(None, STAMP)
    assert not s.changed
    feed(s.tally, 2, LKAS, 300, hz=100.0, start_s=5.0)
    assert s.result(None, STAMP)["verdict"] == Verdict.OK_HDA2 and s.changed


class TestResultContract:
  KEYS = {"verdict", "title", "detail", "action", "camera_bus", "can_fd", "seconds", "t", "mono", "boot", "buses", "decision", "disagrees",
          "radar", "final"}

  @pytest.mark.parametrize("setup", ["ok", "swap", "non_hda2", "likely", "nothing", "empty", "classic"])
  def test_every_verdict_has_the_full_shape_and_is_json(self, setup):
    t = HarnessTally()
    if setup != "empty":
      car_traffic(t)
    feed_for = {"ok": (2, LKAS), "swap": (0, LKAS), "non_hda2": (2, LFA), "likely": (1, LFA)}
    if setup in feed_for:
      feed(t, *feed_for[setup], 200)
    if setup == "classic":
      t = HarnessTally()
      feed(t, 0, OTHER, 100, length=CLASSIC_LEN)
    r = check(t, HDA2_CAR)
    assert set(r) == self.KEYS
    assert r["final"] is False and (r["t"], r["mono"], r["boot"]) == (NOW, MONO, BOOT)
    assert all(isinstance(r[k], str) and r[k] for k in ("title", "detail", "action"))
    assert json.loads(json.dumps(r))["verdict"] == r["verdict"]
