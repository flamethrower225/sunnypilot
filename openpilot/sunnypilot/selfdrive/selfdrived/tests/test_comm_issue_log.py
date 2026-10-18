from unittest import mock

from openpilot.cereal import messaging
from openpilot.cereal.messaging import FrequencyTracker
from openpilot.sunnypilot.selfdrive.selfdrived.comm_issue_log import MAX_ENTRIES, PARAM, CommIssueLog


class FakeParams:
  def __init__(self, stored=None):
    self.data = {PARAM: stored} if stored is not None else {}

  def get(self, key):
    return self.data.get(key)

  def put(self, key, value):
    self.data[key] = value


class FakeSM:
  def __init__(self):
    model = messaging.new_message('modelV2')
    model.modelV2.modelExecutionTime = 0.072
    model.modelV2.frameDropPerc = 12.5
    device = messaging.new_message('deviceState')
    device.deviceState.cpuUsagePercent = [40, 95, 60, 30]
    device.deviceState.gpuUsagePercent = 97
    device.deviceState.maxTempC = 71.5
    device.deviceState.thermalStatus = "overheated"
    self.data = {'modelV2': model.modelV2, 'deviceState': device.deviceState}
    tracker = FrequencyTracker(20., 100., False)  # modelV2: 20 Hz service read by a 100 Hz loop
    for i in range(200):
      tracker.record_recv_time(i * 0.07)  # arriving at ~14 Hz
    self.freq_tracker = {'modelV2': tracker}

  def __getitem__(self, name):
    return self.data[name]


def test_records_late_streams_with_rates_and_context():
  params = FakeParams()
  log = CommIssueLog(params)
  log.record(FakeSM(), {'invalid': [], 'not_alive': [], 'not_freq_ok': ['modelV2']})
  (entry,) = params.data[PARAM]
  assert entry["kind"].startswith("commIssueAvgFreq")
  (late,) = entry["late"]
  assert late["service"] == "modelV2" and abs(late["hz"] - 14.3) < 0.2 and late["min_hz"] == 16.0
  assert entry["model_ms"] == 72.0 and entry["frame_drop_pct"] == 12.5
  assert entry["cpu_pct"] == [40, 95, 60, 30] and entry["gpu_pct"] == 97 and entry["thermal"] == "overheated"


def test_stopped_process_and_history_limit():
  params = FakeParams(stored=[{"time": "old"}] * MAX_ENTRIES)
  log = CommIssueLog(params)
  log.record(FakeSM(), {'invalid': [], 'not_alive': ['radarState'], 'not_freq_ok': []})
  entries = params.data[PARAM]
  assert len(entries) == MAX_ENTRIES and entries[0]["kind"].startswith("commIssue (a process stopped")
  assert entries[0]["not_alive"] == ['radarState'] and entries[0]["late"] == []


def test_nothing_failing_records_nothing_and_errors_never_escape():
  params = FakeParams()
  log = CommIssueLog(params)
  log.record(FakeSM(), {'invalid': [], 'not_alive': [], 'not_freq_ok': []})
  assert PARAM not in params.data

  class Broken:
    freq_tracker: dict = {}

    def __getitem__(self, name):
      raise RuntimeError("boom")
  log.record(Broken(), {'invalid': ['carState'], 'not_alive': [], 'not_freq_ok': []})  # must not raise into selfdrived


def test_services_selfdrived_ignores_are_left_out():
  # what a drive on 2026-10-18 recorded: driver monitoring late, plus the maneuver tools' streams that never run on the road
  sm = FakeSM()
  ignored = ['alertDebug', 'lateralManeuverPlan']
  sm.ignore_alive = sm.ignore_valid = ignored
  sm._check_avg_freq = lambda s: s not in ignored
  params = FakeParams()
  log = CommIssueLog(params)
  log.record(sm, {'invalid': ['driverMonitoringState', 'alertDebug', 'lateralManeuverPlan'],
                  'not_alive': ['alertDebug', 'lateralManeuverPlan'],
                  'not_freq_ok': ['driverMonitoringState', 'alertDebug', 'lateralManeuverPlan']})
  (entry,) = params.data[PARAM]
  assert entry["kind"].startswith("commIssueAvgFreq")   # not "a process stopped sending": nothing that counts stopped
  assert entry["not_alive"] == [] and entry["invalid"] == ['driverMonitoringState']
  assert [r["service"] for r in entry["late"]] == ['driverMonitoringState']

  log.record(sm, {'invalid': ignored, 'not_alive': ignored, 'not_freq_ok': ignored})
  assert len(params.data[PARAM]) == 1   # only ignored services failing: nothing to record


def test_a_change_within_the_same_second_is_one_entry():
  params = FakeParams()
  log = CommIssueLog(params)
  with mock.patch("openpilot.sunnypilot.selfdrive.selfdrived.comm_issue_log.time.strftime", return_value="2026-10-18 20:55:51"):
    log.record(FakeSM(), {'invalid': [], 'not_alive': [], 'not_freq_ok': ['modelV2']})
    log.record(FakeSM(), {'invalid': ['modelV2'], 'not_alive': [], 'not_freq_ok': ['modelV2']})
  entries = params.data[PARAM]
  assert len(entries) == 1 and entries[0]["invalid"] == ['modelV2']
