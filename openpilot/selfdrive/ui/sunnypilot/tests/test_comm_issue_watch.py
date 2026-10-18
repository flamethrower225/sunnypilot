from types import SimpleNamespace
from unittest import mock

from openpilot.cereal import messaging
from openpilot.selfdrive.ui.sunnypilot import comm_issue_watch as cw


class FakeParams:
  def __init__(self):
    self.data = {}

  def get(self, key):
    return self.data.get(key)

  def put(self, key, value):
    self.data[key] = value


class FakeSM:
  def __init__(self):
    self.updated = {'modelV2': False}
    self.data = {n: getattr(messaging.new_message(n), n) for n in ('modelV2', 'longitudinalPlanSP', 'selfdriveState')}

  def __getitem__(self, name):
    return self.data[name]

  def set(self, lat_acc=0.0, scc_v=False, alert=""):
    model = messaging.new_message('modelV2')
    model.modelV2.velocity.x = [20.0] * 33
    model.modelV2.orientationRate.z = [lat_acc / 20.0] * 33
    sp = messaging.new_message('longitudinalPlanSP')
    sp.longitudinalPlanSP.smartCruiseControl.vision.active = scc_v
    ss = messaging.new_message('selfdriveState')
    ss.selfdriveState.alertType = alert
    self.data.update(modelV2=model.modelV2, longitudinalPlanSP=sp.longitudinalPlanSP, selfdriveState=ss.selfdriveState)
    self.updated['modelV2'] = True


def run(watch, ui, sm, frames, frame_s, **kw):
  clock = [100.0]
  for _ in range(frames):
    sm.set(**kw)
    clock[0] += frame_s
    with mock.patch.object(cw.time, "monotonic", return_value=clock[0]):
      watch.update(ui)
  return clock


def test_records_ui_context_once_per_alert():
  sm, params = FakeSM(), FakeParams()
  ui = SimpleNamespace(started=True, sm=sm, params=params, curve_hologram=True, stop_detection=True)
  watch = cw.CommIssueWatch()
  run(watch, ui, sm, 40, 0.05)  # straight road, 20 fps
  run(watch, ui, sm, 20, 0.15, lat_acc=2.4, scc_v=True)  # curve ahead, frames slow down
  run(watch, ui, sm, 10, 0.15, lat_acc=2.4, scc_v=True, alert="commIssueAvgFreq/softDisable")
  (entry,) = params.data[cw.PARAM]  # one entry for the rising edge, not one per frame
  assert entry["alert"] == "commIssueAvgFreq/softDisable"
  assert entry["curve_ahead"] and entry["max_lat_acc"] == 2.4 and entry["scc_v_active"]
  assert entry["ui_frame_max_ms"] >= 149 and entry["overlays"] == ["curve_hologram", "stop_detection"]


def test_offroad_and_other_alerts_record_nothing():
  sm, params = FakeSM(), FakeParams()
  ui = SimpleNamespace(started=False, sm=sm, params=params)
  watch = cw.CommIssueWatch()
  run(watch, ui, sm, 5, 0.05, alert="commIssue/softDisable")
  ui.started = True
  run(watch, ui, sm, 5, 0.05, alert="steerSaturated/userPrompt")
  assert cw.PARAM not in params.data


def test_viewer_pairs_records_and_points_at_the_curve_preview():
  sd = [{"time": "2026-10-18 10:00:02", "kind": "commIssueAvgFreq (low communication rate)", "not_alive": [], "invalid": [],
         "late": [{"service": "modelV2", "hz": 14.3, "recent_hz": 12.0, "min_hz": 16.0, "max_hz": 24.0}],
         "model_ms": 72.0, "frame_drop_pct": 12.5, "cpu_pct": [40, 95], "gpu_pct": 97, "max_temp_c": 71.5, "thermal": "ok"}]
  ui = [{"time": "2026-10-18 10:00:00", "alert": "commIssueAvgFreq/softDisable", "ui_frame_avg_ms": 90.0, "ui_frame_max_ms": 150.0,
         "max_lat_acc": 2.4, "curve_ahead": True, "scc_v_active": False, "overlays": ["curve_hologram"]},
        {"time": "2026-10-18 09:00:00", "alert": "x", "ui_frame_avg_ms": 1, "ui_frame_max_ms": 1, "max_lat_acc": 0, "curve_ahead": False,
         "scc_v_active": False, "overlays": []}]
  html = cw.comm_issues_html(sd, ui)
  assert "late: modelV2 at 14.3 Hz (recent 12.0 Hz), allowed 16.0-24.0 Hz" in html
  assert "curve ahead: yes" in html and "overlays on: curve_hologram" in html
  for hint in ("driving model fell behind", "GPU was nearly full", "UI frames took up to 150.0 ms", "Curve Preview on"):
    assert hint in html
  assert "thermal state" not in html  # ok is not a heat problem
  hot = cw.comm_issues_html([dict(sd[0], thermal="overheated")], [])
  assert "device thermal state is overheated" in hot
  assert "No communication issues recorded" in cw.comm_issues_html(None, None)
