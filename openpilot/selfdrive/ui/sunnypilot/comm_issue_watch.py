"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The UI's side of a communication issue. selfdrived records which streams were late (CommIssueLog); the UI records what
it was doing at that moment: how long its frames took, whether the model saw a curve ahead (when the curve preview draws
the most), whether vision curve speed control was active, which onroad overlays were on, how long the driver monitoring
model took, and what Camera Object Detection (which shares the GPU with both models) was doing.
"""
import time
from collections import deque

import numpy as np

PARAM = "CommIssueUiLog"
MAX_ENTRIES = 20
WINDOW_S = 6.0           # look back this far when the alert appears
CURVE_LAT_ACC = 1.6      # m/s^2 predicted lateral acceleration that counts as a curve ahead (the curve preview's threshold)
OVERLAY_FLAGS = ("curve_hologram", "stop_detection", "detection_log", "radar_scope", "radar_track_brackets", "threat_board",
                 "follow_ghost", "lead_braking_wake", "disengage_horizon", "driver_flags", "hard_brake_flash", "lead_gap_history",
                 "pace_border", "drive_notifications", "lead_gap_badge")  # every HUD overlay toggle (the badge is on when its style isn't Off)
TOOL_ONLY = ("alertDebug", "lateralManeuverPlan")  # only the maneuver tools send these and selfdrived ignores them; older entries list them
DM_SLOW_MS = 38.0        # a driver monitoring run longer than this leaves no margin at 20 Hz


class CommIssueWatch:
  def __init__(self):
    self._samples: deque[tuple[float, float, float, bool]] = deque()  # (mono time, frame ms, max lateral accel, SCC-V active)
    self._dm: deque[tuple[float, float]] = deque()   # (mono time, driver monitoring model ms)
    self._last_mono: float | None = None
    self._lat_acc = 0.0
    self._alert_prev = False

  def update(self, ui_state) -> None:
    now = time.monotonic()
    frame_ms = (now - self._last_mono) * 1000 if self._last_mono is not None else 0.0
    self._last_mono = now
    if not ui_state.started:
      self._samples.clear()
      self._alert_prev = False
      return

    sm = ui_state.sm
    if sm.updated['modelV2']:
      model = sm['modelV2']
      yaw, vx = np.asarray(model.orientationRate.z), np.asarray(model.velocity.x)
      n = min(len(yaw), len(vx))
      self._lat_acc = float(np.max(np.abs(yaw[:n] * vx[:n]))) if n else 0.0
    scc_v = bool(sm['longitudinalPlanSP'].smartCruiseControl.vision.active)
    self._samples.append((now, frame_ms, self._lat_acc, scc_v))
    while self._samples and self._samples[0][0] < now - WINDOW_S:
      self._samples.popleft()
    if sm.updated.get('driverStateV2', False):
      self._dm.append((now, float(sm['driverStateV2'].modelExecutionTime) * 1000))
    while self._dm and self._dm[0][0] < now - WINDOW_S:
      self._dm.popleft()

    alert = str(sm['selfdriveState'].alertType)
    is_comm = alert.startswith("commIssue")
    if is_comm and not self._alert_prev:
      self._record(ui_state, alert)
    self._alert_prev = is_comm

  def _record(self, ui_state, alert: str) -> None:
    frames = [s[1] for s in self._samples if s[1] > 0]
    entry = {
      "time": time.strftime("%Y-%m-%d %H:%M:%S"),
      "alert": alert,
      "ui_frame_avg_ms": round(float(np.mean(frames)), 1) if frames else 0.0,
      "ui_frame_max_ms": round(max(frames), 1) if frames else 0.0,
      "max_lat_acc": round(max((s[2] for s in self._samples), default=0.0), 2),
      "curve_ahead": any(s[2] >= CURVE_LAT_ACC for s in self._samples),
      "scc_v_active": any(s[3] for s in self._samples),
      "overlays": [f for f in OVERLAY_FLAGS if getattr(ui_state, f, False)],
      "dm_ms_avg": round(float(np.mean([d for _, d in self._dm])), 1) if self._dm else 0.0,
      "dm_ms_max": round(max(d for _, d in self._dm), 1) if self._dm else 0.0,
      "scened": scened_state(ui_state),
    }
    try:
      entries = ([entry] + list(ui_state.params.get(PARAM) or []))[:MAX_ENTRIES]
      ui_state.params.put(PARAM, entries)  # non-blocking
    except Exception:
      pass


def scened_state(ui_state) -> dict | None:
  """What Camera Object Detection was doing, or None when it's off."""
  if not getattr(ui_state, "scene_detection", False):
    return None
  sm = ui_state.sm
  if sm.recv_frame.get('sceneDetectionSP', -1) < ui_state.started_frame:
    return {"running": False}
  sd = sm['sceneDetectionSP']
  return {"running": True, "pass_rate": round(float(sd.passRate), 1), "pass_ms": round(float(sd.executionTime) * 1000, 1),
          "throttled": bool(sd.throttled)}


# *** viewer ***

def _safe(text) -> str:
  # the settings viewer parses < as a tag and shows entities literally
  return str(text).replace("&", "＆").replace("<", "‹").replace(">", "›")


def _parse(t: str) -> float:
  try:
    return time.mktime(time.strptime(t, "%Y-%m-%d %H:%M:%S"))
  except (TypeError, ValueError):
    return 0.0


def _hints(sd: dict, ui: dict | None) -> list[str]:
  hints = []
  late = {r.get("service") for r in sd.get("late", [])}
  failing = late | set(sd.get("invalid", [])) | set(sd.get("not_alive", []))
  dm_failing = bool(failing & {"driverMonitoringState", "driverStateV2"})
  if dm_failing:
    slow = f" (its model took up to {ui['dm_ms_max']} ms)" if ui and ui.get("dm_ms_max", 0) > DM_SLOW_MS else ""
    hints.append(f"driver monitoring fell behind{slow}: its model shares a CPU core with the driving model, and the GPU with it")
  scened = ui.get("scened") if ui else None
  if scened and (dm_failing or "modelV2" in failing):
    hints.append("Camera Object Detection was running, and it shares the GPU with both models. It now steps aside when they slow " +
                 "down; if this alert comes back, turn it off in Visuals and see whether it stops")
  if sd.get("thermal", "") not in ("ok", ""):
    hints.append(f"device thermal state is {sd['thermal']}: heat is throttling the CPU")
  if "modelV2" in late or sd.get("frame_drop_pct", 0) > 5:
    hints.append("the driving model fell behind (modelV2 late or frames dropped)")
  if sd.get("gpu_pct", 0) >= 90:
    hints.append("the GPU was nearly full; the model and the UI share it")
  if ui and ui.get("ui_frame_max_ms", 0) > 100:
    hints.append(f"UI frames took up to {ui['ui_frame_max_ms']} ms (50 ms is normal at 20 fps)")
  if ui and ui.get("curve_ahead") and "curve_hologram" in ui.get("overlays", []):
    hints.append("a curve was ahead with Curve Preview on: turn it off and see if the alert stops")
  if ui and ui.get("scc_v_active"):
    hints.append("vision curve speed control (SCC-V) was active")
  return hints


def comm_issues_html(sd_entries: list | None, ui_entries: list | None) -> str:
  sd_entries, ui_entries = sd_entries or [], ui_entries or []
  if not sd_entries and not ui_entries:
    return " ".join(["<h2>No communication issues recorded</h2><p>When openpilot shows Communication Issue or Low Communication",
                     "Rate Between Processes, the streams that were late and what the device and UI were doing are saved here.</p>"])
  parts = ["<h2>Communication issues</h2><p>Newest first. Allowed rates are what openpilot accepts before raising the alert.</p>"]
  for sd in sd_entries:
    t = _parse(sd.get("time", ""))
    ui = min((u for u in ui_entries if abs(_parse(u.get("time", "")) - t) <= 5), key=lambda u: abs(_parse(u["time"]) - t), default=None)
    parts.append(f"<h3>{_safe(sd.get('time', ''))}</h3><p>{_safe(sd.get('kind', ''))}</p>")
    lines = []
    for r in sd.get("late", []):
      if r.get("service") in TOOL_ONLY:
        continue
      if "hz" in r:
        lines.append(f"late: {_safe(r['service'])} at {r['hz']} Hz (recent {r['recent_hz']} Hz), allowed {r['min_hz']}-{r['max_hz']} Hz")
      else:
        lines.append(f"late: {_safe(r['service'])}")
    lines += [f"stopped: {_safe(s)}" for s in sd.get("not_alive", []) if s not in TOOL_ONLY]
    lines += [f"invalid: {_safe(s)}" for s in sd.get("invalid", []) if s not in TOOL_ONLY]
    cpu = sd.get("cpu_pct", [])
    lines.append(f"model: {sd.get('model_ms', 0)} ms per run, {sd.get('frame_drop_pct', 0)}% frames dropped")
    cpu_text = f"CPU max {max(cpu) if cpu else 0}% ({', '.join(map(str, cpu))})"
    lines.append(f"device: {cpu_text}, GPU {sd.get('gpu_pct', 0)}%, {sd.get('max_temp_c', 0)} °C, thermal {_safe(sd.get('thermal', ''))}")
    if ui:
      curve = f"curve ahead: {'yes' if ui['curve_ahead'] else 'no'} (peak {ui['max_lat_acc']} m/s2)"
      frames = f"frames {ui['ui_frame_avg_ms']} ms average, {ui['ui_frame_max_ms']} ms worst"
      lines.append(f"UI: {frames}; {curve}; SCC-V: {'active' if ui['scc_v_active'] else 'off'}")
      lines.append("overlays on: " + (", ".join(ui["overlays"]) or "none"))
      if ui.get("dm_ms_max"):
        lines.append(f"driver monitoring model: {ui['dm_ms_avg']} ms average, {ui['dm_ms_max']} ms worst")
      scened = ui.get("scened")
      if scened is not None:
        if scened.get("running"):
          held = ", held back" if scened.get("throttled") else ""
          lines.append(f"Camera Object Detection: {scened['pass_rate']} passes/s, {scened['pass_ms']} ms per pass{held}")
        else:
          lines.append("Camera Object Detection: on, not running yet")
    parts.append("<ul>" + "".join(f"<li>{line}</li>" for line in lines) + "</ul>")
    hints = _hints(sd, ui)
    if hints:
      parts.append("<p>Points to:</p><ul>" + "".join(f"<li>{_safe(h)}</li>" for h in hints) + "</ul>")
  return "".join(parts)
