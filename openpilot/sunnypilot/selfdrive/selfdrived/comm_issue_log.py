"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Keeps the last communication issues selfdrived raised, with what the device was doing at the time, so they can be read
in Settings > Developer without SSH. selfdrived already logs them to swaglog; this saves the same lists to a param along
with model timing, CPU, GPU and temperatures, and for each late stream how fast it actually arrived.
"""
import time
from collections import deque

from openpilot.common.params import Params

PARAM = "CommIssueLog"
MAX_ENTRIES = 20


def _rate(sm, service: str) -> dict:
  tracker = sm.freq_tracker.get(service) if hasattr(sm, "freq_tracker") else None
  if tracker is None or tracker.avg_dt.count == 0:
    return {"service": service}
  avg, recent = tracker.avg_dt.get_average(), tracker.recent_avg_dt.get_average()
  return {"service": service, "hz": round(1.0 / avg, 1) if avg > 0 else 0.0, "recent_hz": round(1.0 / recent, 1) if recent > 0 else 0.0,
          "min_hz": round(tracker.min_freq, 1), "max_hz": round(tracker.max_freq, 1)}


def checked(sm, logs: dict) -> dict:
  """The failing services selfdrived actually holds against the car. Its lists name every subscribed service, including the ones
  it ignores (alertDebug and lateralManeuverPlan only come from the maneuver tools), which would point the reader the wrong way."""
  ignore_alive, ignore_valid = getattr(sm, "ignore_alive", []), getattr(sm, "ignore_valid", [])
  check_freq = getattr(sm, "_check_avg_freq", lambda s: True)
  return {
    "not_alive": [s for s in logs["not_alive"] if s not in ignore_alive],
    "invalid": [s for s in logs["invalid"] if s not in ignore_valid],
    "not_freq_ok": [s for s in logs["not_freq_ok"] if check_freq(s)],
  }


def snapshot(sm, logs: dict) -> dict:
  logs = checked(sm, logs)
  if logs["not_alive"]:   # in the order selfdrived picks the alert
    kind = "commIssue (a process stopped sending)"
  elif logs["not_freq_ok"]:
    kind = "commIssueAvgFreq (low communication rate)"
  else:
    kind = "commIssue (invalid data)"
  model, device = sm['modelV2'], sm['deviceState']
  return {
    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    "kind": kind,
    "not_alive": list(logs["not_alive"]),
    "invalid": list(logs["invalid"]),
    "late": [_rate(sm, s) for s in logs["not_freq_ok"]],
    "model_ms": round(model.modelExecutionTime * 1000, 1),
    "frame_drop_pct": round(model.frameDropPerc, 1),
    "cpu_pct": [int(c) for c in device.cpuUsagePercent],
    "gpu_pct": int(device.gpuUsagePercent),
    "max_temp_c": round(device.maxTempC, 1),
    "thermal": str(device.thermalStatus),
  }


class CommIssueLog:
  def __init__(self, params: Params):
    self._params = params
    self._entries: deque = deque(params.get(PARAM) or [], maxlen=MAX_ENTRIES)

  def record(self, sm, logs: dict) -> None:
    """Called by selfdrived when the set of failing services changes. Never lets a logging problem reach selfdrived."""
    if not any(checked(sm, logs).values()):
      return
    try:
      entry = snapshot(sm, logs)
      if self._entries and self._entries[0].get("time") == entry["time"] and self._entries[0].get("kind") == entry["kind"]:
        self._entries.popleft()   # the list changed again within the second: one entry, the latest
      self._entries.appendleft(entry)
      self._params.put(PARAM, list(self._entries))  # non-blocking
    except Exception:
      pass
