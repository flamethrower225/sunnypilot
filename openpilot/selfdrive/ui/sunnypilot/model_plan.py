"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The model's plan as numpy arrays, converted once per model frame. Reading a capnp list into numpy walks it element by element in
Python, and the curve preview, the stop detector, the radar overlays and the communication issue watch all want the same arrays.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelPlan:
  x: np.ndarray          # m ahead of the camera
  y: np.ndarray          # m, right positive
  v: np.ndarray          # m/s
  yaw_rate: np.ndarray   # rad/s
  lat_acc: np.ndarray    # m/s^2, |yaw rate * speed|, the predicted lateral acceleration the vision speed controller uses

  @property
  def size(self) -> int:
    return int(self.x.size)


EMPTY_PLAN = ModelPlan(*(np.zeros(0) for _ in range(5)))


def plan_from_model(model) -> ModelPlan:
  """Trims every series to the shortest one, so a custom model with an odd length never indexes out of range."""
  x, y = np.array(model.position.x, dtype=float), np.array(model.position.y, dtype=float)
  v, yaw = np.array(model.velocity.x, dtype=float), np.array(model.orientationRate.z, dtype=float)
  n = min(x.size, v.size, yaw.size)
  y = y[:n] if y.size >= n else np.concatenate((y, np.zeros(n - y.size)))
  return ModelPlan(x[:n], y, v[:n], yaw[:n], np.abs(yaw[:n] * v[:n]))


_cached_model = None
_cache: ModelPlan = EMPTY_PLAN


def model_plan(sm, started_frame: int) -> ModelPlan:
  """The plan from the latest modelV2, or an empty one before the first model frame of this drive. The SubMaster hands out a new
  reader for every message it receives, so the reader itself tells whether the model has moved on; holding on to it keeps its id
  from being reused."""
  global _cached_model, _cache
  if sm.recv_frame['modelV2'] < started_frame:
    return EMPTY_PLAN
  model = sm['modelV2']
  if model is not _cached_model:
    _cached_model, _cache = model, plan_from_model(model)
  return _cache
