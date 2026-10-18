"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The object detector: YOLOX-nano (Megvii, Apache-2.0, see models/README.md), trained on COCO, which knows cars, trucks, buses,
motorcycles and traffic lights. On the device it runs from the tinygrad program the build compiles from models/scene_detector.onnx,
on the GPU next to the driving model; on a PC it can also run straight from the ONNX file.
"""
import pickle
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

from openpilot.common.file_chunker import get_manifest_path
from openpilot.sunnypilot.selfdrive.scened.frames import INPUT_SIZE, Region

MODELS_DIR = Path(__file__).resolve().parent / "models"
ONNX_PATH = MODELS_DIR / "scene_detector.onnx"
PKL_PATH = MODELS_DIR / "scene_detector_tinygrad.pkl"

STRIDES = (8, 16, 32)
NUM_CLASSES = 80
SCORE_MIN = 0.3
NMS_IOU = 0.45
MAX_DETECTIONS = 40


class ObjectClass(IntEnum):
  # COCO ids
  CAR = 2
  MOTORCYCLE = 3
  BUS = 5
  TRUCK = 7
  TRAFFIC_LIGHT = 9


VEHICLE_CLASSES = frozenset({ObjectClass.CAR, ObjectClass.MOTORCYCLE, ObjectClass.BUS, ObjectClass.TRUCK})
KEPT_CLASSES = np.array(sorted(int(c) for c in ObjectClass))
LIGHT_SCORE_MIN = 0.25   # lights are small and the model is less sure of them; the color check and tracking sort out the rest


@dataclass(frozen=True)
class Detection:
  x0: float   # source pixels
  y0: float
  x1: float
  y1: float
  score: float
  cls: ObjectClass

  @property
  def width(self) -> float:
    return self.x1 - self.x0

  @property
  def height(self) -> float:
    return self.y1 - self.y0


def _grid() -> tuple[np.ndarray, np.ndarray]:
  grids, strides = [], []
  for s in STRIDES:
    n = INPUT_SIZE // s
    yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    grids.append(np.stack((xv, yv), -1).reshape(-1, 2))
    strides.append(np.full((n * n, 1), s))
  return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


GRID, GRID_STRIDES = _grid()


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = NMS_IOU) -> list[int]:
  order = scores.argsort()[::-1]
  areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
  keep = []
  while order.size:
    i = order[0]
    keep.append(int(i))
    rest = order[1:]
    w = np.clip(np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]), 0, None)
    h = np.clip(np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]), 0, None)
    inter = w * h
    order = rest[inter / np.maximum(areas[i] + areas[rest] - inter, 1e-6) <= iou]
  return keep


def decode(raw: np.ndarray, region: Region) -> list[Detection]:
  """YOLOX output (1, 3549, 85): box centers and sizes on the grid, objectness, class scores, already sigmoided. Boxes come back in
  source pixels of the region's frame; only the classes the scene detector uses, each class suppressed on its own."""
  out = raw.reshape(-1, 5 + NUM_CLASSES)
  scores = out[:, 4:5] * out[:, 5 + KEPT_CLASSES]
  best = scores.argmax(1)
  score = scores[np.arange(len(best)), best]
  cls = KEPT_CLASSES[best]
  keep = score >= np.where(cls == ObjectClass.TRAFFIC_LIGHT, LIGHT_SCORE_MIN, SCORE_MIN)
  if not keep.any():
    return []
  xy = (out[keep, :2] + GRID[keep]) * GRID_STRIDES[keep]
  wh = np.exp(np.minimum(out[keep, 2:4], 10.0)) * GRID_STRIDES[keep]
  r = region.scale
  boxes = np.concatenate((xy - wh / 2, xy + wh / 2), axis=1) / r + [region.x, region.y, region.x, region.y]
  score, cls = score[keep], cls[keep]

  dets: list[Detection] = []
  for c in np.unique(cls):
    idx = np.flatnonzero(cls == c)
    for i in nms(boxes[idx], score[idx]):
      b = boxes[idx[i]]
      dets.append(Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(score[idx[i]]), ObjectClass(int(c))))
  dets.sort(key=lambda d: -d.score)
  return dets[:MAX_DETECTIONS]


class Detector:
  """Runs the network. Prefers the compiled tinygrad program (the device build makes it); without one it runs the ONNX file through
  tinygrad, which is slower to start but needs no build."""

  def __init__(self, pkl_path: Path = PKL_PATH, onnx_path: Path = ONNX_PATH):
    from tinygrad.tensor import Tensor
    self._tensor = Tensor
    self.compiled = pkl_path.exists() or Path(get_manifest_path(str(pkl_path))).exists()
    if self.compiled:
      from openpilot.common.file_chunker import open_file_chunked
      self._run = pickle.load(open_file_chunked(str(pkl_path)))
    else:
      from tinygrad.device import Device
      from tinygrad.engine.jit import TinyJit
      from tinygrad.nn.onnx import OnnxRunner
      runner = OnnxRunner(str(onnx_path))

      @TinyJit
      def run(images):
        return next(iter(runner({"images": images.to(Device.DEFAULT)}).values())).cast("float32").realize()
      self._run = run

  def __call__(self, image: np.ndarray, region: Region) -> list[Detection]:
    images = self._tensor(np.ascontiguousarray(image, dtype=np.float32), device="NPY")
    raw = self._run(images=images).numpy()
    return decode(raw, region)
