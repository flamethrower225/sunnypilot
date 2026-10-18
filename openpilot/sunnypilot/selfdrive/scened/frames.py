"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Camera frames to detector input. The cameras hand out NV12 buffers; the detector wants a 416 x 416 BGR image. A region of the frame
is sampled straight out of the buffer with index arrays built once per region (nearest neighbour, which is all a detector this size
can use), so a frame costs a gather and a color conversion, not a resize of the whole picture.
"""
from dataclasses import dataclass

import numpy as np

INPUT_SIZE = 416
PAD_VALUE = 114.0    # YOLOX letterboxes with this gray


@dataclass(frozen=True)
class Region:
  """A rectangle of the camera frame, in source pixels."""
  x: int
  y: int
  w: int
  h: int

  @property
  def scale(self) -> float:
    """Detector pixels per source pixel. The region is letterboxed: its long side fills the input."""
    return INPUT_SIZE / max(self.w, self.h)


def clamp_region(x: float, y: float, w: int, h: int, frame_w: int, frame_h: int) -> Region:
  """A w x h region at (x, y), moved to lie inside the frame (and shrunk if the frame is smaller)."""
  w, h = min(w, frame_w), min(h, frame_h)
  return Region(int(min(max(x, 0), frame_w - w)), int(min(max(y, 0), frame_h - h)), w, h)


def nv12_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
  """BT.601 limited range, uint8 in, float32 RGB (0-255) out, any shape."""
  yf = (y.astype(np.float32) - 16.0) * 1.164
  uf = u.astype(np.float32) - 128.0
  vf = v.astype(np.float32) - 128.0
  rgb = np.stack((yf + 1.596 * vf, yf - 0.392 * uf - 0.813 * vf, yf + 2.017 * uf), axis=-1)
  return np.clip(rgb, 0.0, 255.0)


class NV12Sampler:
  """Samples regions of NV12 frames of one size. Index arrays are cached per region."""

  def __init__(self, width: int, height: int, stride: int, uv_offset: int):
    self.width, self.height = width, height
    self.stride, self.uv_offset = stride, uv_offset
    self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

  def _indices(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_idx = rows[:, None] * self.stride + cols[None, :]
    uv_idx = self.uv_offset + (rows[:, None] // 2) * self.stride + (cols[None, :] // 2) * 2
    return y_idx.astype(np.int64), uv_idx.astype(np.int64)

  def sample(self, buf: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """RGB float32 (len(rows), len(cols), 3) at the given source pixel rows and columns."""
    y_idx, uv_idx = self._indices(rows, cols)
    return nv12_to_rgb(buf[y_idx], buf[uv_idx], buf[uv_idx + 1])

  def detector_input(self, buf: np.ndarray, region: Region) -> np.ndarray:
    """(1, 3, 416, 416) float32 BGR, the region letterboxed into the top left like YOLOX's own preprocessing."""
    key = (region.x, region.y, region.w, region.h)
    cached = self._cache.get(key)
    if cached is None:
      r = region.scale
      out_w, out_h = int(region.w * r), int(region.h * r)
      cols = np.minimum(region.x + (np.arange(out_w) / r).astype(np.int64), self.width - 1)
      rows = np.minimum(region.y + (np.arange(out_h) / r).astype(np.int64), self.height - 1)
      cached = self._cache[key] = self._indices(rows, cols)
    y_idx, uv_idx = cached
    rgb = nv12_to_rgb(buf[y_idx], buf[uv_idx], buf[uv_idx + 1])
    out = np.full((1, 3, INPUT_SIZE, INPUT_SIZE), PAD_VALUE, dtype=np.float32)
    h, w = rgb.shape[:2]
    out[0, :, :h, :w] = rgb[..., ::-1].transpose(2, 0, 1)
    return out

  def crop_rgb(self, buf: np.ndarray, x0: float, y0: float, x1: float, y1: float, max_side: int = 48) -> np.ndarray:
    """A small box at (close to) full resolution, for looking at a traffic light's lamps."""
    x0, y0 = int(max(x0, 0)), int(max(y0, 0))
    x1, y1 = int(min(x1, self.width)), int(min(y1, self.height))
    if x1 - x0 < 2 or y1 - y0 < 2:
      return np.zeros((0, 0, 3), np.float32)
    step = max(1, int(np.ceil(max(x1 - x0, y1 - y0) / max_side)))
    return self.sample(buf, np.arange(y0, y1, step), np.arange(x0, x1, step))


def rgb_to_nv12(rgb: np.ndarray, stride: int, y_height: int, uv_height: int) -> np.ndarray:
  """The inverse, for tests and offline replay of images: an NV12 buffer laid out like camerad's."""
  rgb = rgb.astype(np.float32)
  r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
  y = 16 + 0.257 * r + 0.504 * g + 0.098 * b
  u = 128 - 0.148 * r - 0.291 * g + 0.439 * b
  v = 128 + 0.439 * r - 0.368 * g - 0.071 * b
  h, w = y.shape
  buf = np.zeros(stride * y_height + stride * uv_height, dtype=np.uint8)
  ybuf = buf[:stride * y_height].reshape(y_height, stride)
  ybuf[:h, :w] = np.clip(y, 0, 255)
  uvbuf = buf[stride * y_height:].reshape(uv_height, stride)
  uvbuf[:h // 2, 0:w:2] = np.clip(u[0::2, 0::2], 0, 255)
  uvbuf[:h // 2, 1:w:2] = np.clip(v[0::2, 0::2], 0, 255)
  return buf
