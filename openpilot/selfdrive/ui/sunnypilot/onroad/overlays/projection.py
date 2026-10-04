"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np
import pyray as rl


class RoadProjector:
  """Projects points in the car frame onto the road camera image with the ModelRenderer's current transform.

  Car frame: x forward from the front of the car (m, same origin as radarState dRel), y to the right (m),
  height above the road surface (m). radarState/radarTracks yRel is left positive, so pass -yRel.
  """

  def __init__(self, renderer):
    self._r = renderer

  @property
  def rect(self) -> rl.Rectangle:
    return self._r._rect

  @property
  def ready(self) -> bool:
    return self._r._path.raw_points.shape[0] > 1

  @property
  def path_length(self) -> float:
    """Farthest x (m) the model path reaches."""
    pts = self._r._path.raw_points
    return float(pts[-1, 0]) if pts.shape[0] else 0.0

  def ground_z(self, x):
    """Camera-frame z (down) of the road surface at x, following the model path over hills."""
    pts = self._r._path.raw_points
    if pts.shape[0] < 2:
      return np.zeros_like(np.asarray(x, dtype=float)) + self._r._path_offset_z
    return np.interp(x, pts[:, 0], pts[:, 2]) + self._r._path_offset_z

  def path_y(self, x):
    """Lateral position (m, right positive) of the model's planned path at x."""
    pts = self._r._path.raw_points
    if pts.shape[0] < 2:
      return np.zeros_like(np.asarray(x, dtype=float))
    return np.interp(x, pts[:, 0], pts[:, 1]) - self._r._camera_offset

  def to_screen(self, x: float, y: float, height: float = 0.0) -> tuple[float, float] | None:
    """Screen position of a point, or None when it falls outside the drawable region."""
    if x <= 0.0:
      return None
    return self._r._map_to_screen(x, y + self._r._camera_offset, float(self.ground_z(x)) - height)

  def ribbon(self, xs, ys, half_width: float, height: float = 0.0) -> np.ndarray:
    """Polygon for shader_polygon.draw_polygon: a band half_width either side of the polyline (xs, ys) on the road.
    xs must be increasing. Returns an empty array when nothing is visible."""
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    if xs.shape[0] < 2:
      return np.empty((0, 2), dtype=np.float32)
    pts = self._r._path.raw_points
    raw_z = np.interp(xs, pts[:, 0], pts[:, 2]) if pts.shape[0] > 1 else np.zeros_like(xs)
    line = np.column_stack([xs, ys + self._r._camera_offset, raw_z - height]).astype(np.float32)
    return self._r._map_line_to_polygon(line, half_width, self._r._path_offset_z, len(xs) - 1, float(xs[-1]))
