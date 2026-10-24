"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np
import pyray as rl

from openpilot.system.ui.lib.shader_polygon import Gradient, ShaderState, _configure_shader_color, triangulate


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

  def to_screen_many(self, xs, ys, heights=0.0) -> tuple[np.ndarray, np.ndarray]:
    """Screen positions (shape of xs, plus 2) of many points in one matrix product, and a mask of the ones to_screen would not
    drop. Calling to_screen in a loop costs a numpy interpolation and a matrix product per point."""
    xs = np.asarray(xs, dtype=float)
    shape = xs.shape
    xs = xs.ravel()
    ys = np.broadcast_to(np.asarray(ys, dtype=float), shape).ravel()
    heights = np.broadcast_to(np.asarray(heights, dtype=float), shape).ravel()
    proj = self._r._car_space_transform @ np.stack([xs, ys + self._r._camera_offset, self.ground_z(xs) - heights])
    w = proj[2]
    ok = (xs > 0.0) & (np.abs(w) >= 1e-6)
    sx, sy = proj[0] / np.where(ok, w, 1.0), proj[1] / np.where(ok, w, 1.0)
    clip = self._r._clip_region
    if clip is not None:
      ok &= (sx >= clip.x) & (sx <= clip.x + clip.width) & (sy >= clip.y) & (sy <= clip.y + clip.height)
    return np.stack([sx, sy], axis=-1).reshape(shape + (2,)), ok.reshape(shape)

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

  def ribbons(self, xs, center_ys: list, half_widths: list[float], height: float = 0.0) -> list[np.ndarray]:
    """ribbon() for several bands along the same xs, with every edge projected in one go. Same polygons, same dropping of points
    out of view, as calling ribbon() once per band."""
    xs = np.asarray(xs, dtype=float)
    if xs.shape[0] < 2:
      return [np.empty((0, 2), dtype=np.float32) for _ in half_widths]
    edges = np.stack([np.stack([np.asarray(c, dtype=float) - hw, np.asarray(c, dtype=float) + hw]) for c, hw in zip(center_ys, half_widths, strict=True)])
    screen, ok = self.to_screen_many(np.broadcast_to(xs, edges.shape), edges, height)
    polygons = []
    for (left, right), (ok_left, ok_right) in zip(screen, ok, strict=True):
      both = ok_left & ok_right
      polygons.append(np.vstack((left[both], right[both][::-1])).astype(np.float32) if both.any() else np.empty((0, 2), dtype=np.float32))
    return polygons


def draw_ribbons(origin_rect: rl.Rectangle, polygons: list[np.ndarray], gradient: Gradient) -> None:
  """shader_polygon.draw_polygon for several ribbons that share a gradient: the shader is set up and switched to once for all of
  them. Every switch flushes raylib's batch, so drawing both curve rails in one go halves the flushes."""
  polygons = [p for p in polygons if len(p) >= 3]
  if not polygons:
    return
  state = ShaderState.get_instance()
  state.initialize()
  _configure_shader_color(state, None, gradient, origin_rect)
  rl.begin_shader_mode(state.shader)
  for poly in polygons:
    strip = triangulate(np.ascontiguousarray(poly, dtype=np.float32))
    rl.draw_triangle_strip(strip, len(strip), rl.WHITE)
  rl.end_shader_mode()
