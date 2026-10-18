"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

From pixels to the road. Frames as in common/transformations: the calibrated frame has x forward, y right, z down, with its origin at
the camera; the road is the plane z = camera height. A pixel's ray is turned into that frame with the same calibration the UI uses
to draw the path, so a car's wheels (the bottom of its box) land where the path would put them.
"""
from dataclasses import dataclass

import numpy as np

from openpilot.common.transformations.camera import view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler

RADAR_TO_CAMERA = 1.52      # m, radar distances are from the front bumper, the cameras sit this far behind it
DEFAULT_HEIGHT = 1.22       # m, camera height above the road before calibration has one


@dataclass(frozen=True)
class CameraGeometry:
  intrinsics: np.ndarray        # 3 x 3
  view_from_calib: np.ndarray   # 3 x 3, rotation
  height: float                 # m above the road

  def rays(self, u, v) -> np.ndarray:
    """Unit rays (N x 3) in the calibrated frame for pixels (u, v)."""
    u, v = np.atleast_1d(np.asarray(u, dtype=float)), np.atleast_1d(np.asarray(v, dtype=float))
    pix = np.stack((u, v, np.ones_like(u)))
    view = np.linalg.solve(self.intrinsics, pix)
    calib = (self.view_from_calib.T @ view).T
    return calib / np.linalg.norm(calib, axis=1, keepdims=True)

  def ground(self, u, v) -> np.ndarray:
    """Where the rays through (u, v) meet the road, as (x forward, y right) in m from the camera. NaN for pixels at or above the
    horizon, whose rays never come down."""
    r = self.rays(u, v)
    t = np.where(r[:, 2] > 1e-3, self.height / np.maximum(r[:, 2], 1e-3), np.nan)
    return np.stack((r[:, 0] * t, r[:, 1] * t), axis=1)

  def angles(self, u, v) -> tuple[np.ndarray, np.ndarray]:
    """(bearing, elevation) in rad: bearing right positive from straight ahead, elevation above the horizon."""
    r = self.rays(u, v)
    return np.arctan2(r[:, 1], r[:, 0]), np.arctan2(-r[:, 2], np.hypot(r[:, 0], r[:, 1]))

  def project(self, x, y, z=None) -> np.ndarray:
    """Pixels (N x 2) of calibrated-frame points; z defaults to the road. Points behind the camera come back NaN."""
    x, y = np.atleast_1d(np.asarray(x, dtype=float)), np.atleast_1d(np.asarray(y, dtype=float))
    z = np.full_like(x, self.height) if z is None else np.broadcast_to(np.asarray(z, dtype=float), x.shape)
    view = self.view_from_calib @ np.stack((x, y, z))
    pix = self.intrinsics @ view
    with np.errstate(divide="ignore", invalid="ignore"):
      out = (pix[:2] / np.where(pix[2] > 1e-6, pix[2], np.nan)).T
    return out


def camera_geometry(intrinsics: np.ndarray, rpy_calib, height: float, wide_from_device_euler=None) -> CameraGeometry:
  """Geometry of the road camera, or of the wide camera when wide_from_device_euler is given."""
  device_from_calib = rot_from_euler(rpy_calib) if len(rpy_calib) == 3 else np.eye(3)
  view_from_calib = view_frame_from_device_frame @ device_from_calib
  if wide_from_device_euler is not None and len(wide_from_device_euler) == 3:
    view_from_calib = view_frame_from_device_frame @ rot_from_euler(wide_from_device_euler) @ device_from_calib
  return CameraGeometry(np.asarray(intrinsics, dtype=float), view_from_calib, float(height) if height > 0.5 else DEFAULT_HEIGHT)
