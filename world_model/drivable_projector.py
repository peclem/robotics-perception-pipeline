"""
Inverse perspective mapping for the drivable mask.

Projects an image-space drivable mask (boolean per pixel) onto the
world-frame ground plane and rasterises into a nav_msgs/OccupancyGrid-
shaped int8 array. Flat-ground assumption (z = z_ground in world).
Depth-aware projection would slot in alongside via a second backend.

Output convention follows nav_msgs/OccupancyGrid:

    -1   unknown — pixel didn't project here or fell outside the grid
     0   free / drivable
   100   occupied (reserved; not produced by v1)

Math
----
For pixel (u, v) and pinhole intrinsics (fx, fy, cx, cy):

    ray_camera = ((u - cx) / fx, (v - cy) / fy, 1.0)

CameraPose has the world←camera convention (R rotates camera-frame
vectors to world frame; t is the camera origin expressed in world).
So the world-frame ray direction and origin are:

    d_world = R @ ray_camera     (direction; rotation only)
    o_world = t                  (origin)

Intersection with the ground plane z = z_ground:

    s = (z_ground - o_world.z) / d_world.z

Reject pixels where s <= 0 (behind / parallel / pointing away from
the ground). For valid intersections:

    p_world.x = o_world.x + s * d_world.x
    p_world.y = o_world.y + s * d_world.y

Drop into the grid cell at (c, r) where

    c = int((p_world.x - origin_x_m) / resolution_m)
    r = int((p_world.y - origin_y_m) / resolution_m)

References
----------
The classical IPM recipe; see Bertozzi & Broggi (1998) "GOLD: A
parallel real-time stereo vision system for generic obstacle and
lane detection" for the lane-marking variant, and any modern
autonomous-driving freespace pipeline (Mobileye, Tesla AP) for the
semantic-mask variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from world_model.occupancy_grid import OccupancyGridParams

if TYPE_CHECKING:
    from perception.camera_interface import CameraIntrinsics
    from perception.pose_estimator import CameraPose


@dataclass
class DrivableProjectorParams:
    """
    Grid + ground-plane parameters for the IPM projector.

    grid_params  reuses the same dataclass as OccupancyGridBuilder
                 (resolution_m, size, origin) so consumers can share
                 a single grid spec between the dynamic-obstacle layer
                 and the drivable-freespace layer.
    z_ground_m   world-frame z value of the assumed flat ground plane.
                 0.0 for a "robot's wheels are at z=0" convention; set
                 negative if the world origin is at sensor height.
    """
    grid_params: OccupancyGridParams
    z_ground_m:  float = 0.0


def project_drivable_to_grid(
    mask:         np.ndarray,
    intrinsics:   "CameraIntrinsics",
    camera_pose:  "CameraPose",
    params:       DrivableProjectorParams,
) -> np.ndarray:
    """
    Project a (H, W) drivable mask onto the ground plane and rasterise
    into a (Hg, Wg) int8 OccupancyGrid array.

    `mask` may be bool or uint8 (any non-zero value → drivable).

    Returns the int8 grid; values are -1 (unknown) or 0 (drivable). The
    100-occupied class is reserved for a future "non-drivable surface
    projection" backend and is not produced here.
    """
    gp = params.grid_params
    Hg = int(round(gp.size_y_m / gp.resolution_m))
    Wg = int(round(gp.size_x_m / gp.resolution_m))
    grid = np.full((Hg, Wg), -1, dtype=np.int8)

    if mask is None or mask.size == 0:
        return grid

    # Drivable pixel indices in image space (v=row, u=col).
    drivable = np.argwhere(np.asarray(mask) > 0)
    if drivable.shape[0] == 0:
        return grid

    v = drivable[:, 0].astype(np.float64)
    u = drivable[:, 1].astype(np.float64)

    # Ray directions in camera frame, one per drivable pixel.
    d_cam = np.stack([
        (u - intrinsics.cx) / intrinsics.fx,
        (v - intrinsics.cy) / intrinsics.fy,
        np.ones_like(u),
    ], axis=1)                                          # (N, 3)

    # Rotate into world frame (R is world ← camera).
    R = np.asarray(camera_pose.R, dtype=np.float64)
    t = np.asarray(camera_pose.t, dtype=np.float64).reshape(3)
    d_world = d_cam @ R.T                               # (N, 3)

    # Intersection with z = z_ground plane:
    #   s = (z_ground - o.z) / d.z
    # Valid when s > 0 (in front of camera + ray actually hits ground).
    s_num = params.z_ground_m - t[2]
    s_den = d_world[:, 2]
    # Mask out rays parallel-to-ground or pointing the wrong way.
    valid = np.abs(s_den) > 1e-9
    if not np.any(valid):
        return grid
    s = np.where(valid, s_num / np.where(valid, s_den, 1.0), 0.0)
    valid &= (s > 0.0)
    if not np.any(valid):
        return grid

    # World-frame XY of the ground-plane intersection.
    xs = t[0] + s * d_world[:, 0]
    ys = t[1] + s * d_world[:, 1]

    # Grid cell indices.
    res = gp.resolution_m
    cols = np.floor((xs - gp.origin_x_m) / res).astype(np.int64)
    rows = np.floor((ys - gp.origin_y_m) / res).astype(np.int64)

    in_grid = valid & (cols >= 0) & (cols < Wg) & (rows >= 0) & (rows < Hg)
    if not np.any(in_grid):
        return grid

    grid[rows[in_grid], cols[in_grid]] = 0
    return grid
