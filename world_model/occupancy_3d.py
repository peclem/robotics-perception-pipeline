"""
Occupancy3DBuilder — sparse 3D obstacle grid from the SceneGraph.

The 2D OccupancyGridBuilder discards the vertical axis. That is fine for
a flat-ground mobile robot but loses critical information for head-mounted
AR, drones, or manipulator arms, all of which need to know whether the
column above a (x, y) cell is clear.

This builder produces a sparse octomap-shaped 3D voxel grid. Each tracked
object inflates a sphere of radius derived from the same inflation rule
the 2D builder uses (metric uncertainty when intrinsics + depth are
present; per-class fallback otherwise). Occupied voxels are stored as a
dict {(i, j, k): value}; empty space is implicit. Scales with the number
of obstacles, not the grid extent.

Output formats
--------------
The dict can be converted to:
  - PointCloud2 of voxel centres   (always cheap, standard message)
  - octomap_msgs/Octomap            (true octree, requires octomap lib)

Octomap publishing is done in the ROS2 node when octomap_msgs imports;
the builder itself only knows about the voxel dict, keeping it pure
Python with zero ROS / octomap dependencies.

Reference
---------
Hornung et al. (2013) — OctoMap: An Efficient Probabilistic 3D Mapping
                        Framework Based on Octrees. Autonomous Robots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from perception.camera_interface import CameraIntrinsics
from world_model.object_state import ObjectState

log = logging.getLogger(__name__)


OCCUPIED = np.int8(100)


VoxelKey = Tuple[int, int, int]


@dataclass
class Occupancy3DParams:
    """
    Geometry + inflation parameters for the sparse 3D obstacle grid.

    Defaults: a 20×20×3 m volume centred on the robot in XY, spanning
    -0.5 m to +2.5 m in Z (rough person/manipulator workspace) at 10 cm
    resolution. 1.2 M voxels worth of extent, but only occupied voxels
    are stored.
    """
    resolution_m:        float                 = 0.10
    size_x_m:            float                 = 20.0
    size_y_m:            float                 = 20.0
    size_z_m:            float                 = 3.0
    origin_x_m:          float                 = -10.0
    origin_y_m:          float                 = -10.0
    origin_z_m:          float                 = -0.5
    default_inflation_m: float                 = 0.5
    min_inflation_m:     float                 = 0.10
    per_class_inflation_m: Dict[str, float]    = field(default_factory=lambda: {
        "person":     0.40,
        "bicycle":    0.60,
        "car":        2.00,
        "motorcycle": 0.80,
        "bus":        3.00,
        "truck":      2.50,
    })


@dataclass
class Occupancy3D:
    """
    Result of one Occupancy3DBuilder.build() call.

    occupied_voxels
        {(i, j, k): occupancy_value} for cells whose centre lies inside
        an inflation sphere. Cells absent from the dict are implicitly
        free / unknown.
    params
        The parameters used to build this grid. Needed by consumers
        that want to convert voxel indices back to world-frame metres.
    """
    occupied_voxels: Dict[VoxelKey, np.int8]
    params:          Occupancy3DParams

    def __len__(self) -> int:
        return len(self.occupied_voxels)

    def voxel_centres_world(self) -> np.ndarray:
        """
        Return an (N, 3) float64 array of voxel-centre world positions.
        Useful for PointCloud2 emission and unit testing.
        """
        if not self.occupied_voxels:
            return np.zeros((0, 3), dtype=np.float64)
        p = self.params
        res = p.resolution_m
        keys = np.array(list(self.occupied_voxels.keys()), dtype=np.float64)
        # Cell-centre convention matches the 2D builder: stored index is
        # the corner, centre is at +0.5 cells.
        xs = p.origin_x_m + (keys[:, 0] + 0.5) * res
        ys = p.origin_y_m + (keys[:, 1] + 0.5) * res
        zs = p.origin_z_m + (keys[:, 2] + 0.5) * res
        return np.stack([xs, ys, zs], axis=1)


class Occupancy3DBuilder:
    """
    Builds an Occupancy3D from a list of ObjectStates.

    Construct once with params + optional intrinsics; call build() each
    frame with the current SceneGraph objects.
    """

    def __init__(
        self,
        params: Occupancy3DParams,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> None:
        self._p = params
        self._intr = intrinsics
        self._width  = int(round(params.size_x_m / params.resolution_m))
        self._height = int(round(params.size_y_m / params.resolution_m))
        self._depth  = int(round(params.size_z_m / params.resolution_m))
        if self._width <= 0 or self._height <= 0 or self._depth <= 0:
            raise ValueError(
                f"Volume {params.size_x_m}×{params.size_y_m}×{params.size_z_m} m "
                f"at resolution {params.resolution_m} m yields zero voxels "
                f"along at least one axis."
            )

    @property
    def width_cells(self) -> int:
        return self._width

    @property
    def height_cells(self) -> int:
        return self._height

    @property
    def depth_cells(self) -> int:
        return self._depth

    @property
    def params(self) -> Occupancy3DParams:
        return self._p

    def build(self, objects: Iterable[ObjectState]) -> Occupancy3D:
        """
        Build the sparse 3D occupancy grid for the current frame.

        Objects without a world-frame position are skipped (typically
        means no ego-pose available yet). Objects whose world-frame Z
        is missing (2D-only position) fall back to the grid's centre Z.
        """
        occupied: Dict[VoxelKey, np.int8] = {}
        cz_default = self._p.origin_z_m + 0.5 * self._p.size_z_m
        for obj in objects:
            if obj.position_world is None:
                continue
            cx = float(obj.position_world[0])
            cy = float(obj.position_world[1])
            cz = (float(obj.position_world[2])
                  if obj.position_world.shape[0] >= 3 else cz_default)
            radius_m = self._inflation_for(obj)
            self._stamp_sphere(occupied, cx, cy, cz, radius_m)
        return Occupancy3D(occupied_voxels=occupied, params=self._p)

    def _inflation_for(self, obj: ObjectState) -> float:
        """
        Same rule as OccupancyGridBuilder._inflation_for — depth-projected
        2σ pixel uncertainty when available, per-class table otherwise.
        Kept duplicated rather than imported because the 2D and 3D builders
        are independent consumers; sharing the logic would couple them.
        """
        if self._intr is not None and obj.has_metric_depth:
            depth_m = obj.depth_m
            if depth_m > 0.0:
                std_px = obj.position_std
                std_max_px = float(np.max(std_px)) if std_px.size else 0.0
                radius_m = 2.0 * std_max_px * depth_m / self._intr.fx
                return max(radius_m, self._p.min_inflation_m)
        return self._p.per_class_inflation_m.get(
            obj.class_name, self._p.default_inflation_m,
        )

    def _stamp_sphere(
        self,
        occupied: Dict[VoxelKey, np.int8],
        cx_m: float, cy_m: float, cz_m: float,
        radius_m: float,
    ) -> None:
        """Mark voxels whose centres lie within `radius_m` of (cx,cy,cz)."""
        p = self._p
        res = p.resolution_m
        cx_c = (cx_m - p.origin_x_m) / res
        cy_c = (cy_m - p.origin_y_m) / res
        cz_c = (cz_m - p.origin_z_m) / res
        radius_c = radius_m / res

        x_min = max(0, int(np.floor(cx_c - radius_c)))
        x_max = min(self._width,  int(np.ceil(cx_c + radius_c)) + 1)
        y_min = max(0, int(np.floor(cy_c - radius_c)))
        y_max = min(self._height, int(np.ceil(cy_c + radius_c)) + 1)
        z_min = max(0, int(np.floor(cz_c - radius_c)))
        z_max = min(self._depth,  int(np.ceil(cz_c + radius_c)) + 1)
        if x_max <= x_min or y_max <= y_min or z_max <= z_min:
            return  # entire sphere is off-volume

        # Vectorised distance test on the bounding-box subgrid.
        zs, ys, xs = np.mgrid[z_min:z_max, y_min:y_max, x_min:x_max]
        dist2 = ((xs + 0.5 - cx_c) ** 2
               + (ys + 0.5 - cy_c) ** 2
               + (zs + 0.5 - cz_c) ** 2)
        mask = dist2 <= radius_c ** 2
        if not mask.any():
            return
        ii, jj, kk = np.where(mask)
        for a, b, c in zip(ii, jj, kk):
            key: VoxelKey = (int(x_min + c), int(y_min + b), int(z_min + a))
            occupied[key] = OCCUPIED

    def world_to_cell(
        self, x_m: float, y_m: float, z_m: float,
    ) -> Tuple[int, int, int]:
        """Return (i, j, k) for a world (x, y, z). Useful for tests + viz."""
        res = self._p.resolution_m
        return (
            int((x_m - self._p.origin_x_m) / res),
            int((y_m - self._p.origin_y_m) / res),
            int((z_m - self._p.origin_z_m) / res),
        )

    def __repr__(self) -> str:
        return (
            f"Occupancy3DBuilder({self._width}×{self._height}×{self._depth} cells, "
            f"resolution={self._p.resolution_m} m, "
            f"origin=({self._p.origin_x_m},{self._p.origin_y_m},"
            f"{self._p.origin_z_m}) m)"
        )
