"""
OccupancyGridBuilder — dynamic obstacle grid from the SceneGraph.

Each tracked object inflates a region around its world-frame position
proportional to its 2σ position uncertainty. Output is a 2D int8 array
in the standard nav_msgs/OccupancyGrid value convention:
    -1 = unknown (not used here; all cells start at 0)
     0 = free
   100 = occupied

The grid is a top-down (XY) projection in the world frame. Z is
ignored — appropriate for a flat-ground mobile robot. A planner
consumes the grid via a Nav2 costmap layer or equivalent.

Inflation strategy
------------------
Two paths, in order of preference:

1. **Metric uncertainty (preferred).** When camera intrinsics are
   provided AND the object has metric depth (position_3d.Z), convert
   the 2σ pixel uncertainty into metres:
       radius_m ≈ 2σ_px × depth_m / fx
   This honours the KF's reported uncertainty. Inflated objects with
   loose tracks (large covariance) shrink as the filter converges.

2. **Per-class fallback.** Without depth or intrinsics, use a fixed
   metric inflation per class (e.g. person=0.4 m, car=2.0 m) with a
   single default for unknown classes.

Static map
----------
This builder produces only the *dynamic* obstacle layer. A static
layer (walls, furniture) requires SLAM-derived prior maps and is a
separate concern. Combined static + dynamic costmaps live in the
downstream Nav2 layer stack.

Reference
---------
Lu & Hyyppä (2014) — Layered Costmaps for Context-Sensitive Navigation
ROS docs       — nav_msgs/OccupancyGrid + nav2_costmap_2d
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np

from perception.camera_interface import CameraIntrinsics
from world_model.object_state import ObjectState

log = logging.getLogger(__name__)


OCCUPIED = np.int8(100)
FREE = np.int8(0)


@dataclass
class OccupancyGridParams:
    """
    Geometry + inflation parameters for the dynamic obstacle grid.

    Sizes are in metres; resolution is metres/cell. Defaults match the
    Nav2 convention of a 20×20 m local costmap at 5 cm resolution.
    """
    resolution_m:        float                 = 0.05
    size_x_m:            float                 = 20.0
    size_y_m:            float                 = 20.0
    origin_x_m:          float                 = -10.0
    origin_y_m:          float                 = -10.0
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


class OccupancyGridBuilder:
    """
    Builds a (height, width) int8 grid from a list of ObjectStates.

    Construct once with params + optional intrinsics; call build() each
    frame with the current SceneGraph objects.
    """

    def __init__(
        self,
        params: OccupancyGridParams,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> None:
        self._p = params
        self._intr = intrinsics
        self._width  = int(round(params.size_x_m / params.resolution_m))
        self._height = int(round(params.size_y_m / params.resolution_m))
        if self._width <= 0 or self._height <= 0:
            raise ValueError(
                f"Grid size {params.size_x_m}×{params.size_y_m} m at "
                f"resolution {params.resolution_m} m yields zero cells."
            )

    @property
    def width_cells(self) -> int:
        return self._width

    @property
    def height_cells(self) -> int:
        return self._height

    @property
    def params(self) -> OccupancyGridParams:
        return self._p

    def build(self, objects: Iterable[ObjectState]) -> np.ndarray:
        """
        Build the occupancy grid for the current frame.

        Objects without a world-frame position are skipped (typically
        means no ego-pose available yet).

        Returns
        -------
        (H, W) int8 array. H rows × W columns. data[r, c] is the
        occupancy value of the cell at world position
            x = origin_x_m + c × resolution_m
            y = origin_y_m + r × resolution_m
        which matches nav_msgs/OccupancyGrid's row-major flattening.
        """
        grid = np.zeros((self._height, self._width), dtype=np.int8)
        for obj in objects:
            if obj.position_world is None:
                continue
            radius_m = self._inflation_for(obj)
            self._stamp_disk(
                grid,
                cx_m=float(obj.position_world[0]),
                cy_m=float(obj.position_world[1]),
                radius_m=radius_m,
            )
        return grid

    def _inflation_for(self, obj: ObjectState) -> float:
        """
        Metric inflation radius for one object. Tries the
        depth-projected pixel covariance first, falls back to the
        per-class table, then to a single default.
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

    def _stamp_disk(
        self,
        grid:     np.ndarray,
        cx_m:     float,
        cy_m:     float,
        radius_m: float,
    ) -> None:
        """Mark cells within `radius_m` of (cx_m, cy_m) as OCCUPIED."""
        res = self._p.resolution_m
        cx_c = (cx_m - self._p.origin_x_m) / res
        cy_c = (cy_m - self._p.origin_y_m) / res
        radius_c = radius_m / res

        # Tight bounding box, clipped to grid.
        x_min = max(0, int(np.floor(cx_c - radius_c)))
        x_max = min(self._width,  int(np.ceil(cx_c + radius_c)) + 1)
        y_min = max(0, int(np.floor(cy_c - radius_c)))
        y_max = min(self._height, int(np.ceil(cy_c + radius_c)) + 1)
        if x_max <= x_min or y_max <= y_min:
            return  # entire disk is off-grid

        # Compare cell *centers* (i + 0.5) to the disk center — matches
        # nav_msgs/OccupancyGrid's convention that `origin` is the bottom-
        # left corner of cell (0, 0).
        ys, xs = np.mgrid[y_min:y_max, x_min:x_max]
        dist2 = (xs + 0.5 - cx_c) ** 2 + (ys + 0.5 - cy_c) ** 2
        grid[y_min:y_max, x_min:x_max][dist2 <= radius_c ** 2] = OCCUPIED

    def world_to_cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Return (col, row) for a world (x, y). Useful for tests + viz."""
        res = self._p.resolution_m
        return (
            int((x_m - self._p.origin_x_m) / res),
            int((y_m - self._p.origin_y_m) / res),
        )

    def __repr__(self) -> str:
        return (
            f"OccupancyGridBuilder({self._width}×{self._height} cells, "
            f"resolution={self._p.resolution_m} m, "
            f"origin=({self._p.origin_x_m},{self._p.origin_y_m}) m)"
        )
