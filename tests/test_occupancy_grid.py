"""
Unit tests for the dynamic occupancy grid builder.

TestGeometry         : world-to-cell mapping, grid sizing
TestStamping         : single-object inflation correctness
TestInflationChoice  : depth-projected vs per-class fallback selection
TestBuild            : end-to-end multi-object scene
"""

from __future__ import annotations

from collections import deque
import time

import numpy as np
import pytest

from perception.camera_interface import CameraIntrinsics
from world_model.object_state import ObjectState
from world_model.occupancy_grid import (
    OCCUPIED, OccupancyGridBuilder, OccupancyGridParams,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_obj(
    track_id:       int = 1,
    class_name:     str = "person",
    position_world=None,
    position_3d=None,
    std_px:         float = 1.0,
) -> ObjectState:
    """Synthesise an ObjectState with chosen world position + KF std."""
    cov = np.eye(8) * (std_px ** 2)
    return ObjectState(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        position=np.array([100.0, 100.0]),
        covariance=cov,
        velocity=np.zeros(4),
        score=0.9,
        last_seen=time.monotonic(),
        n_updates=1,
        position_3d=(np.asarray(position_3d, dtype=np.float64)
                     if position_3d is not None else None),
        position_world=(np.asarray(position_world, dtype=np.float64)
                        if position_world is not None else None),
        max_history=5,
    )


def intr_640x480(fx: float = 500.0) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=fx, fy=fx, cx=320.0, cy=240.0,
        width=640, height=480,
        dist_coeffs=np.zeros(5),
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class TestGeometry:

    def test_grid_dimensions_from_params(self):
        b = OccupancyGridBuilder(OccupancyGridParams(
            resolution_m=0.1, size_x_m=4.0, size_y_m=2.0,
        ))
        assert b.width_cells == 40
        assert b.height_cells == 20

    def test_world_to_cell_default_origin(self):
        b = OccupancyGridBuilder(OccupancyGridParams(
            resolution_m=0.05, size_x_m=20.0, size_y_m=20.0,
            origin_x_m=-10.0, origin_y_m=-10.0,
        ))
        # World (0,0) → cell (200, 200) at 0.05 m / cell, origin at -10
        assert b.world_to_cell(0.0, 0.0) == (200, 200)
        # World (-10, -10) → cell (0, 0)
        assert b.world_to_cell(-10.0, -10.0) == (0, 0)

    def test_zero_size_rejected(self):
        with pytest.raises(ValueError, match="zero cells"):
            OccupancyGridBuilder(OccupancyGridParams(
                resolution_m=1.0, size_x_m=0.5, size_y_m=0.5,
            ))


# ---------------------------------------------------------------------------
# Stamping
# ---------------------------------------------------------------------------

class TestStamping:

    def test_empty_scene_yields_zero_grid(self):
        b = OccupancyGridBuilder(OccupancyGridParams())
        grid = b.build([])
        assert grid.shape == (b.height_cells, b.width_cells)
        assert grid.dtype == np.int8
        assert (grid == 0).all()

    def test_object_without_world_position_skipped(self):
        b = OccupancyGridBuilder(OccupancyGridParams())
        grid = b.build([make_obj(position_world=None)])
        assert (grid == 0).all()

    def test_single_object_marks_cells_around_position(self):
        # Coarse grid for easy reasoning: 0.5 m/cell, 10×10 m, origin (0,0)
        params = OccupancyGridParams(
            resolution_m=0.5, size_x_m=10.0, size_y_m=10.0,
            origin_x_m=0.0, origin_y_m=0.0,
            default_inflation_m=1.0,    # 2-cell radius
            per_class_inflation_m={},   # force default fallback
        )
        b = OccupancyGridBuilder(params)
        # Object at world (5, 5) → cell (10, 10) centre
        obj = make_obj(position_world=[5.0, 5.0, 0.0])
        grid = b.build([obj])
        # Centre cell must be occupied; corner cell must not.
        assert grid[10, 10] == OCCUPIED
        assert grid[0, 0] == 0
        # Disk of cell radius 2 — π × 4 ≈ 13 cells, tolerate 9..17.
        n_occupied = int((grid == OCCUPIED).sum())
        assert 9 <= n_occupied <= 17, \
            f"radius-2-cell disk should yield 9..17 cells, got {n_occupied}"

    def test_object_outside_grid_silently_ignored(self):
        params = OccupancyGridParams(
            resolution_m=0.5, size_x_m=10.0, size_y_m=10.0,
            origin_x_m=0.0, origin_y_m=0.0, default_inflation_m=0.5,
        )
        b = OccupancyGridBuilder(params)
        # World (100, 100) is far outside the [0, 10] × [0, 10] grid
        grid = b.build([make_obj(position_world=[100.0, 100.0, 0.0])])
        assert (grid == 0).all()

    def test_object_partially_off_grid_marks_intersection(self):
        params = OccupancyGridParams(
            resolution_m=0.5, size_x_m=10.0, size_y_m=10.0,
            origin_x_m=0.0, origin_y_m=0.0, default_inflation_m=2.0,
        )
        b = OccupancyGridBuilder(params)
        # Object on the edge — disk should clip
        grid = b.build([make_obj(position_world=[0.0, 5.0, 0.0])])
        assert grid[10, 0] == OCCUPIED
        # Pixels that would be at negative x must not be addressed (no crash)


# ---------------------------------------------------------------------------
# Inflation source selection
# ---------------------------------------------------------------------------

class TestInflationChoice:

    def test_falls_back_to_per_class_when_no_intrinsics(self):
        params = OccupancyGridParams(
            resolution_m=0.1, size_x_m=10.0, size_y_m=10.0,
            origin_x_m=0.0, origin_y_m=0.0,
            per_class_inflation_m={"person": 1.0, "car": 3.0},
        )
        b = OccupancyGridBuilder(params, intrinsics=None)
        person = make_obj(class_name="person",
                          position_world=[5.0, 5.0, 0.0])
        car = make_obj(track_id=2, class_name="car",
                       position_world=[5.0, 5.0, 0.0])
        # Pure heuristic: count occupied cells; car (r=3) > person (r=1).
        grid_person = b.build([person])
        grid_car = b.build([car])
        assert (grid_car == OCCUPIED).sum() > (grid_person == OCCUPIED).sum()

    def test_falls_back_to_default_for_unknown_class(self):
        params = OccupancyGridParams(
            resolution_m=0.1, size_x_m=10.0, size_y_m=10.0,
            origin_x_m=0.0, origin_y_m=0.0,
            default_inflation_m=0.5,
            per_class_inflation_m={"person": 0.1},
        )
        b = OccupancyGridBuilder(params, intrinsics=None)
        ufo = make_obj(class_name="ufo", position_world=[5.0, 5.0, 0.0])
        person = make_obj(track_id=2, class_name="person",
                          position_world=[5.0, 5.0, 0.0])
        n_ufo = int((b.build([ufo]) == OCCUPIED).sum())
        n_person = int((b.build([person]) == OCCUPIED).sum())
        # Default 0.5 m > per-class person 0.1 m → ufo gets more cells.
        assert n_ufo > n_person

    def test_uses_depth_projected_covariance_when_available(self):
        # Set up so the depth-projected radius is unambiguously bigger
        # than any per-class fallback we'd otherwise pick.
        params = OccupancyGridParams(
            resolution_m=0.05, size_x_m=20.0, size_y_m=20.0,
            origin_x_m=-10.0, origin_y_m=-10.0,
            per_class_inflation_m={"person": 0.1},
        )
        intr = intr_640x480(fx=500.0)
        b = OccupancyGridBuilder(params, intrinsics=intr)
        # std_px=100, depth=10 m, fx=500 → radius = 2 * 100 * 10 / 500 = 4 m
        obj = make_obj(
            class_name="person",
            position_world=[0.0, 0.0, 10.0],
            position_3d=[0.0, 0.0, 10.0],
            std_px=100.0,
        )
        grid = b.build([obj])
        # Disk of radius 4 m at 0.05 m/cell → ~80-cell radius → ~π × 80² ≈ 20106
        n_occupied = int((grid == OCCUPIED).sum())
        assert n_occupied > 1000, \
            f"expected large depth-projected disk, got {n_occupied} cells"


# ---------------------------------------------------------------------------
# End-to-end multi-object
# ---------------------------------------------------------------------------

class TestBuild:

    def test_multiple_objects_compose(self):
        params = OccupancyGridParams(
            resolution_m=0.5, size_x_m=10.0, size_y_m=10.0,
            origin_x_m=0.0, origin_y_m=0.0, default_inflation_m=0.5,
        )
        b = OccupancyGridBuilder(params)
        objs = [
            make_obj(track_id=1, position_world=[2.0, 2.0, 0.0]),
            make_obj(track_id=2, position_world=[8.0, 8.0, 0.0]),
            make_obj(track_id=3, position_world=[5.0, 5.0, 0.0]),
        ]
        grid = b.build(objs)
        # Each of three positions should have its centre cell occupied.
        for x, y in [(2.0, 2.0), (8.0, 8.0), (5.0, 5.0)]:
            col, row = b.world_to_cell(x, y)
            assert grid[row, col] == OCCUPIED

    def test_grid_row_major_matches_navmsgs_convention(self):
        # nav_msgs/OccupancyGrid is row-major H×W. Verify our layout.
        params = OccupancyGridParams(
            resolution_m=1.0, size_x_m=3.0, size_y_m=4.0,
            origin_x_m=0.0, origin_y_m=0.0, default_inflation_m=0.5,
            per_class_inflation_m={},
        )
        b = OccupancyGridBuilder(params)
        # Object at (1.5, 2.5) sits at the centre of cell (col=1, row=2).
        grid = b.build([make_obj(position_world=[1.5, 2.5, 0.0])])
        assert grid.shape == (4, 3)         # H × W
        assert grid[2, 1] == OCCUPIED
