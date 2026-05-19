"""
Unit tests for world_model.drivable_projector.

Tests the IPM math against hand-computed expected outputs at a few
clean camera-pose configurations (looking straight down, looking
forward, tilted) and the edge cases (no drivable pixels, all rays
above the horizon, off-grid projections, etc.).
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.camera_interface import CameraIntrinsics
from perception.pose_estimator import CameraPose
from world_model.drivable_projector import (
    DrivableProjectorParams, project_drivable_to_grid,
)
from world_model.occupancy_grid import OccupancyGridParams


def _intrinsics(w=640, h=480, fx=500.0, fy=500.0):
    return CameraIntrinsics(
        fx=fx, fy=fy, cx=w / 2.0, cy=h / 2.0,
        width=w, height=h,
    )


def _grid_params(res=0.5, size=10.0):
    """A 10×10 m grid at 0.5 m/cell centred on the world origin."""
    return OccupancyGridParams(
        resolution_m=res, size_x_m=size, size_y_m=size,
        origin_x_m=-size / 2.0, origin_y_m=-size / 2.0,
    )


def _looking_down_pose(height=1.0):
    """Camera at (0, 0, height) looking straight down (-Z in world)."""
    # 180° rotation about world X: camera Z → world -Z, camera X → world X.
    R = np.array([[1.0, 0.0,  0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0]])
    return CameraPose(
        R=R, t=np.array([0.0, 0.0, float(height)]),
        timestamp=0.0, frame_idx=0, source="test",
    )


class TestProjectDrivable:

    def test_empty_mask_returns_all_unknown(self):
        intr   = _intrinsics()
        gp     = _grid_params()
        params = DrivableProjectorParams(grid_params=gp)
        mask   = np.zeros((intr.height, intr.width), dtype=np.uint8)
        out    = project_drivable_to_grid(
            mask, intr, _looking_down_pose(), params,
        )
        assert out.shape == (
            int(gp.size_y_m / gp.resolution_m),
            int(gp.size_x_m / gp.resolution_m),
        )
        assert (out == -1).all()
        assert out.dtype == np.int8

    def test_center_pixel_projects_to_world_origin(self):
        """
        Camera at (0, 0, 1) looking straight down. The centre pixel
        ray is along world -Z and hits the ground at (0, 0, 0).
        That world point falls in the grid cell containing (0, 0).
        """
        intr   = _intrinsics()
        gp     = _grid_params()
        params = DrivableProjectorParams(grid_params=gp)
        mask   = np.zeros((intr.height, intr.width), dtype=np.uint8)
        mask[int(intr.cy), int(intr.cx)] = 255

        out = project_drivable_to_grid(
            mask, intr, _looking_down_pose(height=1.0), params,
        )
        # Cell at (col, row) corresponding to world (0, 0):
        #   col = floor((0 - origin_x) / res) = floor(5 / 0.5) = 10
        #   row = floor((0 - origin_y) / res) = floor(5 / 0.5) = 10
        assert out[10, 10] == 0
        # Only one cell should be marked.
        assert (out == 0).sum() == 1

    def test_offset_pixel_projects_to_expected_world_xy(self):
        """
        With fx = fy = 200, camera at (0, 0, 1) looking straight down,
        a pixel at (cx + fx, cy) has camera-frame ray [1, 0, 1]; the
        world-frame ray (after the 180° X rotation) is [1, 0, -1].
        Ground intersection at s=1: world (1, 0).
        """
        intr   = _intrinsics(fx=200.0, fy=200.0)
        gp     = _grid_params()
        params = DrivableProjectorParams(grid_params=gp)
        mask   = np.zeros((intr.height, intr.width), dtype=np.uint8)
        # u = cx + fx → one focal length to the right (520 < width=640).
        mask[int(intr.cy), int(intr.cx + intr.fx)] = 255

        out = project_drivable_to_grid(
            mask, intr, _looking_down_pose(height=1.0), params,
        )
        # World (1, 0) → col = floor((1 - (-5)) / 0.5) = 12, row = 10.
        assert out[10, 12] == 0
        assert (out == 0).sum() == 1

    def test_mask_dtype_bool_works(self):
        intr   = _intrinsics()
        gp     = _grid_params()
        params = DrivableProjectorParams(grid_params=gp)
        mask   = np.zeros((intr.height, intr.width), dtype=bool)
        mask[int(intr.cy), int(intr.cx)] = True
        out = project_drivable_to_grid(
            mask, intr, _looking_down_pose(), params,
        )
        assert (out == 0).sum() == 1

    def test_rays_above_horizon_skipped(self):
        """
        Camera at (0, 0, 1) but rotated to face the +X direction (level
        with the horizon). Pixels above the image centre point above
        the horizon → never hit the ground. The projector must not
        crash or scatter into the grid.
        """
        intr   = _intrinsics()
        gp     = _grid_params()
        params = DrivableProjectorParams(grid_params=gp)
        # Rotation: camera looks along +X world. Camera Z → world X,
        # camera X → world -Y, camera Y → world -Z (so camera Y points
        # down — image-down maps to world-down).
        R = np.array([[0.0, 0.0, 1.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0]])
        pose = CameraPose(
            R=R, t=np.array([0.0, 0.0, 1.0]),
            timestamp=0.0, frame_idx=0, source="test",
        )
        mask = np.zeros((intr.height, intr.width), dtype=np.uint8)
        # Pixel ABOVE the principal point — its ray, after the level
        # rotation, points slightly upward → never reaches ground.
        mask[int(intr.cy - 100), int(intr.cx)] = 255
        out = project_drivable_to_grid(mask, intr, pose, params)
        # No cell marked drivable.
        assert (out == 0).sum() == 0
        assert (out == -1).all()

    def test_off_grid_points_skipped(self):
        """
        A pixel that projects to a point outside the configured grid
        bounds must not raise — it just doesn't get rasterised.
        """
        intr   = _intrinsics(fx=10.0, fy=10.0)   # very wide FOV
        gp     = _grid_params(res=0.1, size=1.0)  # tiny 1 m grid
        params = DrivableProjectorParams(grid_params=gp)
        mask   = np.zeros((intr.height, intr.width), dtype=np.uint8)
        # Pixel far from centre → projects far from world origin → off-grid.
        mask[10, 10] = 255
        out = project_drivable_to_grid(
            mask, intr, _looking_down_pose(height=1.0), params,
        )
        # No drivable cells; grid still entirely unknown.
        assert (out == 0).sum() == 0

    def test_z_ground_offset_changes_intersection_height(self):
        """
        Lower the ground plane to z = -0.5: same camera at (0, 0, 1)
        looking down, the centre-pixel ray hits z = -0.5 at s=1.5, but
        the world XY is still (0, 0) because the ray points along -Z.
        Sanity check: the projector accepts a non-zero z_ground.
        """
        intr = _intrinsics()
        gp   = _grid_params()
        params = DrivableProjectorParams(grid_params=gp, z_ground_m=-0.5)
        mask = np.zeros((intr.height, intr.width), dtype=np.uint8)
        mask[int(intr.cy), int(intr.cx)] = 255
        out  = project_drivable_to_grid(
            mask, intr, _looking_down_pose(height=1.0), params,
        )
        # Still lands at the origin cell.
        assert out[10, 10] == 0

    def test_camera_at_ground_level_rejects_horizontal_rays(self):
        """
        Camera at z=0 with looking-down rays still hits z=0 plane at s=0
        (the camera origin itself). s > 0 strict, so we reject.
        """
        intr = _intrinsics()
        gp   = _grid_params()
        params = DrivableProjectorParams(grid_params=gp)
        mask = np.zeros((intr.height, intr.width), dtype=np.uint8)
        mask[int(intr.cy), int(intr.cx)] = 255
        out  = project_drivable_to_grid(
            mask, intr, _looking_down_pose(height=0.0), params,
        )
        # s = 0 → rejected; nothing marked drivable.
        assert (out == 0).sum() == 0

    def test_multiple_drivable_pixels_scatter_into_grid(self):
        """
        Two drivable pixels symmetric about cx land in symmetric cells.
        """
        intr = _intrinsics(fx=200.0, fy=200.0)
        gp   = _grid_params()
        params = DrivableProjectorParams(grid_params=gp)
        mask = np.zeros((intr.height, intr.width), dtype=np.uint8)
        mask[int(intr.cy), int(intr.cx + intr.fx)] = 255  # → world (+1, 0)
        mask[int(intr.cy), int(intr.cx - intr.fx)] = 255  # → world (-1, 0)
        out = project_drivable_to_grid(
            mask, intr, _looking_down_pose(height=1.0), params,
        )
        # (+1, 0) → col 12, row 10 ; (-1, 0) → col 8, row 10.
        assert out[10, 12] == 0
        assert out[10, 8]  == 0
        assert (out == 0).sum() == 2


class TestDepthAwareProjection:
    """
    depth_map=array → back-project each pixel through metric depth.
    Same downstream rasterisation as the flat-ground backend.
    """

    def test_center_pixel_with_depth_projects_to_origin(self):
        """
        Camera at (0, 0, 1) looking straight down; centre-pixel depth
        of 1.0 m → ground at world (0, 0). Same cell as the flat-
        ground case in TestProjectDrivable.
        """
        intr   = _intrinsics()
        gp     = _grid_params()
        params = DrivableProjectorParams(grid_params=gp)
        mask   = np.zeros((intr.height, intr.width), dtype=np.uint8)
        depth  = np.zeros_like(mask, dtype=np.float32)
        mask [int(intr.cy), int(intr.cx)] = 255
        depth[int(intr.cy), int(intr.cx)] = 1.0
        out = project_drivable_to_grid(
            mask, intr, _looking_down_pose(height=1.0), params,
            depth_map=depth,
        )
        assert out[10, 10] == 0
        assert (out == 0).sum() == 1

    def test_offset_pixel_uses_depth_for_x(self):
        """
        fx=fy=200, camera at (0, 0, 1) looking down. Pixel at
        (cx + fx, cy) with depth 1.0 m:

            p_cam   = (1, 0, 1)   (fx cancels: (u-cx)*d/fx = fx*1/fx)
            p_world = R @ (1,0,1) + (0,0,1)
                    = (1, 0, -1) + (0, 0, 1) = (1, 0, 0)

        So same cell (10, 12) as the flat-ground case — sanity check
        that depth-aware agrees with flat-ground when the depth IS
        the ground-plane depth.
        """
        intr   = _intrinsics(fx=200.0, fy=200.0)
        gp     = _grid_params()
        params = DrivableProjectorParams(grid_params=gp)
        mask   = np.zeros((intr.height, intr.width), dtype=np.uint8)
        depth  = np.zeros_like(mask, dtype=np.float32)
        u = int(intr.cx + intr.fx)
        v = int(intr.cy)
        mask [v, u] = 255
        depth[v, u] = 1.0
        out = project_drivable_to_grid(
            mask, intr, _looking_down_pose(height=1.0), params,
            depth_map=depth,
        )
        assert out[10, 12] == 0
        assert (out == 0).sum() == 1

    def test_depth_disagrees_with_ground_plane(self):
        """
        Critical: depth-aware should *NOT* match flat-ground when the
        actual depth differs (e.g. an elevated drivable surface like a
        stair or a balcony). Centre pixel, camera at (0, 0, 1) looking
        down, depth 0.5 m → camera-frame z=0.5, world z=1-0.5=0.5.
        World XY of the centre ray is still (0, 0), but the projection
        will hit the grid cell at (10, 10) regardless of the z value
        (the grid is a flat top-down map). So same cell, different z.

        This test confirms the depth-aware path scales the camera-
        frame point correctly with depth — try a depth that doesn't
        land on z=0 to make sure the world XY is still right.
        """
        intr   = _intrinsics(fx=200.0, fy=200.0)
        gp     = _grid_params()
        params = DrivableProjectorParams(grid_params=gp)
        mask   = np.zeros((intr.height, intr.width), dtype=np.uint8)
        depth  = np.zeros_like(mask, dtype=np.float32)
        u = int(intr.cx + intr.fx)
        v = int(intr.cy)
        mask [v, u] = 255
        # Depth = 2.0 → p_cam = (2, 0, 2); p_world = (2, 0, -2)+(0,0,1) = (2, 0, -1)
        depth[v, u] = 2.0
        out = project_drivable_to_grid(
            mask, intr, _looking_down_pose(height=1.0), params,
            depth_map=depth,
        )
        # World (2, 0) → col = floor((2-(-5))/0.5) = 14, row = 10.
        assert out[10, 14] == 0
        assert (out == 0).sum() == 1

    def test_invalid_depths_skipped(self):
        """
        Pixels with depth == 0 (no observation) or non-finite (NaN/inf)
        must not be projected. Build a mask of 3 drivable pixels, give
        only one a valid depth, expect exactly one grid cell marked.
        """
        intr   = _intrinsics()
        gp     = _grid_params()
        params = DrivableProjectorParams(grid_params=gp)
        mask   = np.zeros((intr.height, intr.width), dtype=np.uint8)
        depth  = np.zeros_like(mask, dtype=np.float32)
        cy, cx = int(intr.cy), int(intr.cx)
        mask [cy,     cx]     = 255
        mask [cy,     cx - 1] = 255   # depth stays 0 → invalid
        mask [cy + 1, cx]     = 255
        depth[cy,     cx]     = 1.0       # valid
        depth[cy + 1, cx]     = np.nan    # invalid
        out = project_drivable_to_grid(
            mask, intr, _looking_down_pose(height=1.0), params,
            depth_map=depth,
        )
        # Only the (cy, cx) pixel survives → world origin → cell (10, 10).
        assert out[10, 10] == 0
        assert (out == 0).sum() == 1

    def test_depth_shape_mismatch_raises(self):
        intr   = _intrinsics()
        gp     = _grid_params()
        params = DrivableProjectorParams(grid_params=gp)
        mask   = np.zeros((intr.height, intr.width), dtype=np.uint8)
        bad    = np.zeros((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="depth_map shape"):
            project_drivable_to_grid(
                mask, intr, _looking_down_pose(), params, depth_map=bad,
            )

    def test_depth_none_matches_flat_ground_baseline(self):
        """
        Backwards-compatibility: explicit depth_map=None == omitting
        the argument == flat-ground projection. Both produce the same
        grid for a fixed input.
        """
        intr   = _intrinsics()
        gp     = _grid_params()
        params = DrivableProjectorParams(grid_params=gp)
        mask   = np.zeros((intr.height, intr.width), dtype=np.uint8)
        mask[int(intr.cy), int(intr.cx)] = 255
        pose   = _looking_down_pose(height=1.0)
        a = project_drivable_to_grid(mask, intr, pose, params)
        b = project_drivable_to_grid(mask, intr, pose, params, depth_map=None)
        np.testing.assert_array_equal(a, b)
