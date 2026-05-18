"""
Unit tests for the sparse 3D occupancy grid builder.

TestGeometry        : volume sizing + world-to-cell mapping
TestStamping        : single-object sphere stamping correctness
TestInflationChoice : depth-projected vs per-class fallback (3D mirror of 2D)
TestBuild           : end-to-end multi-object scenes
TestOccupancy3D     : helpers on the Occupancy3D result dataclass
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from perception.camera_interface import CameraIntrinsics
from world_model.object_state import ObjectState
from world_model.occupancy_3d import (
    OCCUPIED, Occupancy3D, Occupancy3DBuilder, Occupancy3DParams,
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

    def test_volume_dimensions_from_params(self):
        b = Occupancy3DBuilder(Occupancy3DParams(
            resolution_m=0.1, size_x_m=4.0, size_y_m=2.0, size_z_m=1.0,
        ))
        assert b.width_cells == 40
        assert b.height_cells == 20
        assert b.depth_cells == 10

    def test_world_to_cell(self):
        b = Occupancy3DBuilder(Occupancy3DParams(
            resolution_m=0.5, size_x_m=10.0, size_y_m=10.0, size_z_m=4.0,
            origin_x_m=-5.0, origin_y_m=-5.0, origin_z_m=-2.0,
        ))
        # world origin sits at the volume centre.
        assert b.world_to_cell(0.0, 0.0, 0.0) == (10, 10, 4)
        # near the +X edge.
        assert b.world_to_cell(4.0, 0.0, 0.0) == (18, 10, 4)

    def test_zero_extent_axis_raises(self):
        with pytest.raises(ValueError, match="zero voxels"):
            Occupancy3DBuilder(Occupancy3DParams(
                resolution_m=1.0, size_x_m=0.0, size_y_m=2.0, size_z_m=2.0,
            ))


# ---------------------------------------------------------------------------
# Stamping
# ---------------------------------------------------------------------------

class TestStamping:

    def _builder(self):
        return Occupancy3DBuilder(Occupancy3DParams(
            resolution_m=0.1, size_x_m=4.0, size_y_m=4.0, size_z_m=4.0,
            origin_x_m=-2.0, origin_y_m=-2.0, origin_z_m=-2.0,
            default_inflation_m=0.5,
            per_class_inflation_m={"person": 0.3},
        ))

    def test_object_at_origin_marks_centre_voxel(self):
        b = self._builder()
        obj = make_obj(position_world=[0.0, 0.0, 0.0])
        result = b.build([obj])
        # Centre cell of a 40x40x40 volume with origin at (-2,-2,-2) is
        # (20, 20, 20).
        assert (20, 20, 20) in result.occupied_voxels
        assert result.occupied_voxels[(20, 20, 20)] == OCCUPIED

    def test_object_outside_volume_yields_no_voxels(self):
        b = self._builder()
        # Object at world (100, 100, 100) is way outside the 4x4x4 m volume.
        obj = make_obj(position_world=[100.0, 100.0, 100.0])
        result = b.build([obj])
        assert len(result) == 0

    def test_no_objects_yields_empty_grid(self):
        b = self._builder()
        result = b.build([])
        assert len(result) == 0
        assert result.occupied_voxels == {}

    def test_object_without_position_world_skipped(self):
        b = self._builder()
        obj = make_obj(position_world=None)
        result = b.build([obj])
        assert len(result) == 0

    def test_sphere_inflation_is_spherical_not_cubic(self):
        """
        For a sphere of radius R cells, voxels in the bounding cube
        corners must be excluded. Check a corner voxel is NOT marked.
        """
        b = Occupancy3DBuilder(Occupancy3DParams(
            resolution_m=0.1, size_x_m=4.0, size_y_m=4.0, size_z_m=4.0,
            origin_x_m=-2.0, origin_y_m=-2.0, origin_z_m=-2.0,
            per_class_inflation_m={"person": 0.5},  # 5-voxel radius
        ))
        obj = make_obj(position_world=[0.0, 0.0, 0.0], class_name="person")
        result = b.build([obj])
        centre_cell = (20, 20, 20)
        assert centre_cell in result.occupied_voxels
        # Corner of the bounding cube — at sqrt(3)*5 ≈ 8.66 cells, way
        # outside the 5-cell sphere.
        corner = (25, 25, 25)
        assert corner not in result.occupied_voxels
        # On-axis cell at radius+0 should be inside.
        on_axis = (25, 20, 20)
        # Inside or just outside is fine; the half-cell-offset rule
        # makes this borderline. Use a clearly-inside cell instead.
        clearly_inside = (22, 20, 20)
        assert clearly_inside in result.occupied_voxels


# ---------------------------------------------------------------------------
# Inflation choice
# ---------------------------------------------------------------------------

class TestInflationChoice:

    def _params(self):
        return Occupancy3DParams(
            resolution_m=0.05, size_x_m=10.0, size_y_m=10.0, size_z_m=4.0,
            origin_x_m=-5.0, origin_y_m=-5.0, origin_z_m=-2.0,
            min_inflation_m=0.10,
            per_class_inflation_m={"person": 0.40, "car": 2.00},
            default_inflation_m=0.5,
        )

    def test_per_class_used_when_no_intrinsics(self):
        b = Occupancy3DBuilder(self._params())
        obj = make_obj(class_name="car",
                       position_world=[0.0, 0.0, 0.0],
                       position_3d=[0.0, 0.0, 3.0], std_px=2.0)
        result = b.build([obj])
        # 'car' → 2.0 m radius. Voxel-count at 5 cm res is large (~33 K),
        # just assert presence and that 'person' would have given fewer.
        n_car = len(result)

        b2 = Occupancy3DBuilder(self._params())
        obj2 = make_obj(class_name="person",
                        position_world=[0.0, 0.0, 0.0],
                        position_3d=[0.0, 0.0, 3.0], std_px=2.0)
        n_person = len(b2.build([obj2]))
        assert n_car > n_person * 50  # 2.0 m vs 0.4 m → ~125× volume

    def test_default_when_class_not_in_table(self):
        b = Occupancy3DBuilder(self._params())
        obj = make_obj(class_name="aardvark",
                       position_world=[0.0, 0.0, 0.0])
        result = b.build([obj])
        # default 0.5 m → smaller than 'car' (2.0 m), larger than 'person' (0.4 m).
        n_default = len(result)
        n_person = len(Occupancy3DBuilder(self._params()).build(
            [make_obj(class_name="person", position_world=[0.0, 0.0, 0.0])]
        ))
        assert n_default > n_person

    def test_depth_projected_inflation_used_with_intrinsics(self):
        intr = intr_640x480(fx=500.0)
        b = Occupancy3DBuilder(self._params(), intrinsics=intr)
        # std_px=10 at depth 4 m → 2*10*4/500 = 0.16 m radius. Larger
        # than min_inflation_m=0.10 so the metric path wins.
        obj = make_obj(class_name="person",
                       position_world=[0.0, 0.0, 0.0],
                       position_3d=[0.0, 0.0, 4.0], std_px=10.0)
        # Same object, no intrinsics → falls back to per-class 0.4 m.
        b_no_intr = Occupancy3DBuilder(self._params(), intrinsics=None)
        n_metric = len(b.build([obj]))
        n_class  = len(b_no_intr.build([obj]))
        # 0.16 m radius is smaller than 0.4 m class fallback → fewer voxels.
        assert n_metric < n_class

    def test_min_inflation_floor_enforced(self):
        intr = intr_640x480(fx=500.0)
        # std_px=0.1 at depth 1 m → 2*0.1*1/500 = 4e-4 m. Below the
        # 0.10 m floor — must be clamped, not collapse to a single voxel.
        params = self._params()
        b = Occupancy3DBuilder(params, intrinsics=intr)
        obj = make_obj(class_name="person",
                       position_world=[0.0, 0.0, 0.0],
                       position_3d=[0.0, 0.0, 1.0], std_px=0.1)
        result = b.build([obj])
        # At 0.10 m radius and 0.05 m res → ~33 voxels in the sphere.
        # Just assert it's substantially more than the 1 you'd get
        # without the floor.
        assert len(result) > 5


# ---------------------------------------------------------------------------
# End-to-end build
# ---------------------------------------------------------------------------

class TestBuild:

    def test_multiple_objects_independently_stamped(self):
        b = Occupancy3DBuilder(Occupancy3DParams(
            resolution_m=0.1, size_x_m=10.0, size_y_m=10.0, size_z_m=4.0,
            origin_x_m=-5.0, origin_y_m=-5.0, origin_z_m=-2.0,
            per_class_inflation_m={"person": 0.2},
        ))
        # Two non-overlapping people far apart in XY.
        objs = [
            make_obj(track_id=1, position_world=[-2.0, 0.0, 0.0]),
            make_obj(track_id=2, position_world=[+2.0, 0.0, 0.0]),
        ]
        result = b.build(objs)
        # Each gets a 2-voxel-radius sphere → 2 disjoint clusters.
        # Sanity: both centres present.
        c1 = b.world_to_cell(-2.0, 0.0, 0.0)
        c2 = b.world_to_cell(+2.0, 0.0, 0.0)
        assert c1 in result.occupied_voxels
        assert c2 in result.occupied_voxels

    def test_2d_only_position_world_uses_volume_centre_z(self):
        """
        position_world shape (2,) → no Z. The builder should not crash;
        it picks the volume's Z midpoint and stamps there.
        """
        b = Occupancy3DBuilder(Occupancy3DParams(
            resolution_m=0.1, size_x_m=4.0, size_y_m=4.0, size_z_m=2.0,
            origin_x_m=-2.0, origin_y_m=-2.0, origin_z_m=0.0,
            per_class_inflation_m={"person": 0.2},
        ))
        # Build an ObjectState whose position_world is genuinely 2D.
        obj = make_obj(position_world=[0.0, 0.0])
        # The helper turns it into a (2,) array.
        assert obj.position_world.shape == (2,)
        result = b.build([obj])
        # Volume centre Z = origin_z + size_z/2 = 0 + 1 = 1.0 m.
        # That cell is k = 10 in a 20-tall stack.
        cells_at_k10 = [k for (_, _, k) in result.occupied_voxels.keys()]
        assert all(7 <= k <= 13 for k in cells_at_k10), (
            f"Expected voxels clustered around k=10, got ks={sorted(set(cells_at_k10))}"
        )


# ---------------------------------------------------------------------------
# Occupancy3D helpers
# ---------------------------------------------------------------------------

class TestOccupancy3D:

    def test_voxel_centres_world_empty(self):
        params = Occupancy3DParams()
        o = Occupancy3D(occupied_voxels={}, params=params)
        centres = o.voxel_centres_world()
        assert centres.shape == (0, 3)

    def test_voxel_centres_world_round_trip(self):
        """
        A voxel at index (i,j,k) should map back to a world position
        within half a voxel of the cell centre.
        """
        b = Occupancy3DBuilder(Occupancy3DParams(
            resolution_m=0.5, size_x_m=10.0, size_y_m=10.0, size_z_m=4.0,
            origin_x_m=-5.0, origin_y_m=-5.0, origin_z_m=-2.0,
            per_class_inflation_m={"person": 0.25},  # < 1 voxel → 1-voxel hit
        ))
        # Build with a single object exactly at a cell centre.
        target_xyz = (-2.75, 1.25, 0.25)
        result = b.build([make_obj(position_world=list(target_xyz))])
        centres = result.voxel_centres_world()
        # At least one returned centre should match the target within
        # half a voxel along every axis.
        diffs = np.abs(centres - np.asarray(target_xyz))
        min_max_diff = float(np.min(np.max(diffs, axis=1)))
        assert min_max_diff <= 0.5 / 2 + 1e-9
