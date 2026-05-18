"""
Unit tests for the RoomLayer (3D scene graph room hierarchy).

TestRoomDataclass     : Room invariants + containment
TestRoomExtraction    : occupancy grid → list of rooms
TestObjectAssignment  : ObjectState.position_world → room membership
TestQueries           : room_of_object, room_of_world_xy
TestEdgeCases         : empty grid, all-occupied, tiny components
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from world_model.occupancy_grid import OCCUPIED, OccupancyGridParams
from world_model.object_state import ObjectState
from world_model.room_layer import Room, RoomLayer, RoomLayerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params(
    res: float = 0.1, sx: float = 10.0, sy: float = 10.0,
    ox: float = 0.0, oy: float = 0.0,
) -> OccupancyGridParams:
    return OccupancyGridParams(
        resolution_m=res, size_x_m=sx, size_y_m=sy,
        origin_x_m=ox, origin_y_m=oy,
    )


def _grid(params: OccupancyGridParams) -> np.ndarray:
    """Empty (all-free) grid at the given size."""
    H = int(round(params.size_y_m / params.resolution_m))
    W = int(round(params.size_x_m / params.resolution_m))
    return np.zeros((H, W), dtype=np.int8)


def _two_rooms_with_doorway(
    params: OccupancyGridParams, doorway_cells: int = 4,
) -> np.ndarray:
    """
    Build a 10×10 m grid containing two square rooms separated by a
    vertical wall down the middle, with a small doorway gap. Rooms
    occupy the full grid except for the centre wall.

    Wall: column at x ≈ 5 m (centre), full height except for a
    `doorway_cells`-tall gap centred vertically.
    """
    g = _grid(params)
    H, W = g.shape
    # World x=5 → col 50 at 0.1 m/cell with origin 0
    wall_col = int(round((5.0 - params.origin_x_m) / params.resolution_m))
    g[:, wall_col] = OCCUPIED
    # Doorway: clear `doorway_cells` rows in the centre of that column.
    dh = doorway_cells // 2
    g[H // 2 - dh: H // 2 + dh, wall_col] = 0
    return g


def _obj(
    track_id: int = 1, x: float = 0.0, y: float = 0.0, z: float = 0.0,
) -> ObjectState:
    return ObjectState(
        track_id=track_id, class_id=0, class_name="person",
        position=np.array([0.0, 0.0]),
        covariance=np.eye(8),
        velocity=np.zeros(4),
        score=0.9, last_seen=time.monotonic(), n_updates=1,
        position_world=np.array([x, y, z], dtype=np.float64),
        max_history=1,
    )


# ---------------------------------------------------------------------------
# Room dataclass
# ---------------------------------------------------------------------------

class TestRoomDataclass:

    def _square(self):
        poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
        return Room(
            room_id=1, polygon_world=poly,
            centroid=np.array([5.0, 5.0]),
            area_m2=100.0, bbox_world=(0.0, 0.0, 10.0, 10.0),
        )

    def test_default_label_uses_room_id(self):
        r = self._square()
        assert r.label == "room_1"

    def test_polygon_shape_enforced(self):
        with pytest.raises(AssertionError):
            Room(
                room_id=1,
                polygon_world=np.array([1.0, 2.0, 3.0]),
                centroid=np.zeros(2),
                area_m2=1.0, bbox_world=(0, 0, 1, 1),
            )

    def test_contains_world_xy_inside(self):
        r = self._square()
        assert r.contains_world_xy(5.0, 5.0)

    def test_contains_world_xy_outside_bbox(self):
        r = self._square()
        assert not r.contains_world_xy(50.0, 50.0)

    def test_contains_world_xy_inside_bbox_outside_polygon(self):
        # L-shaped polygon: bbox = (0,0,10,10) but cell (8, 8) is outside.
        poly = np.array([
            [0, 0], [5, 0], [5, 5], [10, 5],
            [10, 10], [0, 10],
        ], dtype=np.float64)
        r = Room(
            room_id=2, polygon_world=poly,
            centroid=np.array([4.0, 5.0]),
            area_m2=75.0, bbox_world=(0.0, 0.0, 10.0, 10.0),
        )
        assert r.contains_world_xy(2.0, 2.0)       # in lower stem
        assert not r.contains_world_xy(8.0, 2.0)   # bbox-in, polygon-out


# ---------------------------------------------------------------------------
# Room extraction
# ---------------------------------------------------------------------------

class TestRoomExtraction:

    def test_single_free_grid_yields_one_room(self):
        params = _params()
        layer = RoomLayer(RoomLayerConfig(erosion_m=0.1, min_area_m2=1.0))
        layer.update(_grid(params), params, objects=[], frame_idx=0)
        assert len(layer.rooms) == 1
        assert layer.rooms[0].area_m2 > 50.0   # ~100 m² minus erosion

    def test_two_rooms_with_thin_doorway_split_apart(self):
        params = _params()
        g = _two_rooms_with_doorway(params, doorway_cells=4)  # 40 cm wide
        # erosion_m=0.45 → erosion radius = 0.45 / 0.1 = 4-5 cells → closes
        # a 40 cm doorway.
        layer = RoomLayer(RoomLayerConfig(erosion_m=0.45, min_area_m2=1.0))
        layer.update(g, params, objects=[], frame_idx=0)
        assert len(layer.rooms) == 2

    def test_open_doorway_keeps_single_room_when_erosion_too_small(self):
        params = _params()
        g = _two_rooms_with_doorway(params, doorway_cells=20)  # 2 m wide
        # Even with erosion=0.45 the 2 m gap stays open.
        layer = RoomLayer(RoomLayerConfig(erosion_m=0.45, min_area_m2=1.0))
        layer.update(g, params, objects=[], frame_idx=0)
        assert len(layer.rooms) == 1

    def test_min_area_filters_tiny_components(self):
        params = _params(res=0.1, sx=5.0, sy=5.0)
        g = _grid(params)
        # Isolate a single 1-cell free island surrounded by wall:
        # mark everything OCCUPIED, then unmark one cell.
        g[:] = OCCUPIED
        g[10, 10] = 0
        layer = RoomLayer(RoomLayerConfig(erosion_m=0.05, min_area_m2=0.5))
        layer.update(g, params, objects=[], frame_idx=0)
        # 1 cell = 0.01 m² → way below min_area_m2 → no rooms.
        assert len(layer.rooms) == 0

    def test_all_occupied_yields_no_rooms(self):
        params = _params()
        g = np.full_like(_grid(params), OCCUPIED)
        layer = RoomLayer()
        layer.update(g, params, objects=[], frame_idx=5)
        assert layer.rooms == []
        assert layer.last_frame_idx == 5

    def test_empty_grid_yields_no_rooms(self):
        params = _params()
        layer = RoomLayer()
        layer.update(np.zeros((0, 0), dtype=np.int8), params, [], 0)
        assert layer.rooms == []

    def test_room_centroid_inside_polygon(self):
        params = _params()
        layer = RoomLayer(RoomLayerConfig(erosion_m=0.1, min_area_m2=1.0))
        layer.update(_grid(params), params, objects=[], frame_idx=0)
        for r in layer.rooms:
            assert r.contains_world_xy(*r.centroid)


# ---------------------------------------------------------------------------
# Object assignment + queries
# ---------------------------------------------------------------------------

class TestObjectAssignment:

    def _layer_two_rooms(self):
        params = _params()
        g = _two_rooms_with_doorway(params, doorway_cells=4)
        layer = RoomLayer(RoomLayerConfig(erosion_m=0.45, min_area_m2=1.0))
        return layer, g, params

    def test_object_in_left_room_assigned(self):
        layer, g, params = self._layer_two_rooms()
        objs = [_obj(track_id=1, x=2.0, y=5.0)]   # left half (x<5)
        layer.update(g, params, objs, frame_idx=0)
        r = layer.room_of_object(1)
        assert r is not None
        # Centroid should also be in the left half.
        assert r.centroid[0] < 5.0

    def test_two_objects_in_different_rooms(self):
        layer, g, params = self._layer_two_rooms()
        objs = [
            _obj(track_id=1, x=2.0, y=5.0),   # left
            _obj(track_id=2, x=7.0, y=5.0),   # right
        ]
        layer.update(g, params, objs, frame_idx=0)
        r1 = layer.room_of_object(1)
        r2 = layer.room_of_object(2)
        assert r1 is not None and r2 is not None
        assert r1.room_id != r2.room_id

    def test_object_outside_any_room_returns_none(self):
        layer, g, params = self._layer_two_rooms()
        # Object out of grid extent entirely.
        objs = [_obj(track_id=1, x=100.0, y=100.0)]
        layer.update(g, params, objs, frame_idx=0)
        assert layer.room_of_object(1) is None

    def test_object_without_position_world_skipped(self):
        layer, g, params = self._layer_two_rooms()
        obj = ObjectState(
            track_id=1, class_id=0, class_name="person",
            position=np.array([0.0, 0.0]),
            covariance=np.eye(8), velocity=np.zeros(4),
            score=0.9, last_seen=time.monotonic(), n_updates=1,
            position_world=None, max_history=1,
        )
        layer.update(g, params, [obj], frame_idx=0)
        assert layer.room_of_object(1) is None
        # And it's not in any room's object_ids either.
        assert all(1 not in r.object_ids for r in layer.rooms)

    def test_room_object_ids_populated(self):
        layer, g, params = self._layer_two_rooms()
        objs = [_obj(track_id=1, x=2.0, y=5.0)]
        layer.update(g, params, objs, frame_idx=0)
        # Exactly one room should have 1 in its object_ids set.
        owners = [r for r in layer.rooms if 1 in r.object_ids]
        assert len(owners) == 1


# ---------------------------------------------------------------------------
# Spatial queries
# ---------------------------------------------------------------------------

class TestQueries:

    def test_room_of_world_xy_correct_side(self):
        params = _params()
        g = _two_rooms_with_doorway(params, doorway_cells=4)
        layer = RoomLayer(RoomLayerConfig(erosion_m=0.45, min_area_m2=1.0))
        layer.update(g, params, objects=[], frame_idx=0)
        r_left  = layer.room_of_world_xy(2.0, 5.0)
        r_right = layer.room_of_world_xy(7.0, 5.0)
        assert r_left  is not None
        assert r_right is not None
        assert r_left.room_id != r_right.room_id

    def test_room_of_world_xy_outside_returns_none(self):
        params = _params()
        layer = RoomLayer()
        layer.update(_grid(params), params, [], 0)
        assert layer.room_of_world_xy(100.0, 100.0) is None

    def test_room_of_object_unknown_id(self):
        params = _params()
        layer = RoomLayer()
        layer.update(_grid(params), params, [], 0)
        assert layer.room_of_object(999) is None
