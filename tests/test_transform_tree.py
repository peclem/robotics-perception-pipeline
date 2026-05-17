"""
Unit tests for the coordinate frame transform tree and pose-aware
SceneGraph integration.

TestTransform           : SE(3) primitive — compose, inverse, point transform
TestTransformTreeBasic  : single-edge and identity lookups
TestTransformTreeChain  : multi-edge composition along robot kinematic chain
TestTransformTreeErrors : invalid inputs, disconnected frames, reparenting
TestSceneGraphWorld     : ObjectState.position_world from pose / tree
TestSceneGraphQueryWorld: world-frame query_nearby
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from perception.pose_estimator import CameraPose
from perception.transform_tree import Transform, TransformTree
from tracking.track import Track, TrackState
from world_model.scene_graph import SceneGraph
from perception.detector import Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rot_z(theta: float) -> np.ndarray:
    """Rotation about Z (world up) by `theta` radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


@pytest.fixture(autouse=True)
def reset_ids():
    Track.reset_id_counter()
    yield
    Track.reset_id_counter()


@pytest.fixture
def world_cfg():
    return {
        "world_model": {
            "max_history":    10,
            "lost_timeout_s": 1.0,
            "camera_frame":   "camera_frame",
        },
        "kalman_filter": {
            "initial_covariance": {
                "p_position": 10.0, "p_size": 10.0, "p_velocity": 100.0,
            },
            "process_noise": {
                "q_position": 1.0, "q_size": 1.0,
                "q_velocity": 0.1, "q_vel_size": 0.02,
            },
            "measurement_noise": {"r_center": 1.0, "r_size": 1.0},
        },
        "tracker": {
            "high_thresh": 0.5, "low_thresh": 0.1,
            "new_track_thresh": 0.5, "iou_threshold": 0.3,
            "max_age": 5, "min_hits": 1, "use_ekf": False,
        },
    }


def make_confirmed_track(cfg, cx=150, cy=150, cls_name="person") -> Track:
    det = Detection(
        bbox_xyxy=np.array([cx-30, cy-40, cx+30, cy+40], dtype=np.float32),
        confidence=0.9, class_id=0, class_name=cls_name,
        frame_idx=0, timestamp=time.monotonic(),
    )
    t = Track(det, cfg)
    t.state = TrackState.CONFIRMED
    return t


class _FakeDepthEstimate:
    """Minimal duck-typed stand-in for the depth_estimator's DepthEstimate."""
    def __init__(self, x: float, y: float, z: float):
        self.position_3d = np.array([x, y, z], dtype=np.float64)


# ---------------------------------------------------------------------------
# Transform primitive
# ---------------------------------------------------------------------------

class TestTransform:

    def test_identity_leaves_point_unchanged(self):
        T = Transform.identity("a", "a")
        p = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(T.transform_point(p), p)

    def test_inverse_round_trip(self):
        T = Transform(R=rot_z(0.7), t=np.array([1.0, -2.0, 0.5]),
                      target_frame="b", source_frame="a")
        p = np.array([0.3, 0.4, 0.5])
        p_b   = T.transform_point(p)
        p_back = T.inverse().transform_point(p_b)
        np.testing.assert_allclose(p_back, p, atol=1e-12)

    def test_inverse_swaps_frames(self):
        T = Transform(R=np.eye(3), t=np.zeros(3),
                      target_frame="b", source_frame="a")
        T_inv = T.inverse()
        assert T_inv.target_frame == "a"
        assert T_inv.source_frame == "b"

    def test_compose_chain(self):
        # b ← a: rotate 90° about z, no translation
        T_ba = Transform(R=rot_z(np.pi/2), t=np.zeros(3),
                         target_frame="b", source_frame="a")
        # c ← b: translate +1 along x
        T_cb = Transform(R=np.eye(3), t=np.array([1.0, 0.0, 0.0]),
                         target_frame="c", source_frame="b")
        # c ← a: T_cb ∘ T_ba
        T_ca = T_cb.compose(T_ba)
        assert T_ca.target_frame == "c"
        assert T_ca.source_frame == "a"
        # Point [1,0,0]_a → [0,1,0]_b → [1,1,0]_c
        np.testing.assert_allclose(
            T_ca.transform_point(np.array([1.0, 0.0, 0.0])),
            np.array([1.0, 1.0, 0.0]),
            atol=1e-12,
        )

    def test_compose_rejects_frame_mismatch(self):
        T1 = Transform(R=np.eye(3), t=np.zeros(3),
                       target_frame="c", source_frame="b")
        T2 = Transform(R=np.eye(3), t=np.zeros(3),
                       target_frame="x", source_frame="a")
        with pytest.raises(ValueError, match="Cannot compose"):
            T1.compose(T2)

    def test_transform_points_batched(self):
        T = Transform(R=rot_z(np.pi/2), t=np.array([1.0, 2.0, 3.0]),
                      target_frame="b", source_frame="a")
        pts = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0]])
        out = T.transform_points(pts)
        # rot_z(pi/2): [1,0,0]→[0,1,0]; [0,1,0]→[-1,0,0]
        # Then + [1,2,3]
        expected = np.array([[1.0, 3.0, 3.0],
                             [0.0, 2.0, 3.0]])
        np.testing.assert_allclose(out, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# TransformTree — single edge / identity
# ---------------------------------------------------------------------------

class TestTransformTreeBasic:

    def test_root_only_has_no_parent(self):
        tree = TransformTree(root_frame="map")
        assert tree.root == "map"
        assert tree.frames == ["map"]

    def test_identity_lookup_same_frame(self):
        tree = TransformTree(root_frame="map")
        T = tree.lookup_transform("map", "map")
        np.testing.assert_allclose(T.R, np.eye(3))
        np.testing.assert_allclose(T.t, np.zeros(3))

    def test_static_edge_lookup(self):
        tree = TransformTree(root_frame="map")
        tree.set_static(parent="map", child="base_link",
                        R=np.eye(3), t=np.array([5.0, 0.0, 0.0]))
        T = tree.lookup_transform("map", "base_link")
        np.testing.assert_allclose(T.t, [5.0, 0.0, 0.0])

    def test_reverse_lookup_inverts(self):
        tree = TransformTree(root_frame="map")
        tree.set_static(parent="map", child="base_link",
                        R=np.eye(3), t=np.array([5.0, 0.0, 0.0]))
        T = tree.lookup_transform("base_link", "map")
        np.testing.assert_allclose(T.t, [-5.0, 0.0, 0.0])

    def test_transform_point_convenience(self):
        tree = TransformTree(root_frame="map")
        tree.set_static(parent="map", child="base_link",
                        R=np.eye(3), t=np.array([5.0, 0.0, 0.0]))
        p = tree.transform_point(np.array([1.0, 0.0, 0.0]),
                                 target="map", source="base_link")
        np.testing.assert_allclose(p, [6.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# TransformTree — multi-edge robot chain
# ---------------------------------------------------------------------------

class TestTransformTreeChain:

    def _build_chain(self) -> TransformTree:
        """map ← odom ← base_link ← camera_frame, all distinct."""
        tree = TransformTree(root_frame="map")
        # map ← odom: identity (no SLAM loop closures yet)
        tree.set_dynamic("map", "odom",
                         R=np.eye(3), t=np.zeros(3), timestamp=0.0)
        # odom ← base_link: robot 2m ahead, rotated 90° about z
        tree.set_dynamic("odom", "base_link",
                         R=rot_z(np.pi/2),
                         t=np.array([2.0, 0.0, 0.0]),
                         timestamp=1.0)
        # base_link ← camera_frame: camera 0.1m up on the body
        tree.set_static("base_link", "camera_frame",
                        R=np.eye(3),
                        t=np.array([0.0, 0.0, 0.1]))
        return tree

    def test_camera_to_map_composes_full_chain(self):
        tree = self._build_chain()
        # Origin of camera_frame → at base_link [0,0,0.1]
        #   → rotated 90° about z (no translation effect on origin) +[2,0,0]
        #   = [2, 0, 0.1] in odom = [2, 0, 0.1] in map (odom→map identity)
        p_map = tree.transform_point(np.zeros(3),
                                     target="map", source="camera_frame")
        np.testing.assert_allclose(p_map, [2.0, 0.0, 0.1], atol=1e-12)

    def test_chain_point_rotation_propagates(self):
        tree = self._build_chain()
        # Point [1, 0, 0] in camera_frame
        # In base_link: [1, 0, 0.1]
        # rot_z(90°) of [1, 0, 0.1] = [0, 1, 0.1], + [2,0,0] = [2, 1, 0.1]
        p_map = tree.transform_point(np.array([1.0, 0.0, 0.0]),
                                     target="map", source="camera_frame")
        np.testing.assert_allclose(p_map, [2.0, 1.0, 0.1], atol=1e-12)

    def test_round_trip_via_root(self):
        tree = self._build_chain()
        p_cam = np.array([0.3, -0.2, 1.5])
        p_map = tree.transform_point(p_cam, "map", "camera_frame")
        p_back = tree.transform_point(p_map, "camera_frame", "map")
        np.testing.assert_allclose(p_back, p_cam, atol=1e-12)

    def test_dynamic_update_overwrites_previous(self):
        tree = TransformTree(root_frame="map")
        tree.set_dynamic("map", "base_link",
                         R=np.eye(3), t=np.array([1.0, 0.0, 0.0]),
                         timestamp=0.0)
        tree.set_dynamic("map", "base_link",
                         R=np.eye(3), t=np.array([5.0, 0.0, 0.0]),
                         timestamp=1.0)
        T = tree.lookup_transform("map", "base_link")
        np.testing.assert_allclose(T.t, [5.0, 0.0, 0.0])

    def test_lookup_between_siblings(self):
        # base_link and camera_frame both descend from map via odom.
        # Lookup base_link → camera_frame should still resolve via the
        # common ancestor.
        tree = self._build_chain()
        T = tree.lookup_transform("base_link", "camera_frame")
        # base_link ← camera_frame is the static edge.
        np.testing.assert_allclose(T.t, [0.0, 0.0, 0.1])

    def test_update_from_camera_pose_writes_edge(self):
        tree = TransformTree(root_frame="map")
        pose = CameraPose(
            R=rot_z(np.pi/4),
            t=np.array([3.0, 4.0, 0.0]),
            timestamp=2.5,
            frame_idx=10,
            source="test",
        )
        tree.update_from_camera_pose(pose, camera_frame="camera_frame")
        T = tree.lookup_transform("map", "camera_frame")
        np.testing.assert_allclose(T.R, rot_z(np.pi/4))
        np.testing.assert_allclose(T.t, [3.0, 4.0, 0.0])


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestTransformTreeErrors:

    def test_unknown_target_raises(self):
        tree = TransformTree(root_frame="map")
        with pytest.raises(KeyError, match="ghost"):
            tree.lookup_transform("ghost", "map")

    def test_unknown_source_raises(self):
        tree = TransformTree(root_frame="map")
        with pytest.raises(KeyError, match="ghost"):
            tree.lookup_transform("map", "ghost")

    def test_reparenting_rejected(self):
        tree = TransformTree(root_frame="map")
        tree.set_static("map", "base_link",
                        R=np.eye(3), t=np.zeros(3))
        with pytest.raises(ValueError, match="already has parent"):
            tree.set_static("odom", "base_link",
                            R=np.eye(3), t=np.zeros(3))

    def test_cannot_reparent_root(self):
        tree = TransformTree(root_frame="map")
        with pytest.raises(ValueError, match="root"):
            tree.set_static("odom", "map",
                            R=np.eye(3), t=np.zeros(3))

    def test_update_from_none_pose_is_noop(self):
        tree = TransformTree(root_frame="map")
        tree.update_from_camera_pose(None)
        assert tree.frames == ["map"]


# ---------------------------------------------------------------------------
# SceneGraph integration — position_world via pose or tree
# ---------------------------------------------------------------------------

class TestSceneGraphWorld:

    def test_no_pose_leaves_position_world_none(self, world_cfg):
        sg = SceneGraph(world_cfg)
        tr = make_confirmed_track(world_cfg)
        depth = {tr.track_id: _FakeDepthEstimate(1.0, 0.5, 3.0)}
        sg.update([tr], [], timestamp=time.monotonic(),
                  depth_estimates=depth, camera_pose=None)
        obj = sg.get_state(tr.track_id)
        assert obj.position_3d is not None
        assert obj.position_world is None

    def test_camera_pose_populates_position_world(self, world_cfg):
        sg = SceneGraph(world_cfg)
        tr = make_confirmed_track(world_cfg)
        depth = {tr.track_id: _FakeDepthEstimate(1.0, 0.0, 3.0)}
        pose = CameraPose(
            R=np.eye(3), t=np.array([10.0, 20.0, 0.0]),
            timestamp=0.0, frame_idx=0, source="test",
        )
        sg.update([tr], [], timestamp=time.monotonic(),
                  depth_estimates=depth, camera_pose=pose)
        obj = sg.get_state(tr.track_id)
        np.testing.assert_allclose(obj.position_world, [11.0, 20.0, 3.0])

    def test_transform_tree_path_takes_precedence(self, world_cfg):
        tree = TransformTree(root_frame="map")
        # base_link 5m ahead in map, no rotation
        tree.set_dynamic("map", "base_link",
                         R=np.eye(3), t=np.array([5.0, 0.0, 0.0]),
                         timestamp=0.0)
        # camera mounted 0.1m above base_link
        tree.set_static("base_link", "camera_frame",
                        R=np.eye(3), t=np.array([0.0, 0.0, 0.1]))
        sg = SceneGraph(world_cfg, transform_tree=tree)

        tr = make_confirmed_track(world_cfg)
        depth = {tr.track_id: _FakeDepthEstimate(1.0, 0.0, 2.0)}
        # camera_pose intentionally inconsistent — tree should win
        bogus_pose = CameraPose(
            R=np.eye(3), t=np.array([-999.0, -999.0, -999.0]),
            timestamp=0.0, frame_idx=0, source="bogus",
        )
        sg.update([tr], [], timestamp=time.monotonic(),
                  depth_estimates=depth, camera_pose=bogus_pose)
        obj = sg.get_state(tr.track_id)
        # Camera-frame [1, 0, 2] → base_link [1, 0, 2.1] → map [6, 0, 2.1]
        np.testing.assert_allclose(obj.position_world, [6.0, 0.0, 2.1])


# ---------------------------------------------------------------------------
# World-frame query_nearby
# ---------------------------------------------------------------------------

class TestSceneGraphQueryWorld:

    def _populate(self, world_cfg, positions_world):
        """Create SceneGraph with one object per provided world position."""
        sg = SceneGraph(world_cfg)
        ts = time.monotonic()
        pose = CameraPose(
            R=np.eye(3), t=np.zeros(3),
            timestamp=ts, frame_idx=0, source="test",
        )
        tracks = []
        depths = {}
        for i, pw in enumerate(positions_world):
            tr = make_confirmed_track(world_cfg, cx=100 + i*10, cy=100)
            tracks.append(tr)
            depths[tr.track_id] = _FakeDepthEstimate(*pw)
        sg.update(tracks, [], timestamp=ts,
                  depth_estimates=depths, camera_pose=pose)
        return sg

    def test_world_query_returns_nearby_only(self, world_cfg):
        sg = self._populate(world_cfg, [
            (1.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
        ])
        hits = sg.query_nearby(
            np.zeros(3), radius=2.0, frame="world",
        )
        # Two objects within 2m of origin.
        assert len(hits) == 2
        assert hits[0][0] <= hits[1][0]   # sorted

    def test_world_query_skips_objects_without_world_position(self, world_cfg):
        sg = SceneGraph(world_cfg)
        ts = time.monotonic()
        tr = make_confirmed_track(world_cfg)
        # No camera_pose → position_world stays None
        sg.update([tr], [], timestamp=ts,
                  depth_estimates={tr.track_id: _FakeDepthEstimate(0.1, 0, 0)},
                  camera_pose=None)
        hits = sg.query_nearby(np.zeros(3), radius=10.0, frame="world")
        assert hits == []

    def test_camera_frame_query_still_works(self, world_cfg):
        # Regression guard for the legacy 2D query path.
        sg = SceneGraph(world_cfg)
        tr = make_confirmed_track(world_cfg, cx=150, cy=150)
        sg.update([tr], [], timestamp=time.monotonic())
        hits = sg.query_nearby(np.array([150.0, 150.0]), radius=10.0)
        assert len(hits) == 1

    def test_invalid_frame_raises(self, world_cfg):
        sg = SceneGraph(world_cfg)
        with pytest.raises(ValueError, match="frame must be"):
            sg.query_nearby(np.zeros(3), radius=1.0, frame="lidar")
