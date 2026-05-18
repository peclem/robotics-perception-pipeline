"""
Unit tests for the world model scene graph.

TestObjectState     : probabilistic state container
TestSceneGraphUpdate: update cycle from tracker output
TestSceneGraphQuery : spatial query interface
TestSceneGraphPrune : stale object removal
TestSceneGraphProps : properties and summary
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import pytest

from tracking.track import Track, TrackState
from world_model.object_state import ObjectState
from world_model.scene_graph import SceneGraph
from perception.detector import Detection


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_ids():
    Track.reset_id_counter()
    yield
    Track.reset_id_counter()


@pytest.fixture
def cfg():
    return {
        "world_model": {
            "max_history":    10,
            "lost_timeout_s": 1.0,
        },
        "tracker": {
            "high_thresh":      0.50,
            "low_thresh":       0.10,
            "new_track_thresh": 0.50,
            "iou_threshold":    0.30,
            "max_age":          5,
            "min_hits":         1,
            "use_ekf":          False,
        },
        "kalman_filter": {
            "initial_covariance": {
                "p_position": 10.0,
                "p_size":     10.0,
                "p_velocity": 100.0,
            },
            "process_noise": {
                "q_position": 1.0,
                "q_size":     1.0,
                "q_velocity": 0.1,
                "q_vel_size": 0.02,
            },
            "measurement_noise": {
                "r_center": 1.0,
                "r_size":   1.0,
            },
        },
    }


def make_detection(
    x1=100, y1=100, x2=200, y2=200,
    conf=0.9, cls_id=0, cls_name="person",
) -> Detection:
    return Detection(
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        confidence=conf, class_id=cls_id, class_name=cls_name,
        frame_idx=0, timestamp=time.monotonic(),
    )


def make_confirmed_track(cfg, cx=150, cy=150, cls_name="person") -> Track:
    det = make_detection(
        x1=cx-30, y1=cy-40, x2=cx+30, y2=cy+40,
        cls_name=cls_name,
    )
    t = Track(det, cfg)
    t.state = TrackState.CONFIRMED
    return t


def make_object_state(
    track_id=1, cx=100.0, cy=100.0, is_lost=False
) -> ObjectState:
    return ObjectState(
        track_id=track_id,
        class_id=0,
        class_name="person",
        position=np.array([cx, cy]),
        covariance=np.eye(8) * 10.0,
        velocity=np.zeros(4),
        score=0.9,
        last_seen=time.monotonic(),
        n_updates=1,
        is_lost=is_lost,
        max_history=10,
    )


# ---------------------------------------------------------------------------
# TestObjectState
# ---------------------------------------------------------------------------

class TestObjectState:

    def test_position_std_positive(self):
        obj = make_object_state()
        std = obj.position_std
        assert std.shape == (2,)
        assert np.all(std > 0)

    def test_position_std_decreases_with_smaller_covariance(self):
        obj_large = make_object_state()
        obj_large.covariance = np.eye(8) * 100.0
        obj_small = make_object_state()
        obj_small.covariance = np.eye(8) * 1.0
        assert np.all(obj_small.position_std < obj_large.position_std)

    def test_position_uncertainty_area_positive(self):
        obj = make_object_state()
        assert obj.position_uncertainty_area > 0.0

    def test_speed_zero_at_rest(self):
        obj = make_object_state()
        assert obj.speed == pytest.approx(0.0)

    def test_speed_nonzero_when_moving(self):
        obj = make_object_state()
        obj.velocity = np.array([3.0, 4.0, 0.0, 0.0])
        assert obj.speed == pytest.approx(5.0)

    def test_trajectory_empty_initially(self):
        obj = make_object_state()
        assert obj.trajectory.shape == (0, 2)

    def test_trajectory_shape_after_snapshots(self):
        from state_estimation.kalman_filter import KFSnapshot
        obj = make_object_state()
        for i in range(5):
            snap = KFSnapshot(
                timestamp=float(i),
                frame_idx=i,
                state=np.array([100.0 + i, 100.0, 60.0, 40.0,
                                 0.0, 0.0, 0.0, 0.0]),
                covariance=np.eye(8),
                nis=1.0,
                n_updates=i+1,
            )
            obj.add_snapshot(snap)
        traj = obj.trajectory
        assert traj.shape == (5, 2)

    def test_history_bounded_by_max_history(self):
        obj = make_object_state()
        from state_estimation.kalman_filter import KFSnapshot
        for i in range(20):
            snap = KFSnapshot(
                timestamp=float(i), frame_idx=i,
                state=np.zeros(8), covariance=np.eye(8),
                nis=1.0, n_updates=i+1,
            )
            obj.add_snapshot(snap)
        assert len(obj.history) <= obj.max_history

    def test_distance_to(self):
        obj = make_object_state(cx=0.0, cy=0.0)
        assert obj.distance_to(np.array([3.0, 4.0])) == pytest.approx(5.0)

    def test_distance_to_self_is_zero(self):
        obj = make_object_state(cx=100.0, cy=200.0)
        assert obj.distance_to(obj.position) == pytest.approx(0.0)

    def test_mahalanobis_to_self_is_zero(self):
        obj = make_object_state(cx=100.0, cy=100.0)
        assert obj.mahalanobis_to(obj.position) == pytest.approx(0.0)

    def test_repr_contains_track_id(self):
        obj = make_object_state(track_id=42)
        assert "42" in repr(obj)

    def test_repr_contains_class_name(self):
        obj = make_object_state()
        assert "person" in repr(obj)


# ---------------------------------------------------------------------------
# TestSceneGraphUpdate
# ---------------------------------------------------------------------------

class TestSceneGraphUpdate:

    def test_empty_update_no_crash(self, cfg):
        sg = SceneGraph(cfg)
        sg.update([], [], timestamp=time.monotonic())
        assert sg.n_objects == 0

    def test_confirmed_track_added(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg)
        sg.update([track], [], timestamp=time.monotonic())
        assert sg.n_objects == 1

    def test_confirmed_track_retrievable_by_id(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg)
        ts    = time.monotonic()
        sg.update([track], [], timestamp=ts)
        obj = sg.get_state(track.track_id)
        assert obj is not None
        assert obj.track_id == track.track_id

    def test_position_matches_track(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg, cx=200, cy=300)
        sg.update([track], [], timestamp=time.monotonic())
        obj = sg.get_state(track.track_id)
        np.testing.assert_allclose(obj.position, track.position, atol=1e-3)

    def test_confirmed_object_not_lost(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg)
        sg.update([track], [], timestamp=time.monotonic())
        obj = sg.get_state(track.track_id)
        assert not obj.is_lost

    def test_lost_track_marked_lost(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg)
        # First confirm it
        sg.update([track], [], timestamp=time.monotonic())
        # Then mark as lost
        track.state = TrackState.LOST
        sg.update([], [track], timestamp=time.monotonic())
        obj = sg.get_state(track.track_id)
        assert obj.is_lost

    def test_redetected_track_unmarked_lost(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg)

        sg.update([track], [], timestamp=time.monotonic())
        track.state = TrackState.LOST
        sg.update([], [track], timestamp=time.monotonic())
        assert sg.get_state(track.track_id).is_lost

        # Re-confirm
        track.state = TrackState.CONFIRMED
        sg.update([track], [], timestamp=time.monotonic())
        assert not sg.get_state(track.track_id).is_lost

    def test_two_tracks_two_objects(self, cfg):
        sg = SceneGraph(cfg)
        t1 = make_confirmed_track(cfg, cx=100, cy=100)
        t2 = make_confirmed_track(cfg, cx=400, cy=100)
        sg.update([t1, t2], [], timestamp=time.monotonic())
        assert sg.n_objects == 2

    def test_class_name_stored(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg, cls_name="bicycle")
        sg.update([track], [], timestamp=time.monotonic())
        obj = sg.get_state(track.track_id)
        assert obj.class_name == "bicycle"

    def test_history_grows_with_updates(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg)
        for _ in range(5):
            sg.update([track], [], timestamp=time.monotonic())
        obj = sg.get_state(track.track_id)
        assert len(obj.history) == 5

    def test_frame_count_increments(self, cfg):
        sg = SceneGraph(cfg)
        for _ in range(3):
            sg.update([], [], timestamp=time.monotonic())
        assert sg.frame_count == 3

    def test_n_confirmed_counts_only_non_lost(self, cfg):
        sg = SceneGraph(cfg)
        t1 = make_confirmed_track(cfg, cx=100, cy=100)
        t2 = make_confirmed_track(cfg, cx=300, cy=100)
        sg.update([t1, t2], [], timestamp=time.monotonic())
        t2.state = TrackState.LOST
        sg.update([t1], [t2], timestamp=time.monotonic())
        assert sg.n_confirmed == 1
        assert sg.n_lost == 1


# ---------------------------------------------------------------------------
# TestSceneGraphSemanticRefinement
# ---------------------------------------------------------------------------

class TestSceneGraphSemanticRefinement:
    """
    Stability refinement using SemanticMask sampled at the object's
    centroid. Rule under test:
        new = min(current_stability, semantic_class_stability(centroid))
    Only demotes (toward DYNAMIC), never promotes.
    """

    def _mask(self, h, w, fill_class_id, names):
        from perception.semantic_segmenter import SemanticMask
        return SemanticMask(
            mask=np.full((h, w), fill_class_id, dtype=np.int32),
            class_names=names,
            dataset="cityscapes",
            timestamp=0.0, frame_idx=0,
        )

    def test_no_demotion_when_centroid_class_more_static(self, cfg):
        from world_model.stability import StabilityClass
        sg = SceneGraph(cfg)
        # person track (DYNAMIC) over a mask where every pixel is 'road'
        # (STATIC). Rule: min(DYNAMIC, STATIC) = DYNAMIC → no change.
        sm = self._mask(300, 300, 0, {0: "road"})
        track = make_confirmed_track(cfg, cx=150, cy=150, cls_name="person")
        sg.update([track], [], timestamp=time.monotonic(), semantic_mask=sm)
        obj = sg.all_objects()[0]
        assert obj.stability == StabilityClass.DYNAMIC

    def test_demotion_when_centroid_class_more_dynamic(self, cfg):
        from world_model.stability import StabilityClass
        sg = SceneGraph(cfg)
        # chair track (STATIC by COCO prior) whose centroid lands on a
        # 'person' pixel — surface stability = DYNAMIC. Rule:
        # min(STATIC, DYNAMIC) = DYNAMIC → demoted.
        sm = self._mask(300, 300, 11, {11: "person"})
        track = make_confirmed_track(cfg, cx=150, cy=150, cls_name="chair")
        sg.update([track], [], timestamp=time.monotonic(), semantic_mask=sm)
        obj = sg.all_objects()[0]
        assert obj.stability == StabilityClass.DYNAMIC

    def test_partial_demotion_to_semi_static(self, cfg):
        from world_model.stability import StabilityClass
        sg = SceneGraph(cfg)
        # chair (STATIC) on a 'car' (SEMI_STATIC) surface — demote to
        # SEMI_STATIC, not all the way to DYNAMIC.
        sm = self._mask(300, 300, 13, {13: "car"})
        track = make_confirmed_track(cfg, cx=150, cy=150, cls_name="chair")
        sg.update([track], [], timestamp=time.monotonic(), semantic_mask=sm)
        obj = sg.all_objects()[0]
        assert obj.stability == StabilityClass.SEMI_STATIC

    def test_no_promotion(self, cfg):
        from world_model.stability import StabilityClass
        sg = SceneGraph(cfg)
        # person (DYNAMIC) over a 'building' (STATIC) mask. Promotion
        # is disallowed regardless of the surface — should stay DYNAMIC.
        sm = self._mask(300, 300, 2, {2: "building"})
        track = make_confirmed_track(cfg, cx=150, cy=150, cls_name="person")
        sg.update([track], [], timestamp=time.monotonic(), semantic_mask=sm)
        obj = sg.all_objects()[0]
        assert obj.stability == StabilityClass.DYNAMIC

    def test_centroid_out_of_bounds_is_noop(self, cfg):
        from world_model.stability import StabilityClass
        sg = SceneGraph(cfg)
        # 100x100 mask but track sits at (500, 500). class_at returns None.
        sm = self._mask(100, 100, 11, {11: "person"})
        track = make_confirmed_track(cfg, cx=500, cy=500, cls_name="chair")
        sg.update([track], [], timestamp=time.monotonic(), semantic_mask=sm)
        obj = sg.all_objects()[0]
        assert obj.stability == StabilityClass.STATIC  # unchanged

    def test_unknown_centroid_class_id_is_noop(self, cfg):
        from world_model.stability import StabilityClass
        sg = SceneGraph(cfg)
        # mask says class id 99 everywhere; class_names doesn't include 99.
        sm = self._mask(300, 300, 99, {0: "road"})
        track = make_confirmed_track(cfg, cx=150, cy=150, cls_name="chair")
        sg.update([track], [], timestamp=time.monotonic(), semantic_mask=sm)
        obj = sg.all_objects()[0]
        assert obj.stability == StabilityClass.STATIC

    def test_no_mask_passed_is_noop(self, cfg):
        """Default semantic_mask=None must not alter behaviour."""
        from world_model.stability import StabilityClass
        sg = SceneGraph(cfg)
        track = make_confirmed_track(cfg, cx=150, cy=150, cls_name="chair")
        sg.update([track], [], timestamp=time.monotonic())
        obj = sg.all_objects()[0]
        assert obj.stability == StabilityClass.STATIC


# ---------------------------------------------------------------------------
# TestSceneGraphQuery
# ---------------------------------------------------------------------------

class TestSceneGraphQuery:

    def test_query_nearby_returns_empty_when_no_objects(self, cfg):
        sg = SceneGraph(cfg)
        result = sg.query_nearby(np.array([320, 240]), radius=200)
        assert result == []

    def test_query_nearby_finds_close_object(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg, cx=320, cy=240)
        sg.update([track], [], timestamp=time.monotonic())
        result = sg.query_nearby(np.array([320, 240]), radius=10)
        assert len(result) == 1

    def test_query_nearby_excludes_far_object(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg, cx=600, cy=400)
        sg.update([track], [], timestamp=time.monotonic())
        result = sg.query_nearby(np.array([0, 0]), radius=50)
        assert len(result) == 0

    def test_query_nearby_sorted_by_distance(self, cfg):
        sg = SceneGraph(cfg)
        t1 = make_confirmed_track(cfg, cx=100, cy=100)
        t2 = make_confirmed_track(cfg, cx=200, cy=100)
        t3 = make_confirmed_track(cfg, cx=300, cy=100)
        sg.update([t1, t2, t3], [], timestamp=time.monotonic())
        result = sg.query_nearby(np.array([0, 100]), radius=500)
        distances = [d for d, _ in result]
        assert distances == sorted(distances)

    def test_query_nearby_excludes_lost_by_default(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg, cx=100, cy=100)
        sg.update([track], [], timestamp=time.monotonic())
        track.state = TrackState.LOST
        sg.update([], [track], timestamp=time.monotonic())
        result = sg.query_nearby(np.array([100, 100]), radius=200)
        assert len(result) == 0

    def test_query_nearby_includes_lost_when_requested(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg, cx=100, cy=100)
        sg.update([track], [], timestamp=time.monotonic())
        track.state = TrackState.LOST
        sg.update([], [track], timestamp=time.monotonic())
        result = sg.query_nearby(
            np.array([100, 100]), radius=200, include_lost=True
        )
        assert len(result) == 1

    def test_query_by_class_filters_correctly(self, cfg):
        sg = SceneGraph(cfg)
        t1 = make_confirmed_track(cfg, cls_name="person")
        t2 = make_confirmed_track(cfg, cls_name="bicycle")
        sg.update([t1, t2], [], timestamp=time.monotonic())
        people = sg.query_by_class("person")
        assert len(people) == 1
        assert people[0].class_name == "person"

    def test_query_by_class_empty_when_no_match(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg, cls_name="person")
        sg.update([track], [], timestamp=time.monotonic())
        cars = sg.query_by_class("car")
        assert cars == []

    def test_get_state_returns_none_for_unknown_id(self, cfg):
        sg = SceneGraph(cfg)
        assert sg.get_state(999) is None

    def test_all_objects_returns_all(self, cfg):
        sg = SceneGraph(cfg)
        t1 = make_confirmed_track(cfg, cx=100, cy=100)
        t2 = make_confirmed_track(cfg, cx=300, cy=100)
        sg.update([t1, t2], [], timestamp=time.monotonic())
        assert len(sg.all_objects()) == 2

    def test_all_objects_exclude_lost(self, cfg):
        sg = SceneGraph(cfg)
        t1 = make_confirmed_track(cfg, cx=100, cy=100)
        t2 = make_confirmed_track(cfg, cx=300, cy=100)
        sg.update([t1, t2], [], timestamp=time.monotonic())
        t2.state = TrackState.LOST
        sg.update([t1], [t2], timestamp=time.monotonic())
        confirmed = sg.all_objects(include_lost=False)
        assert len(confirmed) == 1
        assert not confirmed[0].is_lost

    def test_most_uncertain_returns_n(self, cfg):
        sg = SceneGraph(cfg)
        tracks = [make_confirmed_track(cfg, cx=i*100, cy=100) for i in range(5)]
        sg.update(tracks, [], timestamp=time.monotonic())
        result = sg.most_uncertain(n=3)
        assert len(result) <= 3

    def test_most_certain_sorted(self, cfg):
        sg = SceneGraph(cfg)
        t1 = make_confirmed_track(cfg, cx=100, cy=100)
        t2 = make_confirmed_track(cfg, cx=300, cy=100)
        sg.update([t1, t2], [], timestamp=time.monotonic())
        # Run several update cycles so covariances converge differently
        for _ in range(5):
            sg.update([t1], [t2], timestamp=time.monotonic())
        result = sg.most_certain(n=2)
        if len(result) >= 2:
            assert (result[0].position_uncertainty_area <=
                    result[1].position_uncertainty_area)


# ---------------------------------------------------------------------------
# TestSceneGraphPrune
# ---------------------------------------------------------------------------

class TestSceneGraphPrune:

    def test_lost_object_pruned_after_timeout(self, cfg):
        cfg = dict(cfg)
        cfg["world_model"] = {"max_history": 10, "lost_timeout_s": 0.05}
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg, cx=100, cy=100)

        sg.update([track], [], timestamp=time.monotonic())
        track.state = TrackState.LOST
        sg.update([], [track], timestamp=time.monotonic())
        assert sg.n_objects == 1

        # Wait for timeout
        import time as time_module
        time_module.sleep(0.1)

        # Trigger prune with a new update
        sg.update([], [], timestamp=time.monotonic())
        assert sg.n_objects == 0, "Stale LOST object must be pruned after timeout"

    def test_confirmed_object_not_pruned(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg)
        sg.update([track], [], timestamp=time.monotonic())
        # Many update cycles — confirmed objects must never be pruned
        for _ in range(10):
            sg.update([track], [], timestamp=time.monotonic())
        assert sg.n_objects == 1

    def test_reset_clears_all(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg)
        sg.update([track], [], timestamp=time.monotonic())
        sg.reset()
        assert sg.n_objects == 0
        assert sg.frame_count == 0


# ---------------------------------------------------------------------------
# TestSceneGraphProps
# ---------------------------------------------------------------------------

class TestSceneGraphProps:

    def test_repr_contains_counts(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg)
        sg.update([track], [], timestamp=time.monotonic())
        r = repr(sg)
        assert "confirmed" in r
        assert "SceneGraph" in r

    def test_summary_contains_object_info(self, cfg):
        sg    = SceneGraph(cfg)
        track = make_confirmed_track(cfg)
        sg.update([track], [], timestamp=time.monotonic())
        s = sg.summary()
        assert "SceneGraph" in s
        assert "person" in s

    def test_object_ids_list(self, cfg):
        sg = SceneGraph(cfg)
        t1 = make_confirmed_track(cfg, cx=100, cy=100)
        t2 = make_confirmed_track(cfg, cx=300, cy=100)
        sg.update([t1, t2], [], timestamp=time.monotonic())
        ids = sg.object_ids
        assert len(ids) == 2
        assert t1.track_id in ids
        assert t2.track_id in ids
