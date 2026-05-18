"""
Unit tests for the ByteTrack multi-object tracker.

Test strategy
-------------
TestIoUBatch         : vectorised IoU computation correctness
TestLinearAssignment : Hungarian algorithm + threshold filtering
TestTrack            : Track lifecycle, state machine, KF integration
TestByteTrackerCore  : single-object tracking, state transitions
TestByteTrackerMOT   : multi-object tracking, ID persistence
TestByteTrackerTwoStage : two-stage association (D_low rescue)
TestByteTrackerEdgeCases : empty frames, single frame, ID monotonicity

All tests use synthetic Detection objects — no GPU, no model required.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from perception.detector import Detection
from perception.camera_interface import CameraFrame, CameraIntrinsics
from tracking.track import Track, TrackState
from tracking.tracker import ByteTracker
from tracking.association import (
    iou_batch,
    iou_distance,
    linear_assignment,
    build_iou_cost_matrix_with_gate,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_track_ids():
    """Reset global Track ID counter before every test."""
    Track.reset_id_counter()
    yield
    Track.reset_id_counter()


@pytest.fixture
def cfg():
    return {
        "tracker": {
            "high_thresh":      0.50,
            "low_thresh":       0.10,
            "new_track_thresh": 0.50,
            "iou_threshold":    0.30,
            "max_age":          3,
            "min_hits":         1,
        },
        "kalman_filter": {
            "initial_covariance": {
                "p_position": 10.0,
                "p_size": 10.0,
                "p_velocity": 100.0,
            },
            "process_noise": {
                "q_position": 1.0,
                "q_size": 1.0,
                "q_velocity": 0.1,
                "q_vel_size": 0.02,
            },
            "measurement_noise": {
                "r_center": 1.0,
                "r_size": 1.0,
            },
        },
    }


@pytest.fixture
def tracker(cfg):
    return ByteTracker(cfg)


_T0 = time.monotonic()


def make_detection(
    x1: float, y1: float, x2: float, y2: float,
    conf: float = 0.9,
    cls_id: int = 0,
    cls_name: str = "person",
    frame_idx: int = 0,
    ts_offset: float = 0.0,
) -> Detection:
    return Detection(
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        confidence=conf,
        class_id=cls_id,
        class_name=cls_name,
        frame_idx=frame_idx,
        timestamp=_T0 + ts_offset,
    )


def make_frame(
    frame_idx: int = 0,
    ts_offset: float = 0.0,
    w: int = 640,
    h: int = 480,
) -> CameraFrame:
    intr = CameraIntrinsics(
        fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=w, height=h
    )
    return CameraFrame(
        image=np.zeros((h, w, 3), dtype=np.uint8),
        timestamp=_T0 + ts_offset,
        frame_idx=frame_idx,
        intrinsics=intr,
        source_id="test",
    )


def box(cx: float, cy: float, w: float = 60.0, h: float = 45.0):
    """Convenience: center+size → xyxy."""
    return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)


# ---------------------------------------------------------------------------
# TestIoUBatch
# ---------------------------------------------------------------------------

class TestIoUBatch:

    def test_identical_boxes_iou_one(self):
        boxes = np.array([[10, 10, 110, 110]], dtype=np.float64)
        iou = iou_batch(boxes, boxes)
        assert iou[0, 0] == pytest.approx(1.0)

    def test_no_overlap_iou_zero(self):
        a = np.array([[0, 0, 50, 50]], dtype=np.float64)
        b = np.array([[100, 100, 200, 200]], dtype=np.float64)
        assert iou_batch(a, b)[0, 0] == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = np.array([[0, 0, 100, 100]], dtype=np.float64)
        b = np.array([[50, 50, 150, 150]], dtype=np.float64)
        iou = iou_batch(a, b)[0, 0]
        # inter=50*50=2500, union=10000+10000-2500=17500
        assert iou == pytest.approx(2500 / 17500, abs=1e-5)

    def test_output_shape(self):
        a = np.random.rand(4, 4) * 100
        a[:, 2:] += 50
        b = np.random.rand(7, 4) * 100
        b[:, 2:] += 50
        assert iou_batch(a, b).shape == (4, 7)

    def test_iou_in_zero_one_range(self):
        a = np.array([[0, 0, 80, 60], [100, 100, 180, 160]], dtype=np.float64)
        b = np.array([[40, 30, 120, 90]], dtype=np.float64)
        iou = iou_batch(a, b)
        assert np.all(iou >= 0.0)
        assert np.all(iou <= 1.0)

    def test_symmetric(self):
        a = np.array([[10, 10, 90, 90]], dtype=np.float64)
        b = np.array([[50, 50, 130, 130]], dtype=np.float64)
        assert iou_batch(a, b)[0, 0] == pytest.approx(iou_batch(b, a)[0, 0])

    def test_empty_a_returns_empty(self):
        a = np.empty((0, 4), dtype=np.float64)
        b = np.array([[0, 0, 100, 100]], dtype=np.float64)
        assert iou_batch(a, b).shape == (0, 1)

    def test_empty_b_returns_empty(self):
        a = np.array([[0, 0, 100, 100]], dtype=np.float64)
        b = np.empty((0, 4), dtype=np.float64)
        assert iou_batch(a, b).shape == (1, 0)

    def test_contained_box(self):
        outer = np.array([[0, 0, 100, 100]], dtype=np.float64)
        inner = np.array([[25, 25, 75, 75]], dtype=np.float64)
        iou = iou_batch(outer, inner)[0, 0]
        # inter=2500, union=10000
        assert iou == pytest.approx(0.25, abs=1e-5)


# ---------------------------------------------------------------------------
# TestLinearAssignment
# ---------------------------------------------------------------------------

class TestLinearAssignment:

    def test_perfect_assignment(self):
        # 2 tracks, 2 detections — perfect diagonal matching
        cost = np.array([[0.0, 1.0],
                          [1.0, 0.0]])
        matches, unmatched_t, unmatched_d = linear_assignment(cost, thresh=0.5)
        assert (0, 0) in matches
        assert (1, 1) in matches
        assert unmatched_t == []
        assert unmatched_d == []

    def test_threshold_rejects_poor_matches(self):
        # Only one good match — cost[0,0]=0.2 accepted, cost[1,1]=0.8 rejected
        cost = np.array([[0.2, 0.9],
                          [0.9, 0.8]])
        matches, unmatched_t, unmatched_d = linear_assignment(cost, thresh=0.7)
        assert len(matches) == 1
        assert (0, 0) in matches
        assert 1 in unmatched_t
        assert 1 in unmatched_d

    def test_all_rejected_above_thresh(self):
        cost = np.ones((3, 3)) * 0.9
        matches, unmatched_t, unmatched_d = linear_assignment(cost, thresh=0.5)
        assert matches == []
        assert len(unmatched_t) == 3
        assert len(unmatched_d) == 3

    def test_more_tracks_than_dets(self):
        cost = np.array([[0.1], [0.1], [0.1]])
        matches, unmatched_t, unmatched_d = linear_assignment(cost, thresh=0.7)
        assert len(matches) == 1
        assert len(unmatched_t) == 2
        assert unmatched_d == []

    def test_more_dets_than_tracks(self):
        cost = np.array([[0.1, 0.1, 0.1]])
        matches, unmatched_t, unmatched_d = linear_assignment(cost, thresh=0.7)
        assert len(matches) == 1
        assert unmatched_t == []
        assert len(unmatched_d) == 2

    def test_empty_cost_matrix(self):
        cost = np.empty((0, 0))
        matches, ut, ud = linear_assignment(cost, thresh=0.7)
        assert matches == [] and ut == [] and ud == []

    def test_no_tracks(self):
        cost = np.empty((0, 3))
        matches, ut, ud = linear_assignment(cost, thresh=0.7)
        assert matches == [] and ut == [] and len(ud) == 3

    def test_no_dets(self):
        cost = np.empty((3, 0))
        matches, ut, ud = linear_assignment(cost, thresh=0.7)
        assert matches == [] and len(ut) == 3 and ud == []

    def test_globally_optimal(self):
        # Greedy would pick (0,0)=0.1 then (1,1)=0.9 → 1 match
        # Optimal picks (0,1)=0.2 and (1,0)=0.2 → 2 matches, total=0.4
        cost = np.array([[0.1, 0.2],
                          [0.2, 0.9]])
        matches, ut, ud = linear_assignment(cost, thresh=0.7)
        assert len(matches) == 2
        assert ut == [] and ud == []


# ---------------------------------------------------------------------------
# TestTrack
# ---------------------------------------------------------------------------

class TestTrack:

    def test_born_as_tentative(self, cfg):
        det = make_detection(*box(200, 200))
        track = Track(det, cfg)
        assert track.state == TrackState.TENTATIVE

    def test_id_increments(self, cfg):
        d = make_detection(*box(100, 100))
        t1 = Track(d, cfg)
        t2 = Track(d, cfg)
        assert t2.track_id == t1.track_id + 1

    def test_id_positive(self, cfg):
        det = make_detection(*box(100, 100))
        track = Track(det, cfg)
        assert track.track_id >= 1

    def test_initial_n_hits_one(self, cfg):
        det = make_detection(*box(100, 100))
        track = Track(det, cfg)
        assert track.n_hits == 1

    def test_initial_n_misses_zero(self, cfg):
        det = make_detection(*box(100, 100))
        track = Track(det, cfg)
        assert track.n_misses == 0

    def test_predict_increments_age(self, cfg):
        det = make_detection(*box(100, 100))
        track = Track(det, cfg)
        track.predict(dt=0.033)
        assert track.age == 2

    def test_update_increments_hits(self, cfg):
        det = make_detection(*box(100, 100))
        track = Track(det, cfg)
        track.predict(dt=0.033)
        track.update(det, timestamp=_T0)
        assert track.n_hits == 2

    def test_update_resets_misses(self, cfg):
        det = make_detection(*box(100, 100))
        track = Track(det, cfg)
        track.n_misses = 5  # simulate some misses
        track.update(det, timestamp=_T0)
        assert track.n_misses == 0

    def test_update_adds_to_history(self, cfg):
        det = make_detection(*box(100, 100))
        track = Track(det, cfg)
        assert len(track.history) == 0
        track.predict(dt=0.033)
        track.update(det, timestamp=_T0)
        assert len(track.history) == 1

    def test_bbox_xyxy_shape(self, cfg):
        det = make_detection(*box(200, 150, 80, 60))
        track = Track(det, cfg)
        assert track.bbox_xyxy.shape == (4,)

    def test_bbox_xyxy_x2_gt_x1(self, cfg):
        det = make_detection(*box(200, 150))
        track = Track(det, cfg)
        x1, y1, x2, y2 = track.bbox_xyxy
        assert x2 > x1
        assert y2 > y1

    def test_position_std_positive(self, cfg):
        det = make_detection(*box(200, 150))
        track = Track(det, cfg)
        std = track.position_std
        assert std.shape == (2,)
        assert np.all(std > 0)

    def test_velocity_zero_at_birth(self, cfg):
        det = make_detection(*box(200, 150))
        track = Track(det, cfg)
        vel = track.velocity
        np.testing.assert_array_almost_equal(vel, [0, 0, 0, 0])

    def test_is_confirmed_false_at_birth(self, cfg):
        det = make_detection(*box(100, 100))
        track = Track(det, cfg)
        assert not track.is_confirmed

    def test_is_tentative_true_at_birth(self, cfg):
        det = make_detection(*box(100, 100))
        track = Track(det, cfg)
        assert track.is_tentative

    def test_class_name_from_detection(self, cfg):
        det = make_detection(*box(100, 100), cls_name="bicycle")
        track = Track(det, cfg)
        assert track.class_name == "bicycle"

    def test_score_from_detection(self, cfg):
        det = make_detection(*box(100, 100), conf=0.77)
        track = Track(det, cfg)
        assert track.score == pytest.approx(0.77)

    def test_score_updates_on_update(self, cfg):
        det1 = make_detection(*box(100, 100), conf=0.80)
        det2 = make_detection(*box(105, 100), conf=0.65)
        track = Track(det1, cfg)
        track.predict(dt=0.033)
        track.update(det2, timestamp=_T0)
        assert track.score == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# TestByteTrackerCore — single-object state transitions
# ---------------------------------------------------------------------------

class TestByteTrackerCore:

    def test_empty_detections_no_tracks(self, tracker):
        frame = make_frame(ts_offset=0.0)
        confirmed = tracker.update([], frame)
        assert confirmed == []

    def test_single_detection_creates_track(self, tracker):
        det = make_detection(*box(200, 200))
        frame = make_frame(ts_offset=0.0)
        confirmed = tracker.update([det], frame)
        assert len(confirmed) == 1

    def test_track_id_is_positive_integer(self, tracker):
        det = make_detection(*box(200, 200))
        frame = make_frame(ts_offset=0.0)
        confirmed = tracker.update([det], frame)
        assert confirmed[0].track_id >= 1
        assert isinstance(confirmed[0].track_id, int)

    def test_track_confirmed_with_min_hits_1(self, tracker):
        """With min_hits=1 (default), a track is confirmed on frame 1."""
        det = make_detection(*box(200, 200))
        frame = make_frame(ts_offset=0.0)
        confirmed = tracker.update([det], frame)
        assert confirmed[0].state == TrackState.CONFIRMED

    def test_track_tentative_with_min_hits_3(self, cfg):
        cfg["tracker"]["min_hits"] = 3
        tracker = ByteTracker(cfg)
        det = make_detection(*box(200, 200))
        frame = make_frame(ts_offset=0.0)
        confirmed = tracker.update([det], frame)
        # Not enough hits — should not appear in confirmed
        assert len(confirmed) == 0

    def test_track_confirmed_after_min_hits(self, cfg):
        cfg["tracker"]["min_hits"] = 2
        tracker = ByteTracker(cfg)
        det = make_detection(*box(200, 200))

        # Frame 1 — TENTATIVE
        f1 = make_frame(frame_idx=0, ts_offset=0.000)
        c1 = tracker.update([det], f1)
        assert len(c1) == 0

        # Frame 2 — should CONFIRM
        f2 = make_frame(frame_idx=1, ts_offset=0.033)
        c2 = tracker.update([det], f2)
        assert len(c2) == 1
        assert c2[0].state == TrackState.CONFIRMED

    def test_id_stable_across_frames(self, tracker):
        det = make_detection(*box(200, 200))

        ids = []
        for i in range(5):
            frame = make_frame(frame_idx=i, ts_offset=i * 0.033)
            confirmed = tracker.update([det], frame)
            if confirmed:
                ids.append(confirmed[0].track_id)

        assert len(set(ids)) == 1, (
            f"Track ID changed across frames: {ids}. "
            "A matched track must keep its ID."
        )

    def test_track_goes_lost_after_missed_frame(self, tracker):
        det = make_detection(*box(200, 200))

        f0 = make_frame(frame_idx=0, ts_offset=0.000)
        tracker.update([det], f0)

        # Frame 1: no detections
        f1 = make_frame(frame_idx=1, ts_offset=0.033)
        confirmed = tracker.update([], f1)

        assert len(confirmed) == 0
        assert len(tracker.lost_tracks) == 1
        assert tracker.lost_tracks[0].state == TrackState.LOST

    def test_lost_track_reconfirmed_when_redetected(self, tracker):
        det = make_detection(*box(200, 200))

        f0 = make_frame(frame_idx=0, ts_offset=0.000)
        tracker.update([det], f0)

        # Miss one frame
        f1 = make_frame(frame_idx=1, ts_offset=0.033)
        tracker.update([], f1)
        assert len(tracker.lost_tracks) == 1

        # Re-detect at same position
        f2 = make_frame(frame_idx=2, ts_offset=0.066)
        confirmed = tracker.update([det], f2)
        assert len(confirmed) == 1
        assert confirmed[0].state == TrackState.CONFIRMED

    def test_reconfirmed_track_keeps_id(self, tracker):
        det = make_detection(*box(200, 200))

        f0 = make_frame(frame_idx=0, ts_offset=0.000)
        c0 = tracker.update([det], f0)
        original_id = c0[0].track_id

        # Miss
        f1 = make_frame(frame_idx=1, ts_offset=0.033)
        tracker.update([], f1)

        # Redetect
        f2 = make_frame(frame_idx=2, ts_offset=0.066)
        c2 = tracker.update([det], f2)
        assert c2[0].track_id == original_id, (
            "Re-confirmed track must keep its original ID."
        )

    def test_track_removed_after_max_age(self, cfg):
        cfg["tracker"]["max_age"] = 2
        tracker = ByteTracker(cfg)
        det = make_detection(*box(200, 200))

        f0 = make_frame(frame_idx=0, ts_offset=0.000)
        tracker.update([det], f0)

        # Miss 3 frames (> max_age=2)
        for i in range(1, 4):
            f = make_frame(frame_idx=i, ts_offset=i * 0.033)
            tracker.update([], f)

        assert len(tracker.tracks) == 0, (
            "Track must be REMOVED after exceeding max_age missed frames."
        )

    def test_n_hits_increments_per_update(self, tracker):
        det = make_detection(*box(200, 200))
        for i in range(4):
            frame = make_frame(frame_idx=i, ts_offset=i * 0.033)
            confirmed = tracker.update([det], frame)

        assert confirmed[0].n_hits == 4

    def test_frame_count_increments(self, tracker):
        for i in range(5):
            frame = make_frame(frame_idx=i, ts_offset=i * 0.033)
            tracker.update([], frame)
        assert tracker.frame_count == 5


# ---------------------------------------------------------------------------
# TestByteTrackerMOT — multi-object tracking
# ---------------------------------------------------------------------------

class TestByteTrackerMOT:

    def test_two_objects_get_distinct_ids(self, tracker):
        det_a = make_detection(*box(100, 200))
        det_b = make_detection(*box(400, 200))

        frame = make_frame(ts_offset=0.0)
        confirmed = tracker.update([det_a, det_b], frame)

        assert len(confirmed) == 2
        ids = {t.track_id for t in confirmed}
        assert len(ids) == 2, "Two objects must get distinct track IDs."

    def test_ids_stable_two_objects(self, tracker):
        det_a = make_detection(*box(100, 200))
        det_b = make_detection(*box(400, 200))

        id_sets = []
        for i in range(5):
            frame = make_frame(frame_idx=i, ts_offset=i * 0.033)
            confirmed = tracker.update([det_a, det_b], frame)
            id_sets.append(frozenset(t.track_id for t in confirmed))

        assert len(set(id_sets)) == 1, (
            "Track IDs must be stable across frames for two stationary objects."
        )

    def test_one_object_disappears(self, tracker):
        det_a = make_detection(*box(100, 200))
        det_b = make_detection(*box(400, 200))

        f0 = make_frame(frame_idx=0, ts_offset=0.000)
        c0 = tracker.update([det_a, det_b], f0)
        id_b = next(t.track_id for t in c0 if t.position[0] > 300)

        # Frame 1: only A detected
        f1 = make_frame(frame_idx=1, ts_offset=0.033)
        c1 = tracker.update([det_a], f1)

        confirmed_ids = {t.track_id for t in c1}
        assert id_b not in confirmed_ids, (
            "Track B must not appear in confirmed after missing one frame."
        )
        assert len(tracker.lost_tracks) == 1

    def test_new_object_gets_new_id(self, tracker):
        det_a = make_detection(*box(100, 200))

        f0 = make_frame(frame_idx=0, ts_offset=0.000)
        c0 = tracker.update([det_a], f0)
        id_a = c0[0].track_id

        # Frame 1: A + new object B
        det_b = make_detection(*box(400, 200))
        f1 = make_frame(frame_idx=1, ts_offset=0.033)
        c1 = tracker.update([det_a, det_b], f1)

        ids = {t.track_id for t in c1}
        assert id_a in ids
        new_id = ids - {id_a}
        assert len(new_id) == 1
        assert list(new_id)[0] > id_a

    def test_assignment_correct_nearby_objects(self, tracker):
        """
        Two objects close together — verifies IoU association correctly
        assigns detections to the nearest track and preserves IDs.
        """
        det_a = make_detection(*box(150, 200, 60, 45))
        det_b = make_detection(*box(250, 200, 60, 45))

        f0 = make_frame(frame_idx=0, ts_offset=0.000)
        c0 = tracker.update([det_a, det_b], f0)
        id_map = {
            round(t.position[0]): t.track_id for t in c0
        }

        # Frame 1: slight movement
        det_a2 = make_detection(*box(153, 200, 60, 45))
        det_b2 = make_detection(*box(253, 200, 60, 45))
        f1 = make_frame(frame_idx=1, ts_offset=0.033)
        c1 = tracker.update([det_a2, det_b2], f1)

        for t in c1:
            cx = round(t.position[0])
            closest_init = min(id_map.keys(), key=lambda k: abs(k - cx))
            assert t.track_id == id_map[closest_init], (
                f"Track near cx={cx} got wrong ID after movement."
            )


# ---------------------------------------------------------------------------
# TestByteTrackerTwoStage — the core ByteTrack innovation
# ---------------------------------------------------------------------------

class TestByteTrackerTwoStage:

    def test_low_conf_det_rescues_lost_track(self, cfg):
        """
        A track that a high-conf detection misses can be rescued by a
        low-conf detection in stage 2. This is the core ByteTrack insight.
        """
        cfg["tracker"]["high_thresh"] = 0.50
        cfg["tracker"]["low_thresh"]  = 0.10
        tracker = ByteTracker(cfg)

        # Frame 0: detect object at high confidence
        det_high = make_detection(*box(200, 200), conf=0.85)
        f0 = make_frame(frame_idx=0, ts_offset=0.000)
        c0 = tracker.update([det_high], f0)
        assert len(c0) == 1
        original_id = c0[0].track_id

        # Frame 1: same object returns LOW confidence (partial occlusion)
        det_low = make_detection(*box(200, 200), conf=0.25)
        f1 = make_frame(frame_idx=1, ts_offset=0.033)
        c1 = tracker.update([det_low], f1)

        # Track should be rescued by stage 2 — still confirmed
        assert len(c1) == 1, (
            "Low-confidence detection must rescue the track in stage 2. "
            "Without two-stage association, the track would go LOST."
        )
        assert c1[0].track_id == original_id

    def test_low_conf_det_does_not_create_new_track(self, cfg):
        """
        Low-confidence detections must NEVER create new tracks.
        Only D_high can create tracks.
        """
        tracker = ByteTracker(cfg)

        # Only low-confidence detection — no existing tracks
        det_low = make_detection(*box(300, 300), conf=0.25)
        frame = make_frame(ts_offset=0.0)
        confirmed = tracker.update([det_low], frame)

        assert len(confirmed) == 0, (
            "A low-confidence detection with no matching track must not "
            "create a new track."
        )
        assert len(tracker.tracks) == 0

    def test_below_low_thresh_ignored_entirely(self, cfg):
        """
        Detections below low_thresh are discarded — they participate in
        neither stage 1 nor stage 2 and cannot rescue or create tracks.
        """
        tracker = ByteTracker(cfg)

        # Below even low_thresh (0.10)
        det_noise = make_detection(*box(200, 200), conf=0.05)
        frame = make_frame(ts_offset=0.0)
        confirmed = tracker.update([det_noise], frame)

        assert len(confirmed) == 0
        assert len(tracker.tracks) == 0

    def test_high_conf_creates_track_low_conf_rescues(self, cfg):
        """
        Verify the full two-stage cycle: object born at high conf,
        partially occluded (low conf), then fully visible again.
        """
        tracker = ByteTracker(cfg)
        original_id = None

        for i, conf in enumerate([0.85, 0.25, 0.90]):
            det = make_detection(*box(200, 200), conf=conf)
            frame = make_frame(frame_idx=i, ts_offset=i * 0.033)
            confirmed = tracker.update([det], frame)

            if i == 0:
                assert len(confirmed) == 1
                original_id = confirmed[0].track_id
            else:
                assert len(confirmed) == 1, f"Track lost at frame {i} conf={conf}"
                assert confirmed[0].track_id == original_id, (
                    "Track ID must be stable through low-confidence frames."
                )

    def test_stage1_uses_all_track_states(self, cfg):
        """
        Stage 1 must try to match D_high against LOST tracks too,
        not just CONFIRMED. A high-confidence detection should re-confirm
        a LOST track.
        """
        cfg["tracker"]["max_age"] = 5
        tracker = ByteTracker(cfg)

        det = make_detection(*box(200, 200), conf=0.85)

        # Establish track
        f0 = make_frame(frame_idx=0, ts_offset=0.000)
        c0 = tracker.update([det], f0)
        original_id = c0[0].track_id

        # Miss one frame → LOST
        f1 = make_frame(frame_idx=1, ts_offset=0.033)
        tracker.update([], f1)
        assert len(tracker.lost_tracks) == 1

        # Re-detect at HIGH confidence → stage 1 should re-confirm
        f2 = make_frame(frame_idx=2, ts_offset=0.066)
        c2 = tracker.update([det], f2)
        assert len(c2) == 1
        assert c2[0].track_id == original_id
        assert c2[0].state == TrackState.CONFIRMED


# ---------------------------------------------------------------------------
# TestByteTrackerEdgeCases
# ---------------------------------------------------------------------------

class TestByteTrackerEdgeCases:

    def test_reset_clears_all_tracks(self, tracker):
        det = make_detection(*box(200, 200))
        frame = make_frame(ts_offset=0.0)
        tracker.update([det], frame)
        assert len(tracker.tracks) > 0

        tracker.reset()
        assert len(tracker.tracks) == 0
        assert tracker.frame_count == 0

    def test_first_frame_no_timestamp_crash(self, tracker):
        """First frame must not crash even with no previous timestamp."""
        det = make_detection(*box(200, 200))
        frame = make_frame(ts_offset=0.0)
        # Must not raise
        confirmed = tracker.update([det], frame)
        assert isinstance(confirmed, list)

    def test_many_detections_no_overlap(self, tracker):
        """5 objects spread far apart — each gets its own track, no ID collision."""
        dets = [make_detection(*box(i * 120, 200)) for i in range(5)]
        frame = make_frame(ts_offset=0.0)
        confirmed = tracker.update(dets, frame)
        assert len(confirmed) == 5
        ids = [t.track_id for t in confirmed]
        assert len(set(ids)) == 5

    def test_track_id_monotonically_increasing(self, tracker):
        """IDs must be assigned in ascending order."""
        ids = []
        for i in range(5):
            det = make_detection(*box(i * 100, 100 + i * 80))
            frame = make_frame(frame_idx=i, ts_offset=i * 0.033)
            confirmed = tracker.update([det], frame)
            ids.extend(t.track_id for t in confirmed)

        for a, b in zip(ids, ids[1:]):
            assert b >= a, f"Track ID decreased: {a} → {b}"

    def test_covariance_present_on_confirmed_track(self, tracker):
        det = make_detection(*box(200, 200))
        frame = make_frame(ts_offset=0.0)
        confirmed = tracker.update([det], frame)
        P = confirmed[0].covariance
        assert P.shape == (8, 8)
        assert not np.any(np.isnan(P))

    def test_position_std_finite(self, tracker):
        det = make_detection(*box(200, 200))
        frame = make_frame(ts_offset=0.0)
        confirmed = tracker.update([det], frame)
        std = confirmed[0].position_std
        assert np.all(np.isfinite(std))
        assert np.all(std > 0)

    def test_duplicate_detections_same_location(self, tracker):
        """
        Two detections at identical positions — one becomes a track,
        the other is unmatched and creates a second track.
        (This is expected behaviour — deduplication is the detector's job.)
        """
        det1 = make_detection(*box(200, 200), conf=0.9)
        det2 = make_detection(*box(201, 200), conf=0.8)  # nearly identical
        frame = make_frame(ts_offset=0.0)
        confirmed = tracker.update([det1, det2], frame)
        # Both should create tracks (IoU between them may be high but still distinct)
        assert len(confirmed) >= 1

    def test_repr_does_not_crash(self, tracker):
        det = make_detection(*box(200, 200))
        frame = make_frame(ts_offset=0.0)
        tracker.update([det], frame)
        r = repr(tracker)
        assert "ByteTracker" in r

    def test_tracker_properties_after_update(self, tracker):
        det = make_detection(*box(200, 200))
        frame = make_frame(ts_offset=0.0)
        tracker.update([det], frame)
        assert tracker.n_confirmed >= 0
        assert isinstance(tracker.confirmed_tracks, list)
        assert isinstance(tracker.lost_tracks, list)


# ---------------------------------------------------------------------------
# Appearance-aware association
# ---------------------------------------------------------------------------

from tracking.association import (
    appearance_distance, build_combined_cost_matrix,
)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


@pytest.fixture
def cfg_appearance(cfg):
    """`cfg` fixture with appearance-aware tracking enabled."""
    cfg = {**cfg, "tracker": {**cfg["tracker"]}}
    cfg["tracker"]["use_appearance"]    = True
    cfg["tracker"]["appearance_weight"] = 0.5
    cfg["tracker"]["appearance_ema"]    = 0.9
    return cfg


class TestAppearanceDistance:
    """Cosine-distance cost matrix between tracks and detection embeddings."""

    class _T:
        def __init__(self, emb): self.embedding = emb

    def test_identical_embeddings_give_zero_distance(self):
        e = _unit([1, 0, 0])
        d = appearance_distance([self._T(e)], [e])
        assert d.shape == (1, 1)
        assert d[0, 0] == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_embeddings_give_one(self):
        a = _unit([1, 0, 0]); b = _unit([0, 1, 0])
        d = appearance_distance([self._T(a)], [b])
        assert d[0, 0] == pytest.approx(1.0, abs=1e-9)

    def test_missing_embedding_yields_default_cost(self):
        a = _unit([1, 0, 0])
        # Track has no embedding → cost 1.0 (matcher ignores via IoU).
        d = appearance_distance([self._T(None)], [a])
        assert d[0, 0] == 1.0
        # Detection has no embedding → cost 1.0.
        d2 = appearance_distance([self._T(a)], [None])
        assert d2[0, 0] == 1.0

    def test_empty_inputs_return_empty_matrix(self):
        assert appearance_distance([], []).shape == (0, 0)
        assert appearance_distance([], [_unit([1, 0])]).shape == (0, 1)
        assert appearance_distance([self._T(_unit([1, 0]))], []).shape == (1, 0)

    def test_shape_mismatch_skipped_not_crashes(self):
        d = appearance_distance(
            [self._T(_unit([1, 0, 0]))], [_unit([1, 0])],  # 3-D vs 2-D
        )
        assert d[0, 0] == 1.0  # silently treated as missing


class TestCombinedCostMatrix:
    """Blended IoU + appearance cost matrix correctness."""

    class _T:
        def __init__(self, bbox, emb=None):
            self.bbox_xyxy = np.array(bbox, dtype=np.float64)
            self.embedding = emb

    class _D:
        def __init__(self, bbox):
            self.bbox_xyxy = np.array(bbox, dtype=np.float64)

    def test_zero_weight_collapses_to_iou(self):
        tracks = [self._T([0, 0, 10, 10], _unit([1, 0]))]
        dets   = [self._D([0, 0, 10, 10])]
        embs   = [_unit([0, 1])]  # orthogonal — would penalise heavily
        c = build_combined_cost_matrix(
            tracks, dets, embs, appearance_weight=0.0,
        )
        # Identical bboxes → IoU=1 → cost=0. Appearance should be ignored.
        assert c[0, 0] == pytest.approx(0.0, abs=1e-9)

    def test_blend_50_50(self):
        tracks = [self._T([0, 0, 10, 10], _unit([1, 0]))]
        dets   = [self._D([0, 0, 10, 10])]
        embs   = [_unit([0, 1])]   # orthogonal → app_cost = 1
        c = build_combined_cost_matrix(
            tracks, dets, embs, appearance_weight=0.5,
        )
        # iou_cost=0, app_cost=1, blend=0.5*0 + 0.5*1 = 0.5
        assert c[0, 0] == pytest.approx(0.5, abs=1e-9)

    def test_appearance_disambiguates_overlapping_bboxes(self):
        """
        Two tracks with very similar bboxes (high IoU both ways) but
        distinct appearances. Detection's appearance should pick the
        right track even though the Hungarian on IoU alone could
        plausibly swap them.
        """
        e_red  = _unit([1.0, 0.0])
        e_blue = _unit([0.0, 1.0])
        tracks = [self._T([0, 0, 10, 10], e_red),
                  self._T([1, 1, 11, 11], e_blue)]  # nearly identical bboxes
        dets   = [self._D([0, 0, 10, 10])]
        # Detection's appearance matches track 0 (red).
        c = build_combined_cost_matrix(
            tracks, dets, [e_red], appearance_weight=0.5,
        )
        # Track 0 should have the lower cost.
        assert c[0, 0] < c[1, 0]

    def test_iou_gate_overrides_appearance(self):
        """
        A track with great appearance match but no IoU overlap must
        still be rejected by the gate (motion consistency wins).
        """
        e = _unit([1, 0, 0])
        tracks = [self._T([0, 0, 10, 10],     e)]
        dets   = [self._D([1000, 1000, 1010, 1010])]   # completely disjoint
        c = build_combined_cost_matrix(
            tracks, dets, [e], appearance_weight=0.9,
            max_iou_distance=0.9,
        )
        # Gated to high sentinel cost > threshold (1.9 in this case).
        assert c[0, 0] > 1.0


class TestTrackEmbeddingEMA:
    """Track.update_embedding smoothing + re-normalisation."""

    def _make(self, cfg):
        det = make_detection(*box(100, 100))
        return Track(det, cfg)

    def test_first_embedding_normalised(self, cfg):
        tr = self._make(cfg)
        tr.update_embedding(np.array([3.0, 0.0, 0.0]))
        assert tr.embedding is not None
        assert np.linalg.norm(tr.embedding) == pytest.approx(1.0, abs=1e-9)

    def test_ema_blend_correct(self, cfg):
        tr = self._make(cfg)
        tr.update_embedding(_unit([1, 0]), alpha=0.0)  # seed
        # alpha=0.5: mix is 0.5*old + 0.5*new before re-normalisation
        new = _unit([0, 1])
        tr.update_embedding(new, alpha=0.5)
        # The new embedding should be (1,1)/sqrt(2) after EMA + renorm.
        assert tr.embedding == pytest.approx(_unit([1, 1]), abs=1e-9)

    def test_ema_keeps_unit_norm(self, cfg):
        tr = self._make(cfg)
        rng = np.random.default_rng(0)
        for _ in range(5):
            tr.update_embedding(_unit(rng.standard_normal(8)), alpha=0.9)
            assert np.linalg.norm(tr.embedding) == pytest.approx(1.0, abs=1e-9)

    def test_none_embedding_is_noop(self, cfg):
        tr = self._make(cfg)
        tr.update_embedding(_unit([1, 0]))
        before = tr.embedding.copy()
        tr.update_embedding(None)
        assert np.allclose(tr.embedding, before)


class TestByteTrackerAppearance:
    """End-to-end ByteTracker behaviour with detection_embeddings wired in."""

    def test_default_use_appearance_off_is_backwards_compatible(self, cfg):
        """
        Passing embeddings while `use_appearance=False` must not change
        association vs the embedding-free baseline.
        """
        cfg_off = {**cfg, "tracker": {**cfg["tracker"], "use_appearance": False}}
        tr_off = ByteTracker(cfg_off)
        det = make_detection(*box(100, 100))
        frame = make_frame(ts_offset=0.0)
        confirmed = tr_off.update(
            [det], frame, detection_embeddings=[_unit([1, 0])],
        )
        assert len(confirmed) == 1

    def test_track_embedding_set_after_first_match(self, cfg_appearance):
        tracker = ByteTracker(cfg_appearance)
        det = make_detection(*box(100, 100))
        frame = make_frame(ts_offset=0.0)
        e = _unit([1.0, 0.0, 0.0])
        tracker.update([det], frame, detection_embeddings=[e])
        # New track now exists, with its embedding seeded.
        tr = tracker.tracks[0]
        assert tr.embedding is not None
        assert np.allclose(tr.embedding, e, atol=1e-9)

    def test_track_embedding_updated_via_ema(self, cfg_appearance):
        tracker = ByteTracker(cfg_appearance)
        det1 = make_detection(*box(100, 100))
        det2 = make_detection(*box(102, 100), ts_offset=1.0/30.0)
        frame1 = make_frame(ts_offset=0.0)
        frame2 = make_frame(ts_offset=1.0/30.0)
        tracker.update([det1], frame1, detection_embeddings=[_unit([1, 0])])
        tracker.update([det2], frame2, detection_embeddings=[_unit([0, 1])])
        tr = tracker.tracks[0]
        # alpha=0.9 → mostly the first embedding, leaning slightly toward second.
        x, y = tr.embedding
        assert x > y > 0.0

    def test_appearance_prevents_id_swap_on_ambiguous_iou(self):
        """
        Two tracks sit close enough that each detection has IoU with
        BOTH tracks. IoU-only Hungarian picks the wrong assignment
        (lower total cost) — appearance with distinct embeddings flips
        the result back. Run side-by-side: same setup, only
        use_appearance toggled.

        Setup:
          Track T1 ('red')  at (100, 200), bbox 60×45
          Track T2 ('blue') at (150, 200), bbox 60×45
          Det D1 ('red')   at (130, 200) — closer to T2 by IoU
          Det D2 ('blue')  at (120, 200) — closer to T1 by IoU
          IoU-only Hungarian (min total cost) → {T1↔D2, T2↔D1}: WRONG
          Appearance → {T1↔D1, T2↔D2}: CORRECT
        """
        def run(use_appearance: bool):
            cfg_local = {
                "tracker": {
                    "high_thresh": 0.50, "low_thresh": 0.10,
                    "new_track_thresh": 0.50, "iou_threshold": 0.05,
                    "max_age": 3, "min_hits": 1,
                    "use_appearance": use_appearance,
                    "appearance_weight": 0.5, "appearance_ema": 0.9,
                },
                "kalman_filter": {
                    "initial_covariance": {"p_position": 10.0, "p_size": 10.0, "p_velocity": 100.0},
                    "process_noise":      {"q_position": 1.0,  "q_size": 1.0,  "q_velocity": 0.1, "q_vel_size": 0.02},
                    "measurement_noise":  {"r_center": 1.0,    "r_size": 1.0},
                },
            }
            tracker = ByteTracker(cfg_local)
            e_red  = _unit([1.0, 0.0])
            e_blue = _unit([0.0, 1.0])
            d_red_0  = make_detection(*box(100, 200), cls_name="red")
            d_blue_0 = make_detection(*box(150, 200), cls_name="blue")
            tracker.update(
                [d_red_0, d_blue_0], make_frame(ts_offset=0.0),
                detection_embeddings=[e_red, e_blue],
            )
            id_of = {t.class_name: t.track_id
                     for t in tracker.confirmed_tracks}
            # Frame 1: positions swapped vs nearest neighbour.
            d_red_1  = make_detection(*box(130, 200), cls_name="red")
            d_blue_1 = make_detection(*box(120, 200), cls_name="blue")
            tracker.update(
                [d_red_1, d_blue_1], make_frame(ts_offset=1.0/30.0),
                detection_embeddings=[e_red, e_blue],
            )
            tracks_by_id = {t.track_id: t for t in tracker.confirmed_tracks}
            return id_of, tracks_by_id

        # Without appearance — Hungarian picks the wrong assignment, so
        # the track that originally tracked the 'red' detection now sits
        # at the 'blue' detection's position (~120).
        Track.reset_id_counter()
        id_of_no, tracks_no = run(use_appearance=False)
        red_no  = tracks_no[id_of_no["red"]]
        assert abs(red_no.position[0] - 120) < 10, (
            f"IoU-only baseline didn't behave as expected: "
            f"red track moved to {red_no.position[0]:.1f}"
        )

        # With appearance — Hungarian matches red→red, blue→blue.
        Track.reset_id_counter()
        id_of_yes, tracks_yes = run(use_appearance=True)
        red_yes  = tracks_yes[id_of_yes["red"]]
        blue_yes = tracks_yes[id_of_yes["blue"]]
        assert abs(red_yes.position[0]  - 130) < 10, (
            f"Appearance should have routed red track to D1 (~130), "
            f"got {red_yes.position[0]:.1f}"
        )
        assert abs(blue_yes.position[0] - 120) < 10
