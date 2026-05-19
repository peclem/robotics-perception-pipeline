"""
Tests for WorldMap re-association behaviour.

TestInsert        : fresh entries, persistent ID assignment
TestSpatialGate   : distance-based filtering
TestAppearanceGate: cosine-similarity-based filtering
TestUpdatePolicy  : EMA blending of embedding on re-association
TestClassGate     : cross-class matches always rejected
TestQuery         : query_nearby ordering
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from world_model.stability import StabilityClass
from world_model.world_map import WorldMap, WorldMapEntry


def _emb(*vals: float) -> np.ndarray:
    v = np.asarray(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

class TestInsert:

    def test_insert_assigns_monotonic_ids(self):
        wm = WorldMap()
        e1 = wm.insert("chair", 56, np.array([0.0, 0.0, 0.0]),
                       _emb(1, 0, 0), StabilityClass.STATIC, 1.0)
        e2 = wm.insert("chair", 56, np.array([10.0, 0.0, 0.0]),
                       _emb(0, 1, 0), StabilityClass.STATIC, 2.0)
        assert e1.persistent_id == 1
        assert e2.persistent_id == 2
        assert wm.n_entries == 2

    def test_insert_or_re_associate_creates_when_empty(self):
        wm = WorldMap()
        entry, was_new = wm.insert_or_re_associate(
            "chair", 56, np.array([0.0, 0.0, 0.0]),
            _emb(1, 0, 0), StabilityClass.STATIC, 1.0,
        )
        assert was_new
        assert entry.persistent_id == 1


# ---------------------------------------------------------------------------
# Spatial gate
# ---------------------------------------------------------------------------

class TestSpatialGate:

    def test_within_gate_matches(self):
        wm = WorldMap(spatial_gate_m=2.0, similarity_threshold=0.0)
        wm.insert("chair", 56, np.array([0.0, 0.0, 0.0]),
                  _emb(1, 0, 0), StabilityClass.STATIC, 1.0)
        match = wm.re_associate("chair", np.array([1.5, 0.0, 0.0]),
                                _emb(1, 0, 0))
        assert match is not None and match.persistent_id == 1

    def test_outside_gate_rejects(self):
        wm = WorldMap(spatial_gate_m=2.0, similarity_threshold=0.0)
        wm.insert("chair", 56, np.array([0.0, 0.0, 0.0]),
                  _emb(1, 0, 0), StabilityClass.STATIC, 1.0)
        match = wm.re_associate("chair", np.array([5.0, 0.0, 0.0]),
                                _emb(1, 0, 0))
        assert match is None

    def test_nearest_wins_within_gate(self):
        # With identical embeddings, spatial proximity decides.
        wm = WorldMap(spatial_gate_m=10.0, similarity_threshold=0.0)
        e1 = wm.insert("chair", 56, np.array([0.0, 0.0, 0.0]),
                       None, StabilityClass.STATIC, 1.0)
        e2 = wm.insert("chair", 56, np.array([3.0, 0.0, 0.0]),
                       None, StabilityClass.STATIC, 1.0)
        # Query at (2.5, 0): closer to e2.
        match = wm.re_associate("chair", np.array([2.5, 0.0, 0.0]), None)
        assert match is e2


# ---------------------------------------------------------------------------
# Appearance gate
# ---------------------------------------------------------------------------

class TestAppearanceGate:

    def test_above_threshold_matches(self):
        wm = WorldMap(spatial_gate_m=10.0, similarity_threshold=0.9)
        wm.insert("chair", 56, np.array([0.0, 0.0, 0.0]),
                  _emb(1, 0, 0, 0), StabilityClass.STATIC, 1.0)
        # Same direction → cosine = 1.0
        match = wm.re_associate("chair", np.array([1.0, 0.0, 0.0]),
                                _emb(1, 0, 0, 0))
        assert match is not None

    def test_below_threshold_rejects(self):
        wm = WorldMap(spatial_gate_m=10.0, similarity_threshold=0.9)
        wm.insert("chair", 56, np.array([0.0, 0.0, 0.0]),
                  _emb(1, 0, 0, 0), StabilityClass.STATIC, 1.0)
        # Nearly orthogonal → cosine ≈ 0 < 0.9
        match = wm.re_associate("chair", np.array([1.0, 0.0, 0.0]),
                                _emb(0, 1, 0, 0))
        assert match is None

    def test_highest_similarity_wins(self):
        wm = WorldMap(spatial_gate_m=10.0, similarity_threshold=0.0)
        e1 = wm.insert("chair", 56, np.array([0.0, 0.0, 0.0]),
                       _emb(1, 0, 0), StabilityClass.STATIC, 1.0)
        e2 = wm.insert("chair", 56, np.array([0.0, 1.0, 0.0]),
                       _emb(0, 1, 0), StabilityClass.STATIC, 1.0)
        # Query similar to e2's embedding → match e2 despite e1 being closer
        match = wm.re_associate("chair", np.array([0.1, 0.2, 0.0]),
                                _emb(0, 1, 0))
        assert match is e2

    def test_spatial_only_when_embeddings_missing(self):
        wm = WorldMap(spatial_gate_m=2.0, similarity_threshold=0.99,
                      allow_spatial_only=True)
        wm.insert("chair", 56, np.array([0.0, 0.0, 0.0]),
                  None, StabilityClass.STATIC, 1.0)
        # Query has no embedding, entry has no embedding → spatial-only.
        match = wm.re_associate("chair", np.array([0.5, 0.0, 0.0]), None)
        assert match is not None

    def test_spatial_only_disabled_when_flag_false(self):
        wm = WorldMap(spatial_gate_m=2.0, similarity_threshold=0.99,
                      allow_spatial_only=False)
        wm.insert("chair", 56, np.array([0.0, 0.0, 0.0]),
                  None, StabilityClass.STATIC, 1.0)
        match = wm.re_associate("chair", np.array([0.5, 0.0, 0.0]), None)
        assert match is None


# ---------------------------------------------------------------------------
# Class gate
# ---------------------------------------------------------------------------

class TestClassGate:

    def test_cross_class_never_matches(self):
        wm = WorldMap(spatial_gate_m=10.0, similarity_threshold=0.0)
        wm.insert("chair", 56, np.array([0.0, 0.0, 0.0]),
                  _emb(1, 0, 0), StabilityClass.STATIC, 1.0)
        # Same position, same embedding, different class → no match.
        match = wm.re_associate("car", np.array([0.0, 0.0, 0.0]),
                                _emb(1, 0, 0))
        assert match is None


# ---------------------------------------------------------------------------
# Update policy (EMA)
# ---------------------------------------------------------------------------

class TestUpdatePolicy:

    def test_position_running_average(self):
        wm = WorldMap()
        entry = wm.insert("chair", 56, np.array([0.0, 0.0, 0.0]),
                          None, StabilityClass.STATIC, 1.0)
        entry.update(np.array([2.0, 0.0, 0.0]), None, 2.0)
        # Running average: (0 + 2) / 2 = 1.0
        np.testing.assert_allclose(entry.position_world, [1.0, 0.0, 0.0])
        assert entry.n_observations == 2

    def test_embedding_ema_stays_normalised(self):
        wm = WorldMap()
        entry = wm.insert("chair", 56, np.array([0.0, 0.0, 0.0]),
                          _emb(1, 0, 0), StabilityClass.STATIC, 1.0)
        entry.update(np.array([0.0, 0.0, 0.0]), _emb(0, 1, 0), 2.0)
        # EMA blend (α=0.1): 0.9*[1,0,0] + 0.1*[0,1,0] then renorm
        np.testing.assert_allclose(
            np.linalg.norm(entry.embedding), 1.0, atol=1e-5,
        )

    def test_update_fills_missing_embedding(self):
        wm = WorldMap()
        entry = wm.insert("chair", 56, np.array([0.0, 0.0, 0.0]),
                          None, StabilityClass.STATIC, 1.0)
        new_emb = _emb(1, 0, 0)
        entry.update(np.array([0.0, 0.0, 0.0]), new_emb, 2.0)
        np.testing.assert_allclose(entry.embedding, new_emb)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class TestQuery:

    def test_query_nearby_sorted_by_distance(self):
        wm = WorldMap()
        wm.insert("chair", 56, np.array([5.0, 0.0, 0.0]),
                  None, StabilityClass.STATIC, 1.0)
        wm.insert("chair", 56, np.array([1.0, 0.0, 0.0]),
                  None, StabilityClass.STATIC, 1.0)
        wm.insert("chair", 56, np.array([3.0, 0.0, 0.0]),
                  None, StabilityClass.STATIC, 1.0)
        results = wm.query_nearby(np.zeros(3), radius=10.0)
        distances = [d for d, _ in results]
        assert distances == sorted(distances)

    def test_query_nearby_respects_radius(self):
        wm = WorldMap()
        wm.insert("chair", 56, np.array([1.0, 0.0, 0.0]),
                  None, StabilityClass.STATIC, 1.0)
        wm.insert("chair", 56, np.array([100.0, 0.0, 0.0]),
                  None, StabilityClass.STATIC, 1.0)
        results = wm.query_nearby(np.zeros(3), radius=5.0)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# End-to-end revisit scenario
# ---------------------------------------------------------------------------

class TestSceneGraphIntegration:
    """End-to-end through SceneGraph — exercises the actual wiring."""

    def _cfg(self):
        return {
            "world_model": {"max_history": 10},
            "tracker": {
                "high_thresh": 0.5, "low_thresh": 0.1,
                "new_track_thresh": 0.5, "iou_threshold": 0.3,
                "max_age": 5, "min_hits": 1, "use_ekf": False,
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
        }

    def _make_track(self, cfg, cls_name, cls_id, cx=200, cy=200):
        from perception.detector import Detection
        from tracking.track import Track, TrackState
        Track.reset_id_counter()
        det = Detection(
            bbox_xyxy=np.array([cx-30, cy-40, cx+30, cy+40], dtype=np.float32),
            confidence=0.9, class_id=cls_id, class_name=cls_name,
            frame_idx=0, timestamp=time.monotonic(),
        )
        tr = Track(det, cfg)
        tr.state = TrackState.CONFIRMED
        return tr

    class _FakeDepth:
        def __init__(self, x, y, z):
            self.position_3d = np.array([x, y, z], dtype=np.float64)

    def test_static_object_re_associated_on_revisit(self):
        """
        End-to-end: a chair (STATIC) is seen, then the live track is
        dropped, then a new live track for the same chair (different
        track_id, same world position + similar embedding) gets the
        same persistent_id from the WorldMap.
        """
        from perception.pose_estimator import CameraPose
        from tracking.track import Track, TrackState
        from world_model.scene_graph import SceneGraph

        wm = WorldMap(spatial_gate_m=1.0, similarity_threshold=0.5)
        sg = SceneGraph(self._cfg(), world_map=wm)
        chair_emb = _emb(1, 0, 0, 0)

        # First sighting at world (5, 0, 3) (in camera frame, identity pose).
        tr1 = self._make_track(self._cfg(), "chair", 56)
        depth = {tr1.track_id: self._FakeDepth(5.0, 0.0, 3.0)}
        emb = {tr1.track_id: chair_emb}
        pose = CameraPose(
            R=np.eye(3), t=np.zeros(3),
            timestamp=0.0, frame_idx=0, source="test",
        )
        sg.update([tr1], [], timestamp=time.monotonic(),
                  depth_estimates=depth, camera_pose=pose,
                  appearance_embeddings=emb)

        obj1 = sg.get_state(tr1.track_id)
        assert obj1.persistent_id is not None, "STATIC track should be mapped"
        assert wm.n_entries == 1
        persistent = obj1.persistent_id

        # Robot leaves. Live track expires; a NEW track for the same
        # chair appears later (different ByteTracker ID, same scene).
        sg.reset()
        Track.reset_id_counter()
        tr2 = self._make_track(self._cfg(), "chair", 56, cx=210, cy=210)
        # Tweak observation a hair to simulate viewpoint drift.
        depth = {tr2.track_id: self._FakeDepth(5.1, 0.05, 3.0)}
        emb = {tr2.track_id: _emb(0.95, 0.05, 0.0, 0.0)}
        sg.update([tr2], [], timestamp=time.monotonic() + 60.0,
                  depth_estimates=depth, camera_pose=pose,
                  appearance_embeddings=emb)

        obj2 = sg.get_state(tr2.track_id)
        assert obj2.persistent_id == persistent, \
            "Re-visited chair must adopt the same persistent_id"
        assert wm.n_entries == 1, "Should not have created a new WorldMap entry"

    def test_dynamic_object_not_added_to_world_map(self):
        from perception.pose_estimator import CameraPose
        from world_model.scene_graph import SceneGraph

        wm = WorldMap()
        sg = SceneGraph(self._cfg(), world_map=wm)
        # person = DYNAMIC by default.
        tr = self._make_track(self._cfg(), "person", 0)
        depth = {tr.track_id: self._FakeDepth(5.0, 0.0, 3.0)}
        pose = CameraPose(
            R=np.eye(3), t=np.zeros(3),
            timestamp=0.0, frame_idx=0, source="test",
        )
        sg.update([tr], [], timestamp=time.monotonic(),
                  depth_estimates=depth, camera_pose=pose,
                  appearance_embeddings={tr.track_id: _emb(1, 0, 0, 0)})

        assert wm.n_entries == 0, \
            "DYNAMIC class must not appear in WorldMap"
        assert sg.get_state(tr.track_id).persistent_id is None


class TestRevisitScenario:

    def test_robot_sees_chair_then_revisits_re_associates(self):
        wm = WorldMap(spatial_gate_m=0.5, similarity_threshold=0.8)
        chair_emb = _emb(1, 0, 0, 0)

        # First encounter — fresh entry.
        e1, new = wm.insert_or_re_associate(
            "chair", 56, np.array([3.0, 4.0, 0.0]),
            chair_emb, StabilityClass.STATIC, 1.0,
        )
        assert new
        pid = e1.persistent_id

        # Robot leaves, time passes, returns to roughly the same spot
        # with a slightly perturbed embedding.
        chair_emb_perturbed = _emb(0.95, 0.05, 0.0, 0.05)
        e2, new = wm.insert_or_re_associate(
            "chair", 56, np.array([3.1, 3.9, 0.0]),
            chair_emb_perturbed, StabilityClass.STATIC, 100.0,
        )
        # Re-associated: same persistent_id, no fresh entry.
        assert not new
        assert e2.persistent_id == pid
        assert wm.n_entries == 1
        # n_observations incremented, last_seen updated.
        assert e2.n_observations == 2
        assert e2.last_seen == 100.0


class TestEviction:
    """Capacity + max-age eviction; off-by-default backwards-compat."""

    def _insert_n(self, wm, n: int, base_x: float = 0.0,
                   t_start: float = 0.0, dt: float = 1.0):
        """Insert n entries at distinct positions + monotonic timestamps."""
        for k in range(n):
            wm.insert(
                "chair", 56,
                np.array([base_x + k * 100.0, 0.0, 0.0]),  # far apart
                None, StabilityClass.STATIC, t_start + k * dt,
            )

    def test_defaults_keep_monotonic_growth(self):
        # No max_age, no max_entries → existing behaviour preserved.
        wm = WorldMap()
        self._insert_n(wm, 50, t_start=1.0, dt=1.0)
        assert wm.n_entries == 50

    def test_max_age_drops_stale_entries(self):
        wm = WorldMap(max_age_s=10.0)
        # First 5 entries at t=1..5, then a fresh entry at t=100.
        # The fresh insert triggers the prune; entries at t<90 go.
        self._insert_n(wm, 5, t_start=1.0, dt=1.0)
        assert wm.n_entries == 5
        wm.insert(
            "chair", 56, np.array([99.0, 0.0, 0.0]),
            None, StabilityClass.STATIC, 100.0,
        )
        # Only the entry inserted at t=100 survives.
        assert wm.n_entries == 1

    def test_max_age_keeps_recent_entries(self):
        wm = WorldMap(max_age_s=10.0)
        # Insert at t=1, then re-insert another at t=5 — both within
        # the 10s window of the prune trigger at t=5.
        self._insert_n(wm, 5, t_start=1.0, dt=1.0)
        # Now insert at t=8 — all five originals are within 10s.
        wm.insert("chair", 56, np.array([99.0, 0.0, 0.0]),
                  None, StabilityClass.STATIC, 8.0)
        # All 6 within the window.
        assert wm.n_entries == 6

    def test_max_entries_lru_evicts_oldest(self):
        wm = WorldMap(max_entries=3)
        # Insert 5 entries at monotonic timestamps; cap is 3, so the
        # two oldest should drop.
        self._insert_n(wm, 5, t_start=1.0, dt=1.0)
        assert wm.n_entries == 3
        # Surviving entries are the three most-recent (last_seen = 3,4,5).
        kept = sorted(e.last_seen for e in wm.all_entries())
        assert kept == [3.0, 4.0, 5.0]

    def test_re_association_refreshes_lru_age(self):
        # Re-associating an old entry should update its last_seen, so
        # it survives a subsequent capacity-eviction.
        wm = WorldMap(max_entries=2, spatial_gate_m=1.0,
                       similarity_threshold=-1.0,  # disable appearance gate
                       allow_spatial_only=True)
        wm.insert("chair", 56, np.array([0.0, 0.0, 0.0]),
                  None, StabilityClass.STATIC, 1.0)   # pid 1, t=1
        wm.insert("chair", 56, np.array([10.0, 0.0, 0.0]),
                  None, StabilityClass.STATIC, 2.0)   # pid 2, t=2
        # Re-associate the FIRST entry at t=3 — refreshes its last_seen.
        e, was_new = wm.insert_or_re_associate(
            "chair", 56, np.array([0.0, 0.0, 0.0]),
            None, StabilityClass.STATIC, 3.0,
        )
        assert not was_new
        assert e.persistent_id == 1
        # Now insert a third entry. With max_entries=2, the LRU drop
        # should target pid 2 (now the oldest by last_seen), NOT pid 1.
        wm.insert("chair", 56, np.array([20.0, 0.0, 0.0]),
                  None, StabilityClass.STATIC, 4.0)
        ids = {e.persistent_id for e in wm.all_entries()}
        assert ids == {1, 3}, f"expected pid 1+3 to survive, got {ids}"

    def test_combined_max_age_and_max_entries(self):
        # Both knobs active: age drops first, capacity caps what's left.
        wm = WorldMap(max_age_s=5.0, max_entries=2)
        self._insert_n(wm, 5, t_start=1.0, dt=1.0)   # 5 entries at t=1..5
        # Insert at t=10 → age cutoff = 5, so t=1..4 are stale (dropped);
        # t=5 + new t=10 → 2 entries, exactly at the cap.
        wm.insert("chair", 56, np.array([99.0, 0.0, 0.0]),
                  None, StabilityClass.STATIC, 10.0)
        assert wm.n_entries == 2
        last_seens = sorted(e.last_seen for e in wm.all_entries())
        assert last_seens == [5.0, 10.0]

    def test_public_prune_works_without_insert(self):
        wm = WorldMap(max_age_s=10.0)
        self._insert_n(wm, 5, t_start=1.0, dt=1.0)
        # No insert — call prune manually at a future "now".
        dropped = wm.prune(now=100.0)
        assert dropped == 5
        assert wm.n_entries == 0

    def test_disabled_max_age_when_zero(self):
        wm = WorldMap(max_age_s=0.0)
        wm.insert("chair", 56, np.array([0.0, 0.0, 0.0]),
                  None, StabilityClass.STATIC, 1.0)
        # Prune at a much later "now" — without aging, nothing drops.
        assert wm.prune(now=1e9) == 0
        assert wm.n_entries == 1

    def test_disabled_max_entries_when_zero(self):
        wm = WorldMap(max_entries=0)
        self._insert_n(wm, 100, t_start=1.0, dt=1.0)
        assert wm.n_entries == 100
