"""
Unit tests for the persistent metric-semantic voxel map (semantic SLAM
mapping layer).

TestConstruction  : parameter validation
TestIntegration   : per-frame folding of depth + labels + pose into voxels
TestLabelFusion   : recursive-Bayesian / vote-accumulation label fusion
TestOccupancy     : occupancy log-odds accumulation + clamping
TestPose          : camera→world transform of integrated voxels
TestQueries       : class_at / query_class / centres / labels / histogram
TestMaintenance   : prune, max-voxel eviction, clear

All tests are hardware-free: synthetic depth maps, synthetic SemanticMasks
and hand-built CameraPoses. No GPU, no model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.camera_interface import CameraIntrinsics
from perception.pose_estimator import CameraPose
from perception.semantic_segmenter import SemanticMask
from world_model.semantic_map import (
    SemanticMap, SemanticMapParams, SemanticVoxel, class_colour,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A small Cityscapes-ish label vocabulary used across the tests.
CLASS_NAMES = {0: "road", 1: "building", 7: "person"}


def make_intrinsics(
    fx: float = 100.0, fy: float = 100.0,
    cx: float = 4.0,  cy: float = 4.0,
    w: int = 8, h: int = 8,
) -> CameraIntrinsics:
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)


def make_pose(t=(0.0, 0.0, 0.0), R=None, ts: float = 0.0, idx: int = 0) -> CameraPose:
    R = np.eye(3, dtype=np.float64) if R is None else np.asarray(R, np.float64)
    return CameraPose(
        R=R, t=np.asarray(t, dtype=np.float64),
        timestamp=ts, frame_idx=idx, source="test",
    )


def make_mask(
    label_array, class_names=None, dataset: str = "cityscapes",
    ts: float = 0.0, idx: int = 0,
) -> SemanticMask:
    return SemanticMask(
        mask=np.asarray(label_array, dtype=np.int32),
        class_names=CLASS_NAMES if class_names is None else class_names,
        dataset=dataset, timestamp=ts, frame_idx=idx,
    )


def single_pixel(depth: float = 2.0, label: int = 0):
    """
    A 1×1 frame whose only pixel sits on the principal point — so its
    camera-frame ray is exactly [0, 0, depth], independent of focal length.
    Returns (depth_map, intrinsics).
    """
    intr = make_intrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0, w=1, h=1)
    depth_map = np.array([[depth]], dtype=np.float32)
    return depth_map, intr


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_default_params(self):
        smap = SemanticMap()
        assert len(smap) == 0
        assert smap.frames_integrated == 0
        assert smap.params.voxel_size_m > 0.0

    def test_rejects_nonpositive_voxel_size(self):
        with pytest.raises(ValueError):
            SemanticMap(SemanticMapParams(voxel_size_m=0.0))

    def test_rejects_inverted_range(self):
        with pytest.raises(ValueError):
            SemanticMap(SemanticMapParams(min_range_m=5.0, max_range_m=1.0))

    def test_rejects_bad_stride(self):
        with pytest.raises(ValueError):
            SemanticMap(SemanticMapParams(pixel_stride=0))


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_integrate_creates_voxels(self):
        smap = SemanticMap(SemanticMapParams(voxel_size_m=0.10, pixel_stride=1))
        depth = np.full((8, 8), 2.0, dtype=np.float32)
        mask = make_mask(np.zeros((8, 8), dtype=np.int32))
        touched = smap.integrate(depth, mask, make_pose(), make_intrinsics())
        assert touched > 0
        assert len(smap) == touched
        assert smap.frames_integrated == 1

    def test_shape_mismatch_is_skipped(self):
        smap = SemanticMap()
        depth = np.full((4, 4), 2.0, dtype=np.float32)
        mask = make_mask(np.zeros((8, 8), dtype=np.int32))
        assert smap.integrate(depth, mask, make_pose(), make_intrinsics()) == 0
        assert len(smap) == 0
        assert smap.frames_integrated == 0

    def test_out_of_range_pixels_skipped(self):
        smap = SemanticMap(SemanticMapParams(min_range_m=0.3, max_range_m=8.0))
        # 0.1 m is below the near clip, 50 m above the far clip.
        for bad_depth in (0.1, 50.0):
            depth, intr = single_pixel(depth=bad_depth)
            mask = make_mask([[0]])
            assert smap.integrate(depth, mask, make_pose(), intr) == 0
        assert len(smap) == 0

    def test_non_finite_depth_skipped(self):
        smap = SemanticMap()
        depth = np.array([[np.nan]], dtype=np.float32)
        _, intr = single_pixel()
        mask = make_mask([[0]])
        assert smap.integrate(depth, mask, make_pose(), intr) == 0

    def test_vocabulary_captured_from_first_mask(self):
        smap = SemanticMap()
        depth, intr = single_pixel()
        smap.integrate(depth, make_mask([[0]]), make_pose(), intr)
        assert smap.dataset == "cityscapes"
        assert smap.class_names == CLASS_NAMES


# ---------------------------------------------------------------------------
# Label fusion
# ---------------------------------------------------------------------------

class TestLabelFusion:

    def _integrate_label(self, smap, label, ts=0.0):
        depth, intr = single_pixel(depth=2.0)
        smap.integrate(depth, make_mask([[label]], ts=ts), make_pose(), intr)

    def test_consistent_label_full_confidence(self):
        smap = SemanticMap(SemanticMapParams(voxel_size_m=0.10))
        for _ in range(3):
            self._integrate_label(smap, label=7)
        voxel = smap.voxel_at(0.0, 0.0, 2.0)
        assert voxel is not None
        assert voxel.hits == 3
        assert voxel.dominant_class_id == 7
        assert voxel.confidence == pytest.approx(1.0)

    def test_majority_label_wins(self):
        smap = SemanticMap(SemanticMapParams(voxel_size_m=0.10))
        self._integrate_label(smap, label=0)   # road, 1 vote
        self._integrate_label(smap, label=1)   # building, 1 vote
        self._integrate_label(smap, label=1)   # building, 2 votes
        voxel = smap.voxel_at(0.0, 0.0, 2.0)
        assert voxel.dominant_class_id == 1
        assert voxel.confidence == pytest.approx(2.0 / 3.0)

    def test_observation_weight_scales_scores(self):
        smap = SemanticMap(SemanticMapParams(
            voxel_size_m=0.10, observation_weight=2.5,
        ))
        self._integrate_label(smap, label=0)
        voxel = smap.voxel_at(0.0, 0.0, 2.0)
        assert voxel.class_scores[0] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Occupancy log-odds
# ---------------------------------------------------------------------------

class TestOccupancy:

    def test_single_observation_is_occupied(self):
        smap = SemanticMap(SemanticMapParams(occupancy_hit_logodds=0.85))
        depth, intr = single_pixel()
        smap.integrate(depth, make_mask([[0]]), make_pose(), intr)
        voxel = smap.voxel_at(0.0, 0.0, 2.0)
        assert voxel.log_odds == pytest.approx(0.85)
        assert voxel.is_occupied

    def test_log_odds_clamped(self):
        smap = SemanticMap(SemanticMapParams(
            occupancy_hit_logodds=0.85, occupancy_clamp=4.0,
        ))
        depth, intr = single_pixel()
        for _ in range(20):
            smap.integrate(depth, make_mask([[0]]), make_pose(), intr)
        voxel = smap.voxel_at(0.0, 0.0, 2.0)
        assert voxel.log_odds == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Pose transform
# ---------------------------------------------------------------------------

class TestPose:

    def test_identity_pose_places_voxel_on_optical_axis(self):
        smap = SemanticMap(SemanticMapParams(voxel_size_m=0.10))
        depth, intr = single_pixel(depth=2.0)
        smap.integrate(depth, make_mask([[0]]), make_pose(), intr)
        # Camera ray [0,0,2] under an identity pose → world [0,0,2].
        assert smap.voxel_at(0.0, 0.0, 2.0) is not None

    def test_translation_shifts_voxels(self):
        smap = SemanticMap(SemanticMapParams(voxel_size_m=0.10))
        depth, intr = single_pixel(depth=2.0)
        # Camera translated +5 m in world X → voxel at world [5,0,2].
        smap.integrate(
            depth, make_mask([[0]]), make_pose(t=(5.0, 0.0, 0.0)), intr,
        )
        assert smap.voxel_at(5.0, 0.0, 2.0) is not None
        assert smap.voxel_at(0.0, 0.0, 2.0) is None

    def test_rotation_applied(self):
        smap = SemanticMap(SemanticMapParams(voxel_size_m=0.10))
        depth, intr = single_pixel(depth=2.0)
        # 90° yaw about world Z maps camera +Z onto world... R @ [0,0,2].
        R = np.array([[0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0],
                      [-1.0, 0.0, 0.0]], dtype=np.float64)
        smap.integrate(depth, make_mask([[0]]), make_pose(R=R), intr)
        pw = R @ np.array([0.0, 0.0, 2.0])
        assert smap.voxel_at(*pw) is not None


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

class TestQueries:

    def _two_class_map(self):
        """Voxel A (road) near origin, voxel B (building) 5 m away in X."""
        smap = SemanticMap(SemanticMapParams(voxel_size_m=0.10))
        depth, intr = single_pixel(depth=2.0)
        smap.integrate(depth, make_mask([[0]]), make_pose(), intr)
        smap.integrate(
            depth, make_mask([[1]]), make_pose(t=(5.0, 0.0, 0.0)), intr,
        )
        return smap

    def test_class_at_resolves_names(self):
        smap = self._two_class_map()
        assert smap.class_at(0.0, 0.0, 2.0) == "road"
        assert smap.class_at(5.0, 0.0, 2.0) == "building"

    def test_class_at_unmapped_is_none(self):
        smap = self._two_class_map()
        assert smap.class_at(100.0, 100.0, 100.0) is None

    def test_centres_labels_confidence_aligned(self):
        smap = self._two_class_map()
        centres = smap.voxel_centres_world()
        labels = smap.voxel_labels()
        conf = smap.voxel_confidence()
        assert centres.shape == (2, 3)
        assert labels.shape == (2,)
        assert conf.shape == (2,)
        assert set(labels.tolist()) == {0, 1}
        # Each centre is within one voxel of its integrated world point.
        for centre, label in zip(centres, labels):
            expected = np.array([0.0, 0.0, 2.0]) if label == 0 \
                else np.array([5.0, 0.0, 2.0])
            assert np.all(np.abs(centre - expected) <= 0.10)

    def test_query_class_returns_matching_voxels(self):
        smap = self._two_class_map()
        road = smap.query_class("road")
        assert road.shape == (1, 3)
        assert np.all(np.abs(road[0] - np.array([0.0, 0.0, 2.0])) <= 0.10)
        assert smap.query_class("person").shape == (0, 3)

    def test_class_histogram(self):
        smap = self._two_class_map()
        assert smap.class_histogram() == {"road": 1, "building": 1}

    def test_empty_map_queries(self):
        smap = SemanticMap()
        assert smap.voxel_centres_world().shape == (0, 3)
        assert smap.voxel_labels().shape == (0,)
        assert smap.voxel_confidence().shape == (0,)
        assert smap.query_class("road").shape == (0, 3)
        assert smap.class_histogram() == {}


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

class TestMaintenance:

    def test_prune_drops_stale_voxels(self):
        smap = SemanticMap()
        depth, intr = single_pixel()
        smap.integrate(depth, make_mask([[0]], ts=1.0), make_pose(), intr)
        # Observed at t=1.0; at t=10.0 with a 5 s horizon it is stale.
        removed = smap.prune(now=10.0, max_age_s=5.0)
        assert removed == 1
        assert len(smap) == 0

    def test_prune_zero_age_is_noop(self):
        smap = SemanticMap()
        depth, intr = single_pixel()
        smap.integrate(depth, make_mask([[0]], ts=1.0), make_pose(), intr)
        assert smap.prune(now=10_000.0, max_age_s=0.0) == 0
        assert len(smap) == 1

    def test_max_voxels_evicts_oldest(self):
        smap = SemanticMap(SemanticMapParams(voxel_size_m=0.10, max_voxels=1))
        depth, intr = single_pixel(depth=2.0)
        # Frame 1 (older) at the origin.
        smap.integrate(depth, make_mask([[0]], ts=1.0), make_pose(), intr)
        # Frame 2 (newer) 100 m away → a distinct voxel; cap forces eviction.
        smap.integrate(
            depth, make_mask([[0]], ts=2.0),
            make_pose(t=(100.0, 0.0, 0.0), ts=2.0), intr,
        )
        assert len(smap) == 1
        assert smap.voxel_at(0.0, 0.0, 2.0) is None        # oldest evicted
        assert smap.voxel_at(100.0, 0.0, 2.0) is not None  # newest kept

    def test_clear_empties_map(self):
        smap = SemanticMap()
        depth, intr = single_pixel()
        smap.integrate(depth, make_mask([[0]]), make_pose(), intr)
        smap.clear()
        assert len(smap) == 0


# ---------------------------------------------------------------------------
# SemanticVoxel unit
# ---------------------------------------------------------------------------

class TestSemanticVoxel:

    def test_observe_accumulates(self):
        v = SemanticVoxel()
        v.observe(class_id=3, weight=1.0, hit_logodds=0.85,
                  clamp=4.0, timestamp=1.0)
        v.observe(class_id=3, weight=1.0, hit_logodds=0.85,
                  clamp=4.0, timestamp=2.0)
        assert v.hits == 2
        assert v.last_seen == 2.0
        assert v.dominant_class_id == 3
        assert v.log_odds == pytest.approx(1.70)

    def test_empty_voxel_confidence_zero(self):
        assert SemanticVoxel().confidence == 0.0


# ---------------------------------------------------------------------------
# class_colour viz helper
# ---------------------------------------------------------------------------

class TestClassColour:

    def test_returns_rgb_triple_in_range(self):
        for cid in range(0, 40):
            c = class_colour(cid)
            assert len(c) == 3
            assert all(isinstance(v, int) for v in c)
            assert all(0 <= v <= 255 for v in c)

    def test_deterministic(self):
        assert class_colour(7) == class_colour(7)

    def test_distinct_ids_mostly_distinct_colours(self):
        # The id→hue map need not be a perfect injection, but a small
        # vocabulary should land on a healthy spread of colours.
        colours = {class_colour(cid) for cid in range(19)}
        assert len(colours) >= 15
