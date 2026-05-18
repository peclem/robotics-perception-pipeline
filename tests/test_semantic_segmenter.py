"""
Unit tests for the semantic segmentation interface + helpers.

These tests cover the pure-Python paths: the dataclass, the Null
backend, drivable-surface masking, and the stability prior lookup.
The actual Mask2Former model inference is gated on transformers +
network availability and is exercised separately (skipped here).

TestSemanticMask          : dataclass invariants + pixel-lookup helper
TestNullSemanticSegmenter : Null backend always returns None
TestDrivableMask          : drivable-surface boolean mask over known classes
TestSemanticStability     : Cityscapes-name → StabilityClass mapping
TestMask2FormerGraceful   : Mask2Former-class instances cope with a
                            missing transformers / model dependency
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.camera_interface import CameraFrame, CameraIntrinsics
from perception.semantic_segmenter import (
    SemanticMask, SemanticSegmenter,
    NullSemanticSegmenter, Mask2FormerSemanticSegmenter,
    drivable_mask, semantic_class_stability,
)
from world_model.stability import StabilityClass


def _intr() -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        width=640, height=480, dist_coeffs=np.zeros(5),
    )


def _frame(h: int = 32, w: int = 48) -> CameraFrame:
    return CameraFrame(
        image=np.zeros((h, w, 3), dtype=np.uint8),
        timestamp=0.0, frame_idx=0, intrinsics=_intr(), source_id="test",
    )


# ---------------------------------------------------------------------------
# SemanticMask
# ---------------------------------------------------------------------------

class TestSemanticMask:

    def test_dataclass_coerces_mask_to_int32(self):
        sm = SemanticMask(
            mask=np.zeros((4, 4), dtype=np.uint8),
            class_names={0: "road"},
            dataset="cityscapes",
            timestamp=0.0, frame_idx=0,
        )
        assert sm.mask.dtype == np.int32
        assert sm.shape == (4, 4)

    def test_class_at_returns_name(self):
        m = np.array([[0, 1], [1, 2]], dtype=np.int32)
        sm = SemanticMask(
            mask=m,
            class_names={0: "road", 1: "sidewalk", 2: "building"},
            dataset="cityscapes",
            timestamp=0.0, frame_idx=0,
        )
        assert sm.class_at(0, 0) == "road"     # (u=col=0, v=row=0)
        assert sm.class_at(1, 1) == "building" # (u=col=1, v=row=1)
        assert sm.class_at(0, 1) == "sidewalk" # (u=col=0, v=row=1)

    def test_class_at_out_of_bounds_returns_none(self):
        sm = SemanticMask(
            mask=np.zeros((4, 4), dtype=np.int32),
            class_names={0: "road"},
            dataset="cityscapes",
            timestamp=0.0, frame_idx=0,
        )
        assert sm.class_at(-1, 0) is None
        assert sm.class_at(0, 100) is None

    def test_unknown_class_id_returns_none(self):
        sm = SemanticMask(
            mask=np.array([[42]], dtype=np.int32),
            class_names={0: "road"},
            dataset="cityscapes",
            timestamp=0.0, frame_idx=0,
        )
        # Pixel says class 42, but no entry in class_names → None.
        assert sm.class_at(0, 0) is None

    def test_3d_mask_rejected(self):
        with pytest.raises(AssertionError):
            SemanticMask(
                mask=np.zeros((4, 4, 3), dtype=np.int32),
                class_names={}, dataset="cityscapes",
                timestamp=0.0, frame_idx=0,
            )


# ---------------------------------------------------------------------------
# Null backend
# ---------------------------------------------------------------------------

class TestNullSemanticSegmenter:

    def test_segment_returns_none(self):
        seg: SemanticSegmenter = NullSemanticSegmenter()
        assert seg.segment(_frame()) is None

    def test_warmup_is_noop(self):
        NullSemanticSegmenter().warmup()  # no exception

    def test_inference_latency_zero(self):
        assert NullSemanticSegmenter().mean_inference_ms == 0.0


# ---------------------------------------------------------------------------
# Drivable-surface mask
# ---------------------------------------------------------------------------

class TestDrivableMask:

    def _cityscapes_sm(self, m: np.ndarray) -> SemanticMask:
        # Cityscapes ids per the standard Mask2Former mapping.
        names = {
            0: "road",       1: "sidewalk",   2: "building",
            8: "vegetation", 9: "terrain",    10: "sky",
            11: "person",    13: "car",
        }
        return SemanticMask(
            mask=m, class_names=names, dataset="cityscapes",
            timestamp=0.0, frame_idx=0,
        )

    def test_drivable_classes_marked_true(self):
        m = np.array([
            [0, 1, 9],       # road | sidewalk | terrain — all drivable
            [2, 11, 13],     # building | person | car   — none drivable
        ], dtype=np.int32)
        out = drivable_mask(self._cityscapes_sm(m))
        assert out.shape == m.shape
        assert out[0, 0] and out[0, 1] and out[0, 2]
        assert not out[1, 0] and not out[1, 1] and not out[1, 2]

    def test_unknown_dataset_returns_all_false(self):
        m = np.zeros((4, 4), dtype=np.int32)
        sm = SemanticMask(
            mask=m, class_names={0: "anything"}, dataset="oxford-robotcar",
            timestamp=0.0, frame_idx=0,
        )
        out = drivable_mask(sm)
        assert out.shape == m.shape
        assert not out.any()

    def test_dataset_with_no_drivable_names_present_returns_all_false(self):
        # Cityscapes label set but stripped of drivable ids.
        sm = SemanticMask(
            mask=np.zeros((4, 4), dtype=np.int32),
            class_names={0: "building"},  # the only present id isn't drivable
            dataset="cityscapes",
            timestamp=0.0, frame_idx=0,
        )
        out = drivable_mask(sm)
        assert not out.any()


# ---------------------------------------------------------------------------
# Stability prior on segmenter classes
# ---------------------------------------------------------------------------

class TestSemanticStability:

    @pytest.mark.parametrize("name,expected", [
        ("road",        StabilityClass.STATIC),
        ("building",    StabilityClass.STATIC),
        ("person",      StabilityClass.DYNAMIC),
        ("car",         StabilityClass.SEMI_STATIC),
        ("motorcycle",  StabilityClass.SEMI_STATIC),
    ])
    def test_cityscapes_classes_map_as_expected(self, name, expected):
        assert semantic_class_stability(name, "cityscapes") == expected

    def test_unknown_class_defaults_to_semi_static(self):
        assert (semantic_class_stability("unicorn", "cityscapes")
                == StabilityClass.SEMI_STATIC)

    def test_unknown_dataset_defaults_to_semi_static(self):
        assert (semantic_class_stability("road", "oxford-robotcar")
                == StabilityClass.SEMI_STATIC)


# ---------------------------------------------------------------------------
# Mask2Former graceful degradation
# ---------------------------------------------------------------------------

class TestMask2FormerGraceful:
    """
    These tests assert the *defensive* path — Mask2Former-class
    instances should construct, warm up, and segment without raising
    even if transformers / weights / GPU is unavailable. Successful
    inference is gated on the optional dep and is covered by separate
    integration testing.
    """

    @pytest.mark.integration
    def test_construct_does_not_raise(self):
        # The constructor calls _load_model under try/except; even
        # without transformers this must not propagate.
        # Marked integration because if transformers IS available, this
        # downloads + loads the real Mask2Former weights (~30 s + ~200 MB).
        seg = Mask2FormerSemanticSegmenter(device="cpu")
        assert seg.mean_inference_ms == 0.0

    def test_segment_returns_none_when_not_ready(self, monkeypatch):
        # Skip the real model download/load so this stays a unit test
        # even when transformers + GPU is available locally.
        monkeypatch.setattr(
            Mask2FormerSemanticSegmenter, "_load_model", lambda self: None,
        )
        seg = Mask2FormerSemanticSegmenter(device="cpu")
        assert not seg.is_ready
        assert seg.segment(_frame()) is None

    def test_warmup_does_nothing_when_not_ready(self, monkeypatch):
        monkeypatch.setattr(
            Mask2FormerSemanticSegmenter, "_load_model", lambda self: None,
        )
        seg = Mask2FormerSemanticSegmenter(device="cpu")
        seg.warmup()  # must not raise
