"""
Unit tests for CameraMotionCompensator.

Hardware-free: synthetic textured images shifted by a known offset, so
the estimated homography has a closed-form expected value. The headline
test pins the *direction* of the warp — the original implementation
applied H⁻¹ and displaced tracks by the negative of the camera motion,
doubling apparent drift instead of removing it. A CMC on/off benchmark
on moving-camera MOT17 caught it (ID-switches ~doubled); this file
guards against the regression.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tracking.motion_compensation import CameraMotionCompensator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _textured_image(w: int = 320, h: int = 240, n_blobs: int = 60, seed: int = 0):
    """A black image scattered with white blobs — distinctive, trackable."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(n_blobs):
        cx, cy = int(rng.integers(15, w - 15)), int(rng.integers(15, h - 15))
        cv2.circle(img, (cx, cy), int(rng.integers(2, 6)), (255, 255, 255), -1)
    return img


def _translate(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


class _StubKF:
    def __init__(self, cx: float, cy: float):
        self._x = np.array([cx, cy, 0, 0, 0, 0, 0, 0], dtype=np.float64)


class _StubTrack:
    """Minimal stand-in: apply() only touches position + kf._x + bbox."""
    def __init__(self, cx: float, cy: float, w: float = 20, h: float = 40):
        self.kf = _StubKF(cx, cy)
        self.bbox_xyxy = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @property
    def position(self):
        return self.kf._x[0], self.kf._x[1]


def _translation_H(dx: float, dy: float) -> np.ndarray:
    H = np.eye(3, dtype=np.float64)
    H[0, 2] = dx
    H[1, 2] = dy
    return H


# ---------------------------------------------------------------------------
# apply() — direction (the regression guard)
# ---------------------------------------------------------------------------

class TestApplyDirection:
    def test_apply_warps_in_camera_motion_direction(self):
        # H maps prev->curr. A track anchored in the previous frame must
        # move BY +(dx, dy) into the current frame — not -(dx, dy).
        cmc = CameraMotionCompensator()
        track = _StubTrack(100.0, 80.0)
        cmc.apply([track], _translation_H(12.0, -7.0))
        np.testing.assert_allclose(track.position, [112.0, 73.0], atol=1e-9)

    def test_apply_is_not_the_inverse(self):
        # Explicitly: applying H must not move the track the wrong way.
        cmc = CameraMotionCompensator()
        track = _StubTrack(50.0, 50.0)
        cmc.apply([track], _translation_H(10.0, 0.0))
        assert track.position[0] == pytest.approx(60.0)   # not 40.0

    def test_identity_homography_leaves_tracks_put(self):
        cmc = CameraMotionCompensator()
        track = _StubTrack(33.0, 44.0)
        cmc.apply([track], np.eye(3))
        np.testing.assert_allclose(track.position, [33.0, 44.0], atol=1e-9)

    def test_apply_handles_rotation_scale(self):
        # 90° rotation about the origin: (x, y) -> (-y, x).
        cmc = CameraMotionCompensator()
        H = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        track = _StubTrack(10.0, 0.0)
        cmc.apply([track], H)
        np.testing.assert_allclose(track.position, [0.0, 10.0], atol=1e-9)

    def test_apply_updates_all_tracks(self):
        cmc = CameraMotionCompensator()
        tracks = [_StubTrack(x, 0.0) for x in (0.0, 100.0, 200.0)]
        cmc.apply(tracks, _translation_H(5.0, 5.0))
        assert [t.position[0] for t in tracks] == [5.0, 105.0, 205.0]


# ---------------------------------------------------------------------------
# estimate() — recovers a known transform
# ---------------------------------------------------------------------------

class TestEstimate:
    def test_recovers_known_translation(self):
        cmc = CameraMotionCompensator()
        prev = _textured_image(seed=1)
        curr = _translate(prev, 9.0, -6.0)
        H = cmc.estimate(prev, curr, tracks=[])
        assert H is not None
        # H maps prev->curr, so its translation column is the shift.
        assert H[0, 2] == pytest.approx(9.0, abs=1.0)
        assert H[1, 2] == pytest.approx(-6.0, abs=1.0)

    def test_static_camera_gives_near_identity(self):
        cmc = CameraMotionCompensator()
        img = _textured_image(seed=2)
        H = cmc.estimate(img, img.copy(), tracks=[])
        assert H is not None
        np.testing.assert_allclose(H, np.eye(3), atol=0.5)

    def test_textureless_frame_returns_none(self):
        cmc = CameraMotionCompensator()
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        assert cmc.estimate(blank, blank.copy(), tracks=[]) is None

    def test_estimate_counts_failure(self):
        cmc = CameraMotionCompensator()
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        cmc.estimate(blank, blank.copy(), tracks=[])
        assert cmc.n_failures == 1
        assert cmc.n_compensations == 0


# ---------------------------------------------------------------------------
# compensate() — estimate + apply end to end
# ---------------------------------------------------------------------------

class TestCompensate:
    def test_compensate_moves_track_with_camera(self):
        # Camera pans so the scene shifts by (+10, +4); a track anchored
        # in the previous frame should follow into the current frame.
        cmc = CameraMotionCompensator()
        prev = _textured_image(seed=3)
        curr = _translate(prev, 10.0, 4.0)
        track = _StubTrack(160.0, 120.0)
        H = cmc.compensate(prev, curr, [track])
        assert H is not None
        assert track.position[0] == pytest.approx(170.0, abs=1.5)
        assert track.position[1] == pytest.approx(124.0, abs=1.5)
        assert cmc.n_compensations == 1

    def test_compensate_returns_none_on_failure(self):
        cmc = CameraMotionCompensator()
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        track = _StubTrack(50.0, 50.0)
        assert cmc.compensate(blank, blank.copy(), [track]) is None
        np.testing.assert_allclose(track.position, [50.0, 50.0])  # untouched


# ---------------------------------------------------------------------------
# background mask
# ---------------------------------------------------------------------------

class TestBackgroundMask:
    def test_track_boxes_are_excluded(self):
        cmc = CameraMotionCompensator()
        track = _StubTrack(160.0, 120.0, w=40, h=60)
        mask = cmc._build_background_mask((240, 320), [track])
        assert mask[120, 160] == 0           # inside the box → excluded
        assert mask[10, 10] == 255           # background → kept

    def test_empty_tracks_keeps_all(self):
        cmc = CameraMotionCompensator()
        mask = cmc._build_background_mask((240, 320), [])
        assert mask.min() == 255


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_compensation_rate(self):
        cmc = CameraMotionCompensator()
        prev = _textured_image(seed=4)
        cmc.compensate(prev, _translate(prev, 6.0, 3.0), [])      # success
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        cmc.compensate(blank, blank.copy(), [])                   # failure
        assert cmc.compensation_rate == pytest.approx(0.5)

    def test_reset_clears_counters(self):
        cmc = CameraMotionCompensator()
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        cmc.estimate(blank, blank.copy(), tracks=[])
        cmc.reset()
        assert cmc.n_failures == 0 and cmc.n_compensations == 0
