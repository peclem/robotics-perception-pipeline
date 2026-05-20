"""
Unit tests for the TUM RGB-D evaluation metrics.

The metric functions in scripts/eval_tum.py are pure — they take
numpy arrays / pose lists and return numbers. These tests pin their
behaviour on inputs with a known closed-form answer (identity, exact
similarity transforms, uniform depth maps), so a regression in the
ATE / RPE / depth math is caught without a real dataset or any model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_tum import (
    ate,
    depth_metrics,
    format_report,
    pose_to_matrix,
    rpe,
    score,
    umeyama_alignment,
)
from perception.pose_estimator import CameraPose


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# A non-degenerate point cloud (spans all three axes) — required for a
# well-posed 3D similarity fit.
_PTS = np.array(
    [[0.0, 0.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 2.0, 0.0],
     [0.0, 0.0, 3.0],
     [1.0, 1.0, 1.0],
     [2.0, -1.0, 0.5]],
    dtype=np.float64,
)


def _rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _pose(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# umeyama_alignment
# ---------------------------------------------------------------------------

class TestUmeyama:
    def test_identity(self):
        s, R, t = umeyama_alignment(_PTS, _PTS)
        assert s == pytest.approx(1.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-9)

    def test_pure_translation(self):
        dst = _PTS + np.array([1.0, -2.0, 3.0])
        s, R, t = umeyama_alignment(_PTS, dst)
        assert s == pytest.approx(1.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(t, [1.0, -2.0, 3.0], atol=1e-9)

    def test_recovers_scale(self):
        dst = 2.5 * _PTS
        s, R, t = umeyama_alignment(_PTS, dst, with_scale=True)
        assert s == pytest.approx(2.5)

    def test_scale_fixed_when_disabled(self):
        dst = 2.5 * _PTS
        s, _, _ = umeyama_alignment(_PTS, dst, with_scale=False)
        assert s == pytest.approx(1.0)

    def test_recovers_full_similarity(self):
        R_true = _rot_z(37.0)
        s_true, t_true = 1.7, np.array([4.0, -1.0, 0.5])
        dst = (s_true * (R_true @ _PTS.T)).T + t_true
        s, R, t = umeyama_alignment(_PTS, dst, with_scale=True)
        assert s == pytest.approx(s_true, rel=1e-6)
        np.testing.assert_allclose(R, R_true, atol=1e-6)
        np.testing.assert_allclose(t, t_true, atol=1e-6)

    def test_result_is_proper_rotation(self):
        dst = (_rot_z(120.0) @ _PTS.T).T
        _, R, _ = umeyama_alignment(_PTS, dst)
        assert np.linalg.det(R) == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            umeyama_alignment(_PTS, _PTS[:3])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            umeyama_alignment(np.empty((0, 3)), np.empty((0, 3)))


# ---------------------------------------------------------------------------
# ate
# ---------------------------------------------------------------------------

class TestATE:
    def test_identical_trajectory_is_zero(self):
        m = ate(_PTS, _PTS)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-9)
        assert m["num"] == len(_PTS)

    def test_invariant_to_similarity_transform(self):
        # A trajectory that is a pure similarity transform of the GT
        # has zero ATE once Sim(3)-aligned.
        est = (3.0 * (_rot_z(50.0) @ _PTS.T)).T + np.array([1.0, 2.0, 3.0])
        m = ate(est, _PTS, with_scale=True)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert m["scale"] == pytest.approx(1.0 / 3.0, rel=1e-6)

    def test_known_residual(self):
        # Shift exactly one point by a known amount; alignment spreads
        # it but RMSE stays positive and bounded by the shift.
        est = _PTS.copy()
        est[0] += np.array([0.0, 0.0, 1.0])
        m = ate(est, _PTS, with_scale=False)
        assert 0.0 < m["rmse"] <= 1.0
        assert m["max"] <= 1.0

    def test_se3_alignment_keeps_scale_error(self):
        est = 2.0 * _PTS
        m = ate(est, _PTS, with_scale=False)
        assert m["rmse"] > 0.0  # scale not removed -> residual remains


# ---------------------------------------------------------------------------
# rpe
# ---------------------------------------------------------------------------

class TestRPE:
    def test_identical_poses_zero(self):
        poses = [_pose(_rot_z(10.0 * i), [i, 0, 0]) for i in range(6)]
        m = rpe(poses, poses)
        assert m["trans_rmse"] == pytest.approx(0.0, abs=1e-9)
        assert m["rot_rmse"] == pytest.approx(0.0, abs=1e-9)
        assert m["num"] == 5

    def test_constant_translation_offset(self):
        gt = [_pose(np.eye(3), [i, 0, 0]) for i in range(6)]
        # Estimate drifts an extra 0.1 m per frame.
        est = [_pose(np.eye(3), [i * 1.1, 0, 0]) for i in range(6)]
        m = rpe(est, gt, delta=1)
        assert m["trans_rmse"] == pytest.approx(0.1, abs=1e-9)
        assert m["rot_rmse"] == pytest.approx(0.0, abs=1e-9)

    def test_rotation_residual_in_degrees(self):
        gt = [_pose(np.eye(3), [0, 0, 0]) for _ in range(4)]
        est = [_pose(_rot_z(5.0 * i), [0, 0, 0]) for i in range(4)]
        m = rpe(est, gt, delta=1)
        assert m["rot_rmse"] == pytest.approx(5.0, abs=1e-6)

    def test_scale_applied_to_translations(self):
        gt = [_pose(np.eye(3), [i, 0, 0]) for i in range(6)]
        est = [_pose(np.eye(3), [i * 0.5, 0, 0]) for i in range(6)]
        # Half-scale estimate, corrected by scale=2.0 -> zero RPE.
        m = rpe(est, gt, delta=1, scale=2.0)
        assert m["trans_rmse"] == pytest.approx(0.0, abs=1e-9)

    def test_delta_gap(self):
        poses = [_pose(np.eye(3), [i, 0, 0]) for i in range(6)]
        assert rpe(poses, poses, delta=3)["num"] == 3

    def test_length_mismatch_raises(self):
        poses = [_pose(np.eye(3), [i, 0, 0]) for i in range(4)]
        with pytest.raises(ValueError):
            rpe(poses, poses[:3])

    def test_delta_too_large_raises(self):
        poses = [_pose(np.eye(3), [i, 0, 0]) for i in range(3)]
        with pytest.raises(ValueError):
            rpe(poses, poses, delta=5)

    def test_bad_delta_raises(self):
        poses = [_pose(np.eye(3), [i, 0, 0]) for i in range(3)]
        with pytest.raises(ValueError):
            rpe(poses, poses, delta=0)


# ---------------------------------------------------------------------------
# depth_metrics
# ---------------------------------------------------------------------------

class TestDepthMetrics:
    def test_perfect_prediction(self):
        gt = np.full((48, 64), 2.0)
        m = depth_metrics(gt.copy(), gt)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["abs_rel"] == pytest.approx(0.0)
        assert m["delta1"] == pytest.approx(1.0)
        assert m["num"] == 48 * 64

    def test_known_constant_error(self):
        gt = np.full((10, 10), 4.0)
        est = np.full((10, 10), 5.0)  # +1 m everywhere
        m = depth_metrics(est, gt)
        assert m["rmse"] == pytest.approx(1.0)
        assert m["abs_rel"] == pytest.approx(0.25)

    def test_nan_pixels_excluded(self):
        gt = np.full((8, 8), 3.0)
        gt[0, 0] = np.nan  # no-return pixel
        est = np.full((8, 8), 3.0)
        est[0, 0] = 999.0  # garbage where GT is invalid — must be ignored
        m = depth_metrics(est, gt)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["num"] == 8 * 8 - 1

    def test_all_invalid_returns_none(self):
        gt = np.full((8, 8), np.nan)
        assert depth_metrics(np.full((8, 8), 2.0), gt) is None

    def test_max_depth_clips_ground_truth(self):
        gt = np.full((8, 8), 3.0)
        gt[0, :] = 50.0  # beyond max_depth -> dropped
        est = np.full((8, 8), 3.0)
        m = depth_metrics(est, gt, max_depth=10.0)
        assert m["num"] == 8 * 8 - 8

    def test_median_alignment_rescales(self):
        gt = np.full((10, 10), 4.0)
        est = np.full((10, 10), 8.0)  # 2x off — relative-depth backend
        m = depth_metrics(est, gt, align="median")
        assert m["rmse"] == pytest.approx(0.0, abs=1e-6)

    def test_resizes_estimate_to_gt(self):
        gt = np.full((48, 64), 2.0)
        est = np.full((24, 32), 2.0)  # half resolution
        m = depth_metrics(est, gt)
        assert m["num"] == 48 * 64
        assert m["rmse"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# pose_to_matrix / score / format_report
# ---------------------------------------------------------------------------

class TestPoseConversion:
    def test_pose_to_matrix(self):
        R = _rot_z(30.0)
        p = CameraPose(R=R, t=np.array([1.0, 2.0, 3.0]), timestamp=0.0, frame_idx=0)
        T = pose_to_matrix(p)
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T[:3, :3], R)
        np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(T[3], [0, 0, 0, 1])


class TestScore:
    def test_score_with_depth_and_trajectory(self):
        poses = [_pose(_rot_z(2.0 * i), [i * 0.1, 0, 0]) for i in range(8)]
        acc = {
            "depth_rows": [
                {"rmse": 0.2, "abs_rel": 0.05, "delta1": 0.9, "num": 100},
                {"rmse": 0.4, "abs_rel": 0.07, "delta1": 0.8, "num": 100},
            ],
            "est_poses": poses,
            "gt_poses":  poses,
            "n_frames":  8,
            "wall_s":    1.234,
        }
        report = score(acc)
        assert report["depth"]["rmse"] == pytest.approx(0.3)
        assert report["depth"]["frames"] == 2
        assert report["trajectory"]["frames"] == 8
        assert report["trajectory"]["ate_sim3"]["rmse"] == pytest.approx(0.0, abs=1e-6)

    def test_score_skips_missing_sections(self):
        acc = {"depth_rows": [], "est_poses": [], "gt_poses": [],
               "n_frames": 0, "wall_s": 0.0}
        report = score(acc)
        assert "depth" not in report
        assert "trajectory" not in report

    def test_score_needs_three_poses_for_trajectory(self):
        poses = [_pose(np.eye(3), [i, 0, 0]) for i in range(2)]
        acc = {"depth_rows": [], "est_poses": poses, "gt_poses": poses,
               "n_frames": 2, "wall_s": 0.0}
        assert "trajectory" not in score(acc)

    def test_format_report_runs(self):
        acc = {"depth_rows": [], "est_poses": [], "gt_poses": [],
               "n_frames": 5, "wall_s": 0.5}
        text = format_report(score(acc), "fr1_room")
        assert "fr1_room" in text
        assert "skipped" in text
