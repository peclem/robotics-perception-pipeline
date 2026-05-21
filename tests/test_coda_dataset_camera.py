"""
Unit tests for CODaDatasetCamera.

A tiny synthetic CODa sequence is written to a tmp directory — no real
dataset download, hardware-free, deterministic. CODa's on-disk format
is text pose/calibration files + JPEG frames, so it is cheap to fake.

What we validate:
  - CameraFrame contract (typed output, BGR uint8, intrinsics present)
  - Timestamp offset: starts at 0.0, monotonic, preserves true dt
  - EOF signal after the sequence is exhausted
  - Ground-truth accessors: pose_gt() 4x4, depth_gt() always None
  - Index-based frame <-> pose association
  - Rectified intrinsics taken from the projection matrix
  - Error handling for missing directories / calibration
"""

import warnings

import cv2
import numpy as np
import pytest
import yaml

from perception.camera_interface import CameraFrame, CameraIntrinsics
from perception.coda_dataset_camera import (
    CODaDatasetCamera,
    coda_intrinsics_from_yaml,
    read_coda_trajectory,
    _detect_sequence_id,
)


# ---------------------------------------------------------------------------
# Synthetic CODa sequence fixture
# ---------------------------------------------------------------------------

def _calib_dict(with_projection: bool = True) -> dict:
    calib = {
        "image_width": 64,
        "image_height": 48,
        "camera_matrix": {
            "rows": 3, "cols": 3,
            "data": [728.73, 0, 30.0, 0, 729.0, 24.0, 0, 0, 1.0],
        },
        "distortion_coefficients": {
            "rows": 1, "cols": 5, "data": [-0.04, 0.14, 0.0, 0.0, -0.08],
        },
    }
    if with_projection:
        calib["projection_matrix"] = {
            "rows": 3, "cols": 4,
            "data": [769.33, 0, 32.0, 0, 0, 769.33, 26.0, 0, 0, 0, 1, 0],
        }
    return calib


def _make_coda_sequence(
    root,
    name: str = "seq_test",
    seq_id: str = "0",
    n: int = 5,
    with_poses: bool = True,
    with_calib: bool = True,
    with_projection: bool = True,
    t0: float = 1673884185.5,
    dt: float = 0.1,
):
    """Write a minimal but format-correct CODa sequence directory."""
    seq = root / name
    frame_dir = seq / "2d_rect" / "cam0" / seq_id
    frame_dir.mkdir(parents=True)
    stamps = [t0 + i * dt for i in range(n)]

    for i in range(n):
        img = np.full((48, 64, 3), (i * 10) % 256, dtype=np.uint8)
        cv2.imwrite(str(frame_dir / f"2d_rect_cam0_{seq_id}_{i}.jpg"), img)

    if with_poses:
        pose_dir = seq / "poses" / "dense_global"
        pose_dir.mkdir(parents=True)
        lines = []
        for i, ts in enumerate(stamps):
            # 'ts x y z qw qx qy qz' — w-first quaternion, identity rotation.
            lines.append(f"{ts:.6f} {float(i)} 0.0 0.0 1.0 0.0 0.0 0.0\n")
        (pose_dir / f"{seq_id}.txt").write_text("".join(lines))

    if with_calib:
        calib_dir = seq / "calibrations" / seq_id
        calib_dir.mkdir(parents=True)
        (calib_dir / "calib_cam0_intrinsics.yaml").write_text(
            yaml.safe_dump(_calib_dict(with_projection))
        )

    return seq


@pytest.fixture
def coda_seq(tmp_path):
    return _make_coda_sequence(tmp_path)


@pytest.fixture
def cam(coda_seq):
    c = CODaDatasetCamera({}, coda_seq)
    c.open()
    yield c
    c.release()


# ---------------------------------------------------------------------------
# CameraFrame contract
# ---------------------------------------------------------------------------

class TestFrameContract:
    def test_returns_camera_frame(self, cam):
        assert isinstance(cam.get_frame(), CameraFrame)

    def test_image_is_bgr_uint8(self, cam):
        frame = cam.get_frame()
        assert frame.image.dtype == np.uint8
        assert frame.image.shape == (48, 64, 3)

    def test_intrinsics_present(self, cam):
        assert isinstance(cam.get_frame().intrinsics, CameraIntrinsics)

    def test_source_id_tags_sequence(self, cam):
        assert cam.get_frame().source_id.startswith("coda:")

    def test_frame_idx_increments(self, cam):
        assert [cam.get_frame().frame_idx for _ in range(3)] == [0, 1, 2]

    def test_context_manager(self, coda_seq):
        with CODaDatasetCamera({}, coda_seq) as c:
            assert c.is_open
            assert c.get_frame() is not None
        assert not c.is_open

    def test_sequence_id_detected(self, cam):
        assert cam.sequence_id == "0"


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

class TestTimestamps:
    def test_first_frame_starts_at_zero(self, cam):
        assert cam.get_frame().timestamp == pytest.approx(0.0)

    def test_monotonic_increasing(self, cam):
        ts = [cam.get_frame().timestamp for _ in range(5)]
        assert all(b > a for a, b in zip(ts, ts[1:]))

    def test_preserves_true_dt(self, cam):
        f0, f1 = cam.get_frame(), cam.get_frame()
        assert f1.timestamp - f0.timestamp == pytest.approx(0.1, abs=1e-4)

    def test_current_timestamp_is_absolute(self, cam):
        cam.get_frame()
        assert cam.current_timestamp == pytest.approx(1673884185.5)


# ---------------------------------------------------------------------------
# EOF
# ---------------------------------------------------------------------------

class TestEOF:
    def test_returns_none_after_exhaustion(self, cam):
        for _ in range(cam.total_frames):
            assert cam.get_frame() is not None
        assert cam.get_frame() is None

    def test_total_frames(self, cam):
        assert cam.total_frames == 5

    def test_max_frames_truncates(self, coda_seq):
        cam = CODaDatasetCamera({}, coda_seq, max_frames=2)
        cam.open()
        assert cam.total_frames == 2
        assert cam.get_frame() is not None and cam.get_frame() is not None
        assert cam.get_frame() is None


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

class TestGroundTruth:
    def test_depth_gt_is_always_none(self, cam):
        cam.get_frame()
        assert cam.depth_gt() is None

    def test_pose_gt_is_4x4(self, cam):
        cam.get_frame()
        pose = cam.pose_gt()
        assert pose.shape == (4, 4)
        np.testing.assert_allclose(pose[3], [0, 0, 0, 1])

    def test_pose_gt_translation_tracks_frame(self, cam):
        cam.get_frame()
        cam.get_frame()  # frame 1 -> tx == 1.0
        np.testing.assert_allclose(cam.pose_gt()[:3, 3], [1.0, 0.0, 0.0])

    def test_pose_gt_none_when_no_pose_file(self, tmp_path):
        seq = _make_coda_sequence(tmp_path, name="seq_np", with_poses=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cam = CODaDatasetCamera({}, seq)
            cam.open()
        cam.get_frame()
        assert cam.pose_gt() is None


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

class TestIntrinsics:
    def test_uses_projection_matrix(self, cam):
        # Rectified frames -> projection matrix, not the raw camera matrix.
        assert cam.intrinsics.fx == pytest.approx(769.33)
        assert cam.intrinsics.cx == pytest.approx(32.0)

    def test_distortion_is_zero(self, cam):
        np.testing.assert_allclose(cam.intrinsics.dist_coeffs, np.zeros(5))

    def test_falls_back_to_camera_matrix(self, tmp_path):
        seq = _make_coda_sequence(tmp_path, name="seq_nc", with_projection=False)
        calib = seq / "calibrations" / "0" / "calib_cam0_intrinsics.yaml"
        intr = coda_intrinsics_from_yaml(calib)
        assert intr.fx == pytest.approx(728.73)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            CODaDatasetCamera({}, "/nonexistent/coda/seq").open()

    def test_missing_2d_rect_raises(self, tmp_path):
        bare = tmp_path / "seq_bare"
        bare.mkdir()
        with pytest.raises(FileNotFoundError):
            CODaDatasetCamera({}, bare).open()

    def test_missing_calibration_raises(self, tmp_path):
        seq = _make_coda_sequence(tmp_path, name="seq_ncal", with_calib=False)
        with pytest.raises(FileNotFoundError):
            CODaDatasetCamera({}, seq).open()

    def test_get_frame_before_open_returns_none(self, coda_seq):
        assert CODaDatasetCamera({}, coda_seq).get_frame() is None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_read_trajectory(self, coda_seq):
        traj = read_coda_trajectory(coda_seq / "poses" / "dense_global" / "0.txt")
        assert len(traj) == 5
        ts, pose = traj[0]
        assert ts == pytest.approx(1673884185.5)
        assert pose.shape == (4, 4)
        np.testing.assert_allclose(pose[:3, :3], np.eye(3))

    def test_trajectory_translation(self, coda_seq):
        traj = read_coda_trajectory(coda_seq / "poses" / "dense_global" / "0.txt")
        np.testing.assert_allclose(traj[3][1][:3, 3], [3.0, 0.0, 0.0])

    def test_detect_sequence_id(self, coda_seq):
        assert _detect_sequence_id(coda_seq) == "0"

    def test_detect_sequence_id_missing_raises(self, tmp_path):
        bare = tmp_path / "seq_empty"
        bare.mkdir()
        with pytest.raises(FileNotFoundError):
            _detect_sequence_id(bare)
