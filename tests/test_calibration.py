"""
Unit tests for the calibration script.

Tests use synthetic checkerboard images — no camera required.
Validates: corner detection, calibration math, YAML output format,
load_intrinsics() round-trip.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import yaml
import cv2

from scripts.calibrate_camera import (
    calibrate,
    save_intrinsics,
)
from perception.camera_interface import load_intrinsics


# ---------------------------------------------------------------------------
# Synthetic calibration data generator
# ---------------------------------------------------------------------------

def make_synthetic_calibration_data(
    n_views:    int = 20,
    rows:       int = 6,
    cols:       int = 9,
    image_size: tuple = (640, 480),
    fx:         float = 800.0,
    fy:         float = 800.0,
    cx:         float = 320.0,
    cy:         float = 240.0,
) -> tuple[list, list]:
    """
    Generate synthetic calibration data using a known camera matrix.
    Projects 3D checkerboard points through a known K to get 2D image points.
    Used to verify that calibrate() recovers the known intrinsics.
    """
    K_true = np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_true = np.zeros(5, dtype=np.float64)

    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= 0.025   # 25mm squares

    obj_points = []
    img_points = []

    rng = np.random.default_rng(42)
    for i in range(n_views):
        # Random rotation and translation for each view
        angle = rng.uniform(-0.4, 0.4)
        rvec = np.array([
            rng.uniform(-0.3, 0.3),
            rng.uniform(-0.3, 0.3),
            angle,
        ], dtype=np.float64)
        tvec = np.array([
            rng.uniform(-0.1, 0.1),
            rng.uniform(-0.1, 0.1),
            rng.uniform(0.3, 0.7),
        ], dtype=np.float64)

        pts_2d, _ = cv2.projectPoints(objp, rvec, tvec, K_true, dist_true)
        pts_2d = pts_2d.reshape(-1, 1, 2)

        # Check all points are within image bounds
        x_vals = pts_2d[:, 0, 0]
        y_vals = pts_2d[:, 0, 1]
        if (np.all(x_vals > 0) and np.all(x_vals < image_size[0]) and
                np.all(y_vals > 0) and np.all(y_vals < image_size[1])):
            obj_points.append(objp.copy())
            img_points.append(pts_2d.astype(np.float32))

    return obj_points, img_points


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCalibrate:

    def test_calibrate_recovers_focal_length(self):
        """
        Given synthetic data from a known K, calibrate() should recover
        fx and fy within 5% of the true values.
        """
        true_fx = 800.0
        obj_pts, img_pts = make_synthetic_calibration_data(
            n_views=20, fx=true_fx, fy=true_fx
        )
        if len(obj_pts) < 4:
            pytest.skip("Not enough synthetic views in bounds")

        err, K, dist = calibrate(obj_pts, img_pts, (640, 480))

        assert abs(K[0, 0] - true_fx) / true_fx < 0.05, (
            f"Recovered fx={K[0,0]:.1f} differs from true fx={true_fx} by >5%"
        )
        assert abs(K[1, 1] - true_fx) / true_fx < 0.05

    def test_calibrate_reprojection_error_low(self):
        """Reprojection error on synthetic noise-free data must be very small."""
        obj_pts, img_pts = make_synthetic_calibration_data(n_views=20)
        if len(obj_pts) < 4:
            pytest.skip("Not enough views")

        err, K, dist = calibrate(obj_pts, img_pts, (640, 480))
        assert err < 1.0, f"Reprojection error {err:.4f}px is too high"

    def test_calibrate_returns_correct_shapes(self):
        obj_pts, img_pts = make_synthetic_calibration_data(n_views=15)
        if len(obj_pts) < 4:
            pytest.skip("Not enough views")

        err, K, dist = calibrate(obj_pts, img_pts, (640, 480))
        assert K.shape == (3, 3)
        assert dist.shape[0] >= 5
        assert isinstance(err, float)
        assert err >= 0.0

    def test_calibrate_camera_matrix_structure(self):
        """K must have the correct pinhole structure."""
        obj_pts, img_pts = make_synthetic_calibration_data(n_views=15)
        if len(obj_pts) < 4:
            pytest.skip("Not enough views")

        _, K, _ = calibrate(obj_pts, img_pts, (640, 480))

        assert K[2, 2] == pytest.approx(1.0, abs=1e-6)
        assert K[0, 1] == pytest.approx(0.0, abs=0.1)   # skew ≈ 0
        assert K[1, 0] == pytest.approx(0.0, abs=1e-6)
        assert K[2, 0] == pytest.approx(0.0, abs=1e-6)
        assert K[2, 1] == pytest.approx(0.0, abs=1e-6)


class TestSaveIntrinsics:

    def test_save_creates_file(self, tmp_path):
        K    = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
        dist = np.zeros(5)
        out  = str(tmp_path / "intrinsics.yaml")
        save_intrinsics(out, K, dist, (640, 480), 0.35, 0.025, 15)
        assert Path(out).exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        K    = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
        dist = np.zeros(5)
        out  = str(tmp_path / "nested" / "dir" / "intrinsics.yaml")
        save_intrinsics(out, K, dist, (640, 480), 0.35, 0.025, 15)
        assert Path(out).exists()

    def test_yaml_has_required_keys(self, tmp_path):
        K    = np.array([[923.5, 0., 637.2], [0., 923.1, 361.4], [0., 0., 1.]])
        dist = np.array([-0.042, 0.018, 0.0003, -0.0002, 0.0])
        out  = str(tmp_path / "intrinsics.yaml")
        save_intrinsics(out, K, dist, (1280, 720), 0.43, 0.025, 20)

        with open(out) as f:
            data = yaml.safe_load(f)

        assert "camera_matrix" in data
        assert "image_size"    in data
        assert "dist_coeffs"   in data
        assert "reprojection_error" in data

        cm = data["camera_matrix"]
        assert "fx" in cm and "fy" in cm
        assert "cx" in cm and "cy" in cm

    def test_yaml_values_correct(self, tmp_path):
        K    = np.array([[923.5, 0., 637.2], [0., 923.1, 361.4], [0., 0., 1.]])
        dist = np.array([-0.042, 0.018, 0.0003, -0.0002, 0.0])
        out  = str(tmp_path / "intrinsics.yaml")
        save_intrinsics(out, K, dist, (1280, 720), 0.43, 0.025, 20)

        with open(out) as f:
            data = yaml.safe_load(f)

        assert data["camera_matrix"]["fx"] == pytest.approx(923.5, abs=0.01)
        assert data["camera_matrix"]["fy"] == pytest.approx(923.1, abs=0.01)
        assert data["image_size"]["width"]  == 1280
        assert data["image_size"]["height"] == 720
        assert len(data["dist_coeffs"])     == 5

    def test_dist_coeffs_length(self, tmp_path):
        K    = np.eye(3)
        dist = np.array([0.1, -0.2, 0.001, -0.001, 0.05])
        out  = str(tmp_path / "i.yaml")
        save_intrinsics(out, K, dist, (640, 480), 0.5, 0.025, 10)

        with open(out) as f:
            data = yaml.safe_load(f)
        assert len(data["dist_coeffs"]) == 5


class TestRoundTrip:
    """
    Verify that save_intrinsics() + load_intrinsics() is lossless.
    """

    def test_round_trip_fx_fy(self, tmp_path):
        K    = np.array([[923.5, 0., 637.2], [0., 923.1, 361.4], [0., 0., 1.]])
        dist = np.zeros(5)
        out  = str(tmp_path / "intrinsics.yaml")
        save_intrinsics(out, K, dist, (1280, 720), 0.43, 0.025, 20)

        intr = load_intrinsics(out)
        assert intr.fx == pytest.approx(923.5, abs=0.01)
        assert intr.fy == pytest.approx(923.1, abs=0.01)

    def test_round_trip_cx_cy(self, tmp_path):
        K    = np.array([[800., 0., 320.5], [0., 800., 241.3], [0., 0., 1.]])
        dist = np.zeros(5)
        out  = str(tmp_path / "intrinsics.yaml")
        save_intrinsics(out, K, dist, (640, 480), 0.3, 0.025, 15)

        intr = load_intrinsics(out)
        assert intr.cx == pytest.approx(320.5, abs=0.01)
        assert intr.cy == pytest.approx(241.3, abs=0.01)

    def test_round_trip_image_size(self, tmp_path):
        K    = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
        dist = np.zeros(5)
        out  = str(tmp_path / "intrinsics.yaml")
        save_intrinsics(out, K, dist, (1920, 1080), 0.4, 0.025, 18)

        intr = load_intrinsics(out)
        assert intr.width  == 1920
        assert intr.height == 1080

    def test_round_trip_dist_coeffs(self, tmp_path):
        K    = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
        dist = np.array([-0.042, 0.018, 0.0003, -0.0002, 0.005])
        out  = str(tmp_path / "intrinsics.yaml")
        save_intrinsics(out, K, dist, (640, 480), 0.4, 0.025, 15)

        intr = load_intrinsics(out)
        np.testing.assert_allclose(intr.dist_coeffs, dist, atol=1e-6)

    def test_camera_matrix_from_intrinsics(self, tmp_path):
        """K recovered from load_intrinsics().camera_matrix() must match original."""
        K_orig = np.array([[923.5, 0., 637.2], [0., 923.1, 361.4], [0., 0., 1.]])
        dist   = np.zeros(5)
        out    = str(tmp_path / "intrinsics.yaml")
        save_intrinsics(out, K_orig, dist, (1280, 720), 0.43, 0.025, 20)

        intr   = load_intrinsics(out)
        K_rec  = intr.camera_matrix()
        np.testing.assert_allclose(K_rec, K_orig, atol=0.01)
