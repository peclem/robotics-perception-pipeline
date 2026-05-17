"""
Tests for the DPVO PoseEstimator wrapper.

TestQuaternionToRotation : the local SE(3) helper
TestDPVOWrapperLogic     : stride, lazy init, bootstrap handling — uses
                           a fake DPVO module to avoid GPU dependency
TestDPVOLive             : end-to-end against the real DPVO + GPU.
                           Marked integration; excluded from CI.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from perception.camera_interface import CameraFrame, CameraIntrinsics


# ---------------------------------------------------------------------------
# Quaternion helper
# ---------------------------------------------------------------------------

class TestQuaternionToRotation:

    def _import_helper(self):
        # Import here so the module's heavy side-effects don't fire at
        # collection time on machines without DPVO present.
        from perception.dpvo_pose_estimator import _quat_to_rotation
        return _quat_to_rotation

    def test_identity_quaternion(self):
        f = self._import_helper()
        R = f(0.0, 0.0, 0.0, 1.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_90deg_about_z(self):
        f = self._import_helper()
        # qz = sin(45°), qw = cos(45°)
        s = np.sin(np.pi/4); c = np.cos(np.pi/4)
        R = f(0.0, 0.0, s, c)
        expected = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_orthonormal(self):
        f = self._import_helper()
        R = f(0.3, 0.5, -0.2, 0.8)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Wrapper logic — fake DPVO, no GPU
# ---------------------------------------------------------------------------

class _FakeSE3:
    """SE3 stand-in that returns a fixed pose vector when .data accessed."""
    def __init__(self, vec: np.ndarray):
        self._vec = vec

    def inv(self):
        return self

    @property
    def data(self):
        class _T:
            def __init__(s, v): s._v = v
            def cpu(s): return s
            def numpy(s): return s._v
        return _T(self._vec)


class _FakePoseGraph:
    """Stand-in for slam.pg with a poses_ buffer addressable by integer index."""
    def __init__(self):
        # Store as a dict so any index works during tests.
        self._poses = {}

    def __setitem__(self, idx, vec):
        self._poses[idx] = vec

    @property
    def poses_(self):
        return self


class _FakeTensor:
    """Stub for slam.pg.poses_[i] that quacks like a 1-D torch tensor."""
    def __init__(self, vec): self._vec = vec
    def unsqueeze(self, dim): return self  # only used to make SE3 accept it


class _FakeDPVO:
    """Minimal DPVO stand-in. Counts calls, fakes is_initialized and pg.poses_."""
    last_instance = None

    def __init__(self, cfg, network, ht, wd, viz=False):
        self.cfg = cfg
        self.ht = ht
        self.wd = wd
        self.calls = []
        self.is_initialized = False
        self.n = 0
        self.pg = _FakePoseGraphTensor()  # see below
        _FakeDPVO.last_instance = self

    def __call__(self, t, image, intrinsics):
        self.calls.append(t)
        if len(self.calls) >= 8:
            self.is_initialized = True
            self.n = len(self.calls) - 7   # arbitrary: grows after init


class _FakePoseGraphTensor:
    """slam.pg with .poses_ that supports .poses_[i].unsqueeze(0)."""
    class _Indexable:
        def __getitem__(self, idx):
            # Return identity-ish pose vec (7,) wrapped to satisfy unsqueeze().
            v = np.array([float(idx)*0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            return _FakeTensor(v)

    def __init__(self):
        self.poses_ = self._Indexable()


class _FakeSE3WithInv:
    """SE3 stand-in: stores a vec, .inv() returns self, .data → numpy bridge."""
    def __init__(self, t):
        # t is _FakeTensor; just unwrap.
        self._vec = t._vec if isinstance(t, _FakeTensor) else np.asarray(t)

    def inv(self):
        return self

    @property
    def data(self):
        class _T:
            def __init__(s, v): s._v = v
            def cpu(s): return s
            def numpy(s): return s._v.reshape(1, -1)
        return _T(self._vec)


@pytest.fixture
def fake_dpvo_module(tmp_path, monkeypatch):
    """
    Install fake `dpvo.config`, `dpvo.dpvo`, and `dpvo.lietorch` into
    sys.modules so the wrapper imports without DPVO actually present.
    """
    fake_config = types.ModuleType("dpvo.config")
    class _Cfg:
        PATCHES_PER_FRAME = 96
        def clone(self): return self
    fake_config.cfg = _Cfg()

    fake_dpvo_mod = types.ModuleType("dpvo.dpvo")
    fake_dpvo_mod.DPVO = _FakeDPVO

    fake_lietorch = types.ModuleType("dpvo.lietorch")
    fake_lietorch.SE3 = _FakeSE3WithInv

    fake_root = types.ModuleType("dpvo")

    monkeypatch.setitem(sys.modules, "dpvo", fake_root)
    monkeypatch.setitem(sys.modules, "dpvo.config", fake_config)
    monkeypatch.setitem(sys.modules, "dpvo.dpvo", fake_dpvo_mod)
    monkeypatch.setitem(sys.modules, "dpvo.lietorch", fake_lietorch)

    # Fake checkpoint file so the constructor's existence check passes.
    ckpt = tmp_path / "dpvo.pth"
    ckpt.write_bytes(b"\x00")
    yield ckpt


@pytest.fixture
def fake_torch(monkeypatch):
    """Fake torch.from_numpy + .permute().cuda() chain used by estimate()."""
    fake_torch_mod = types.ModuleType("torch")

    class _T:
        def __init__(self, arr): self.arr = arr
        def permute(self, *args): return self
        def cuda(self): return self

    class _NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    fake_torch_mod.from_numpy = lambda a: _T(a)
    fake_torch_mod.no_grad = _NoGrad
    monkeypatch.setitem(sys.modules, "torch", fake_torch_mod)
    yield


def _make_frame(idx: int, w=640, h=480) -> CameraFrame:
    intr = CameraIntrinsics(
        fx=500.0, fy=500.0, cx=w/2, cy=h/2,
        width=w, height=h, dist_coeffs=np.zeros(5),
    )
    return CameraFrame(
        image=np.zeros((h, w, 3), dtype=np.uint8),
        timestamp=time.monotonic() + idx * 0.033,
        frame_idx=idx,
        intrinsics=intr,
        source_id="test",
    )


class TestDPVOWrapperLogic:

    def _make_estimator(self, ckpt, stride=1):
        from perception.dpvo_pose_estimator import DPVOPoseEstimator
        return DPVOPoseEstimator({
            "pose_estimator": {
                "type":               "dpvo",
                "checkpoint":         str(ckpt),
                "stride":             stride,
                "patches_per_frame":  96,
            }
        })

    def test_missing_checkpoint_raises(self, fake_dpvo_module):
        from perception.dpvo_pose_estimator import DPVOPoseEstimator
        with pytest.raises(FileNotFoundError, match="checkpoint"):
            DPVOPoseEstimator({
                "pose_estimator": {
                    "type":       "dpvo",
                    "checkpoint": "/does/not/exist.pth",
                }
            })

    def test_invalid_stride_rejected(self, fake_dpvo_module):
        from perception.dpvo_pose_estimator import DPVOPoseEstimator
        with pytest.raises(ValueError, match="stride"):
            DPVOPoseEstimator({
                "pose_estimator": {
                    "type":       "dpvo",
                    "checkpoint": str(fake_dpvo_module),
                    "stride":     0,
                }
            })

    def test_lazy_slam_construction(self, fake_dpvo_module, fake_torch):
        est = self._make_estimator(fake_dpvo_module)
        assert est._slam is None
        est.estimate(_make_frame(0))
        assert est._slam is not None
        # SLAM instantiated with the frame's H, W
        assert _FakeDPVO.last_instance.ht == 480
        assert _FakeDPVO.last_instance.wd == 640

    def test_returns_none_before_bootstrap(self, fake_dpvo_module, fake_torch):
        est = self._make_estimator(fake_dpvo_module, stride=1)
        # First 7 frames: not yet initialised
        for i in range(7):
            assert est.estimate(_make_frame(i)) is None

    def test_returns_pose_after_bootstrap(self, fake_dpvo_module, fake_torch):
        est = self._make_estimator(fake_dpvo_module, stride=1)
        for i in range(8):
            pose = est.estimate(_make_frame(i))
        # 8th call triggers is_initialized → returns CameraPose
        assert pose is not None
        assert pose.source == "dpvo"
        np.testing.assert_allclose(pose.R, np.eye(3))

    def test_stride_skips_frames(self, fake_dpvo_module, fake_torch):
        est = self._make_estimator(fake_dpvo_module, stride=3)
        for i in range(9):
            est.estimate(_make_frame(i))
        # stride=3 → DPVO called on frames 0, 3, 6 → 3 calls
        assert len(_FakeDPVO.last_instance.calls) == 3
        assert _FakeDPVO.last_instance.calls == [0, 3, 6]

    def test_stride_returns_cached_pose_between_calls(
        self, fake_dpvo_module, fake_torch,
    ):
        est = self._make_estimator(fake_dpvo_module, stride=3)
        # With stride=3, DPVO is called on frames 0, 3, 6, ..., 21.
        # The fake DPVO flips is_initialized after 8 calls (frame 21).
        poses = [est.estimate(_make_frame(i)) for i in range(25)]
        # First pose is emitted on frame 21 (8th DPVO call).
        assert poses[21] is not None
        # Frames 22, 23 are skipped by stride → return the cached pose.
        assert poses[22] is poses[21]
        assert poses[23] is poses[21]
        # Frame 24 = next DPVO call (24 % 3 == 0) → fresh pose object.
        assert poses[24] is not None
        assert poses[24] is not poses[21]

    def test_reset_clears_state(self, fake_dpvo_module, fake_torch):
        est = self._make_estimator(fake_dpvo_module, stride=1)
        for i in range(10):
            est.estimate(_make_frame(i))
        est.reset()
        assert est._slam is None
        assert est._counter == 0
        assert est._latest is None
        assert est.is_initialised is False


# ---------------------------------------------------------------------------
# Live DPVO — hardware integration test
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDPVOLive:

    def test_end_to_end_short_run(self):
        """
        Run real DPVO against the sample video. Asserts:
          - estimator becomes initialised within the bootstrap window
          - emitted poses have valid SE(3) (orthonormal R, finite t)
        """
        ckpt = Path(__file__).resolve().parent.parent / \
            "third_party" / "DPVO" / "models" / "dpvo.pth"
        if not ckpt.exists():
            pytest.skip("DPVO checkpoint not present")

        from perception.dpvo_pose_estimator import DPVOPoseEstimator

        est = DPVOPoseEstimator({
            "pose_estimator": {
                "type":       "dpvo",
                "checkpoint": str(ckpt),
                "stride":     1,
            },
        })

        import cv2
        video = Path(__file__).resolve().parent.parent / "data" / "sample.mp4"
        cap = cv2.VideoCapture(str(video))

        # DPVO initialises on its 8th *keyframe*, not its 8th input frame.
        # Keyframe acceptance depends on motion magnitude. sample.mp4 is
        # 300 frames at 30 Hz; 150 frames is a safe upper bound.
        last_pose = None
        n_pose_emitted = 0
        for i in range(150):
            ok, img = cap.read()
            if not ok:
                pytest.fail(f"sample video ran out at frame {i}")
            h, w = img.shape[:2]
            intr = CameraIntrinsics(
                fx=0.85 * w, fy=0.85 * w, cx=w/2, cy=h/2,
                width=w, height=h, dist_coeffs=np.zeros(5),
            )
            frame = CameraFrame(
                image=img,
                timestamp=time.monotonic() + i * 0.033,
                frame_idx=i,
                intrinsics=intr,
                source_id=str(video),
            )
            pose = est.estimate(frame)
            if pose is not None:
                last_pose = pose
                n_pose_emitted += 1
        cap.release()

        assert est.is_initialised, (
            f"DPVO failed to initialise within 150 frames "
            f"(slam.n={est._slam.n if est._slam else 'n/a'})"
        )
        assert n_pose_emitted >= 10, \
            f"Expected at least 10 poses after init, got {n_pose_emitted}"
        assert last_pose is not None
        # R orthonormal, det = 1
        np.testing.assert_allclose(
            last_pose.R @ last_pose.R.T, np.eye(3), atol=1e-5,
        )
        np.testing.assert_allclose(
            np.linalg.det(last_pose.R), 1.0, atol=1e-5,
        )
        assert np.all(np.isfinite(last_pose.t))
