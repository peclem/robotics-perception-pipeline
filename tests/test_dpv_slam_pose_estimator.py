"""
Tests for the DPV-SLAM PoseEstimator wrapper.

DPV-SLAM = DPVO + loop closure. The wrapper is a thin subclass of
DPVOPoseEstimator that only flips loop-closure config flags — so the
tests here focus on *config*: that the LOOP_CLOSURE / CLASSIC_LOOP_CLOSURE
flags and their tuning knobs land on the DPVO cfg object, that the
defaults are sane, and that the factory dispatches type='dpv_slam'.

The per-frame pose-readout path is inherited unchanged from
DPVOPoseEstimator and is covered by test_dpvo_pose_estimator.py — not
re-tested here.

TestDPVSLAMConfig : construction + loop-closure config (fake DPVO module)
TestDPVSLAMFactory: pose_estimator_factory dispatch
TestDPVSLAMLive   : end-to-end against the real DPVO + GPU (integration)
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fake DPVO module — construction-level only (no slam object / pose readout).
# ---------------------------------------------------------------------------

class _FakeCfg:
    """yacs-CfgNode stand-in. clone() returns self; attrs set freely."""
    PATCHES_PER_FRAME = 96
    # The loop-closure defaults DPVO's real config.py carries — the
    # wrapper overwrites them, but they exist so a plain DPVO cfg is
    # realistic.
    LOOP_CLOSURE = False
    CLASSIC_LOOP_CLOSURE = False
    BACKEND_THRESH = 64.0
    MAX_EDGE_AGE = 1000
    GLOBAL_OPT_FREQ = 15
    LOOP_CLOSE_WINDOW_SIZE = 3
    LOOP_RETR_THRESH = 0.04

    def clone(self):
        return self


@pytest.fixture
def fake_dpvo(tmp_path, monkeypatch):
    """
    Install a fake `dpvo` / `dpvo.config` / `dpvo.dpvo` into sys.modules
    so DPVSLAMPoseEstimator constructs without the real CUDA extension.
    Yields the fake checkpoint path. Resets the shared _FakeCfg flags so
    tests don't bleed into each other.
    """
    _FakeCfg.LOOP_CLOSURE = False
    _FakeCfg.CLASSIC_LOOP_CLOSURE = False

    fake_config = types.ModuleType("dpvo.config")
    fake_config.cfg = _FakeCfg()

    fake_dpvo_mod = types.ModuleType("dpvo.dpvo")
    fake_dpvo_mod.DPVO = object   # never instantiated — estimate() not called

    fake_root = types.ModuleType("dpvo")

    monkeypatch.setitem(sys.modules, "dpvo", fake_root)
    monkeypatch.setitem(sys.modules, "dpvo.config", fake_config)
    monkeypatch.setitem(sys.modules, "dpvo.dpvo", fake_dpvo_mod)

    ckpt = tmp_path / "dpvo.pth"
    ckpt.write_bytes(b"\x00")
    yield ckpt


def _cfg(ckpt, **pose_overrides):
    """A pipeline-config dict with a pose_estimator section."""
    pe = {"type": "dpv_slam", "checkpoint": str(ckpt)}
    pe.update(pose_overrides)
    return {"pose_estimator": pe}


# ---------------------------------------------------------------------------
# TestDPVSLAMConfig
# ---------------------------------------------------------------------------

class TestDPVSLAMConfig:

    def test_loop_closure_enabled_by_default(self, fake_dpvo):
        from perception.dpv_slam_pose_estimator import DPVSLAMPoseEstimator
        est = DPVSLAMPoseEstimator(_cfg(fake_dpvo))
        assert est._cfg.LOOP_CLOSURE is True

    def test_classic_loop_closure_off_by_default(self, fake_dpvo):
        from perception.dpv_slam_pose_estimator import DPVSLAMPoseEstimator
        est = DPVSLAMPoseEstimator(_cfg(fake_dpvo))
        # DBoW2 long-term LC needs the extra build — opt-in only.
        assert est._cfg.CLASSIC_LOOP_CLOSURE is False

    def test_loop_closure_can_be_disabled(self, fake_dpvo):
        from perception.dpv_slam_pose_estimator import DPVSLAMPoseEstimator
        est = DPVSLAMPoseEstimator(_cfg(fake_dpvo, loop_closure=False))
        assert est._cfg.LOOP_CLOSURE is False

    def test_classic_loop_closure_opt_in(self, fake_dpvo):
        from perception.dpv_slam_pose_estimator import DPVSLAMPoseEstimator
        est = DPVSLAMPoseEstimator(
            _cfg(fake_dpvo, classic_loop_closure=True))
        assert est._cfg.CLASSIC_LOOP_CLOSURE is True

    def test_tuning_knobs_land_on_cfg(self, fake_dpvo):
        from perception.dpv_slam_pose_estimator import DPVSLAMPoseEstimator
        est = DPVSLAMPoseEstimator(_cfg(
            fake_dpvo,
            backend_thresh=48.0,
            max_edge_age=500,
            global_opt_freq=10,
            loop_close_window=5,
            loop_retr_thresh=0.06,
        ))
        assert est._cfg.BACKEND_THRESH == 48.0
        assert est._cfg.MAX_EDGE_AGE == 500
        assert est._cfg.GLOBAL_OPT_FREQ == 10
        assert est._cfg.LOOP_CLOSE_WINDOW_SIZE == 5
        assert est._cfg.LOOP_RETR_THRESH == 0.06

    def test_tuning_defaults(self, fake_dpvo):
        from perception.dpv_slam_pose_estimator import DPVSLAMPoseEstimator
        est = DPVSLAMPoseEstimator(_cfg(fake_dpvo))
        assert est._cfg.BACKEND_THRESH == 64.0
        assert est._cfg.MAX_EDGE_AGE == 1000
        assert est._cfg.GLOBAL_OPT_FREQ == 15

    def test_source_tag_is_dpv_slam(self, fake_dpvo):
        # Loop-corrected poses must be tagged distinctly from raw VO so
        # downstream diagnostics can tell them apart.
        from perception.dpv_slam_pose_estimator import DPVSLAMPoseEstimator
        est = DPVSLAMPoseEstimator(_cfg(fake_dpvo))
        assert est._source == "dpv_slam"

    def test_inherits_dpvo_knobs(self, fake_dpvo):
        from perception.dpv_slam_pose_estimator import DPVSLAMPoseEstimator
        est = DPVSLAMPoseEstimator(_cfg(fake_dpvo, stride=3,
                                         patches_per_frame=64))
        assert est._stride == 3
        assert est._cfg.PATCHES_PER_FRAME == 64

    def test_repr_says_dpv_slam(self, fake_dpvo):
        from perception.dpv_slam_pose_estimator import DPVSLAMPoseEstimator
        est = DPVSLAMPoseEstimator(_cfg(fake_dpvo))
        assert "DPVSLAMPoseEstimator" in repr(est)

    def test_missing_checkpoint_raises(self, fake_dpvo, tmp_path):
        from perception.dpv_slam_pose_estimator import DPVSLAMPoseEstimator
        bad = {"pose_estimator": {"type": "dpv_slam",
                                  "checkpoint": str(tmp_path / "nope.pth")}}
        with pytest.raises(FileNotFoundError):
            DPVSLAMPoseEstimator(bad)


# ---------------------------------------------------------------------------
# TestDPVSLAMFactory
# ---------------------------------------------------------------------------

class TestDPVSLAMFactory:

    def test_factory_dispatches_dpv_slam(self, fake_dpvo):
        from perception.config_loader import PipelineConfig
        from perception.pose_estimator_factory import build_visual_pose_estimator
        from perception.dpv_slam_pose_estimator import DPVSLAMPoseEstimator

        cfg = PipelineConfig()
        cfg.pose_estimator.type = "dpv_slam"
        cfg.pose_estimator.checkpoint = str(fake_dpvo)
        raw = cfg.as_dict()
        est = build_visual_pose_estimator(cfg, raw)
        assert isinstance(est, DPVSLAMPoseEstimator)

    def test_config_validation_accepts_dpv_slam(self, tmp_path):
        # dpv_slam must pass config validation (was missing from the
        # pose_estimator.type whitelist before this backend landed).
        # Base off the real default.yaml so all required sections are
        # present — we only override pose_estimator.type.
        import yaml
        from perception.config_loader import load_config

        with open("config/default.yaml") as f:
            raw = yaml.safe_load(f)
        raw.setdefault("pose_estimator", {})["type"] = "dpv_slam"

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(raw))
        cfg = load_config(str(cfg_path))   # raises if validation rejects it
        assert cfg.pose_estimator.type == "dpv_slam"


# ---------------------------------------------------------------------------
# TestDPVSLAMLive — real DPVO + GPU
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDPVSLAMLive:

    def test_end_to_end_short_run(self):
        """
        Smoke-test against the real DPVO with loop closure enabled.
        Requires the DPVO CUDA extension built + a GPU + a short clip
        at data/sample.mp4. Excluded from CI.
        """
        sample = Path("data/sample.mp4")
        if not sample.exists():
            pytest.skip("data/sample.mp4 not present")

        import cv2
        from perception.camera_interface import CameraFrame, CameraIntrinsics
        from perception.dpv_slam_pose_estimator import DPVSLAMPoseEstimator

        est = DPVSLAMPoseEstimator({"pose_estimator": {
            "type": "dpv_slam", "stride": 2, "loop_closure": True,
        }})
        cap = cv2.VideoCapture(str(sample))
        intr = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240,
                                width=640, height=480)
        got_pose = False
        for i in range(120):
            ok, img = cap.read()
            if not ok:
                break
            img = cv2.resize(img, (640, 480))
            pose = est.estimate(CameraFrame(
                image=img, timestamp=float(i) / 30.0,
                frame_idx=i, intrinsics=intr, source_id="test",
            ))
            if pose is not None:
                got_pose = True
        cap.release()
        assert got_pose, "DPV-SLAM produced no pose over 120 frames"
