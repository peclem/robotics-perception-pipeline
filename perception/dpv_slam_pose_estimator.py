"""
DPV-SLAM-backed ego-pose estimator.

DPV-SLAM (Lipson, Teed & Deng 2024 — "Deep Patch Visual SLAM") is the
loop-closure extension of DPVO, shipped in the same Princeton-VL
repository. It keeps DPVO's deep-patch VO front-end and adds a
loop-closure backend, so the trajectory is *drift-corrected* on
revisit instead of accumulating error monotonically.

Relationship to DPVOPoseEstimator
----------------------------------
Mechanically, DPV-SLAM is the same `dpvo.dpvo.DPVO` class with two
config flags flipped on:

  LOOP_CLOSURE          proximity-based loop closure. Pose-graph loop
                        edges added from spatial proximity of
                        keyframes; a periodic global optimisation
                        (every GLOBAL_OPT_FREQ keyframes) folds them
                        in. Pure geometry — no extra learned weights.

  CLASSIC_LOOP_CLOSURE  long-term place recognition via DBoW2
                        bag-of-words retrieval + the DPRetrieval
                        network. Catches loops the proximity test
                        misses (large drift, returning from far
                        away). Needs the DBoW2 C++ build + DPRetrieval
                        weights present under third_party/DPVO/. DPVO
                        disables it gracefully (logs + falls back to
                        proximity-only) if the import fails — so
                        requesting it is safe even when the components
                        aren't built.

This estimator therefore subclasses DPVOPoseEstimator and only
overrides the config: construction, the per-frame DPVO call, the
pose readout, bootstrap handling and reset are all inherited.

What loop closure changes for downstream consumers
--------------------------------------------------
When a loop closes, DPV-SLAM retroactively corrects *past* keyframe
poses in the pose graph. estimate() still returns the current pose,
now in the loop-corrected frame — so `position_world` can jump
discretely at a loop event. That jump is *correct* SLAM behaviour:
it's the drift being cancelled. The standard place to absorb it is
the TransformTree's `map ← odom` edge — `odom` stays continuous,
`map` jumps. The pipeline's frame chain (`map → odom → base_link →
camera_frame`) is already set up for exactly this split.

Reference
---------
Lipson, Teed & Deng — "Deep Patch Visual SLAM" (ECCV 2024).
DPVO loop-closure config: third_party/DPVO/dpvo/config.py.
"""

from __future__ import annotations

import logging

from perception.dpvo_pose_estimator import DPVOPoseEstimator

log = logging.getLogger(__name__)


class DPVSLAMPoseEstimator(DPVOPoseEstimator):
    """
    DPV-SLAM PoseEstimator backend — DPVO + loop closure.

    Parameters
    ----------
    config : full pipeline config dict. Reads the `pose_estimator`
        section. Inherits all DPVO knobs (checkpoint, stride,
        patches_per_frame) and adds:

        loop_closure          enable proximity loop closure (default True)
        backend_thresh        loop-edge distance threshold (default 64.0)
        max_edge_age          drop loop edges older than this (default 1000)
        global_opt_freq       run global optimisation every N keyframes
                              (default 15)
        classic_loop_closure  enable DBoW2 long-term loop closure
                              (default False — needs the extra build)
        loop_close_window     DBoW2 retrieval window size (default 3)
        loop_retr_thresh      DBoW2 retrieval score threshold (default 0.04)

    The estimator is a no-op-different drop-in for DPVOPoseEstimator:
    same ABC, same estimate() contract. Select via
    `pose_estimator.type: dpv_slam`.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        pe = config.get("pose_estimator", {})

        # Proximity loop closure — the always-on, no-extra-deps path.
        self._cfg.LOOP_CLOSURE   = bool(pe.get("loop_closure", True))
        self._cfg.BACKEND_THRESH = float(pe.get("backend_thresh", 64.0))
        self._cfg.MAX_EDGE_AGE   = int(pe.get("max_edge_age", 1000))
        self._cfg.GLOBAL_OPT_FREQ = int(pe.get("global_opt_freq", 15))

        # Classic (DBoW2) long-term loop closure — opt-in; DPVO itself
        # disables it gracefully if the DBoW2 / DPRetrieval components
        # aren't importable, so requesting it can't crash construction.
        self._cfg.CLASSIC_LOOP_CLOSURE   = bool(
            pe.get("classic_loop_closure", False))
        self._cfg.LOOP_CLOSE_WINDOW_SIZE = int(
            pe.get("loop_close_window", 3))
        self._cfg.LOOP_RETR_THRESH       = float(
            pe.get("loop_retr_thresh", 0.04))

        # Poses are loop-corrected — tag them distinctly from raw VO.
        self._source = "dpv_slam"

        log.info(
            "DPVSLAMPoseEstimator: stride=%d patches=%d "
            "loop_closure=%s classic_loop_closure=%s "
            "(backend_thresh=%.1f global_opt_freq=%d)",
            self._stride, self._cfg.PATCHES_PER_FRAME,
            self._cfg.LOOP_CLOSURE, self._cfg.CLASSIC_LOOP_CLOSURE,
            self._cfg.BACKEND_THRESH, self._cfg.GLOBAL_OPT_FREQ,
        )
