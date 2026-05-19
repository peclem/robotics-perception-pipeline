"""
DepthEstimator factory.

Single source of truth for constructing the perception stack's depth
backend, used by both `launch.py` (standalone Python pipeline) and the
ROS2 `depth_node` adapter. Mirrors the pose_estimator_factory pattern.

Adding a new depth backend is a one-file change here; both consumers
pick it up automatically.
"""

from __future__ import annotations

import logging

from perception.depth_estimator import (
    DepthEstimator, DepthAnythingEstimator, NullDepthEstimator,
    RAFTStereoDepthEstimator, StereoSGBMDepthEstimator,
)

log = logging.getLogger(__name__)


def build_depth_estimator(cfg) -> DepthEstimator:
    """Construct a DepthEstimator from a typed DepthConfig (cfg.depth).

    `depth.enabled=False` always returns NullDepthEstimator regardless
    of `depth.type` — master switch, backward-compatible.
    """
    dc = cfg.depth
    if not dc.enabled or dc.type == "null":
        return NullDepthEstimator()

    if dc.type == "depth_anything":
        log.info("Loading Depth Anything V2: %s ...", dc.model)
        est = DepthAnythingEstimator(device=dc.device, model_name=dc.model)
        est.warmup()
        log.info("Depth Anything V2 ready. Latency: %.1f ms",
                 est.mean_inference_ms)
        return est

    if dc.type == "stereo_sgbm":
        log.info(
            "Building StereoSGBMDepthEstimator (CPU; num_disp=%d, block=%d)",
            dc.sgbm_num_disparities, dc.sgbm_block_size,
        )
        est = StereoSGBMDepthEstimator(
            min_disparity=dc.sgbm_min_disparity,
            num_disparities=dc.sgbm_num_disparities,
            block_size=dc.sgbm_block_size,
        )
        est.warmup()
        return est

    if dc.type == "raft_stereo":
        log.info(
            "Building RAFTStereoDepthEstimator (GPU; iters=%d, repo=%s)",
            dc.raft_iters, dc.raft_repo_dir,
        )
        est = RAFTStereoDepthEstimator(
            repo_dir=dc.raft_repo_dir,
            checkpoint=dc.raft_checkpoint,
            device=dc.device,
            iters=dc.raft_iters,
        )
        est.warmup()
        return est

    raise ValueError(
        f"Unknown depth.type={dc.type!r}. "
        "Supported: 'null', 'depth_anything', 'stereo_sgbm', 'raft_stereo'."
    )
