"""
Typed configuration loader for the robotics perception pipeline.

Why typed dataclasses over raw dicts
--------------------------------------
Every module in steps 1-5 does config.get("tracker", {}).get("high_thresh", 0.5).
Problems with that pattern:
  - Silent key typos return the default, never an error
  - No IDE autocomplete on nested keys
  - Type coercion happens inconsistently at each call site
  - No centralised validation — a negative Q variance is only
    caught when the KF produces NaN, not at startup

The loader solves all of this:
  1. Reads YAML once at startup
  2. Applies environment variable overrides
  3. Validates all values with explicit range and cross-field checks
  4. Returns PipelineConfig — fully typed nested dataclasses

Backward compatibility
----------------------
All step 1-5 modules take config: dict.
PipelineConfig.as_dict() returns the exact nested structure they expect.
No changes to steps 1-5 are required.

Usage
-----
    from perception.config_loader import load_config
    cfg = load_config("config/default.yaml")

    # Typed access (new code)
    cfg.tracker.high_thresh
    cfg.kalman_filter.process_noise.q_position

    # Dict access (backward compat with steps 1-5)
    ByteTracker(cfg.as_dict())
    KalmanFilter(state, cfg.as_dict())
    SyntheticCamera(cfg.as_dict(), num_frames=90)
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment variable overrides
# Format: ENV_VAR → (dot-path into raw dict, cast type)
# Applied after YAML load, before validation.
# ---------------------------------------------------------------------------

OVERRIDES: Dict[str, tuple[str, type]] = {
    "DEVICE":        ("detector.device",               str),
    "CONFIDENCE":    ("detector.confidence_threshold",  float),
    "HIGH_THRESH":   ("tracker.high_thresh",            float),
    "MAX_AGE":       ("tracker.max_age",                int),
    "LOG_LEVEL":     ("pipeline.log_level",             str),
    "RERUN_ENABLED": ("visualization.rerun_enabled",    bool),
}


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PipelineSettings:
    name:              str           = "robotics-perception-pipeline"
    log_level:         str           = "INFO"
    target_hz:         float         = 30.0
    output_video_path: Optional[str] = None
    seed:              int           = 42

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CameraConfig:
    device_index:    int           = 0
    width:           int           = 1280
    height:          int           = 720
    fps:             int           = 30
    intrinsics_path: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class VideoConfig:
    input_path:   str            = "data/test_clip.mp4"
    output_path:  str            = "data/output.mp4"
    playback_fps: Optional[float] = None
    loop:         bool           = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SyntheticCameraConfig:
    width:       int   = 640
    height:      int   = 480
    num_frames:  int   = 300
    fps:         float = 30.0
    num_objects: int   = 3
    seed:        int   = 42

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectorConfig:
    model:                str                 = "yolov8n.pt"
    confidence_threshold: float               = 0.25
    iou_threshold:        float               = 0.45
    device:               str                 = "cuda:0"
    half_precision:       bool                = True
    class_filter:         Optional[List[int]] = None
    max_detections:       int                 = 100
    img_size:             int                 = 640

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrackerConfig:
    high_thresh:        float = 0.50
    low_thresh:         float = 0.10
    new_track_thresh:   float = 0.50
    iou_threshold:      float = 0.30
    max_age:            int   = 30
    min_hits:           int   = 1
    use_ekf:            bool  = False
    # Appearance-aware association. When True AND detection embeddings
    # are computed (typically via DINOv2), the cost matrix blends IoU
    # with cosine distance: cost = (1 - w) * iou + w * appearance.
    # Defaults match StrongSORT (w=0.25) and Deep OC-SORT (EMA=0.9).
    use_appearance:     bool  = False
    appearance_weight:  float = 0.25
    appearance_ema:     float = 0.9
    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class InitialCovarianceConfig:
    p_position: float = 10.0
    p_size:     float = 10.0
    p_velocity: float = 100.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProcessNoiseConfig:
    q_position: float = 1.0
    q_size:     float = 1.0
    q_velocity: float = 0.1
    q_vel_size: float = 0.02

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class MeasurementNoiseConfig:
    r_center: float = 1.0
    r_size:   float = 1.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class KalmanFilterConfig:
    initial_covariance: InitialCovarianceConfig = field(
        default_factory=InitialCovarianceConfig
    )
    process_noise:      ProcessNoiseConfig      = field(
        default_factory=ProcessNoiseConfig
    )
    measurement_noise:  MeasurementNoiseConfig  = field(
        default_factory=MeasurementNoiseConfig
    )

    def as_dict(self) -> dict:
        return {
            "initial_covariance": self.initial_covariance.as_dict(),
            "process_noise":      self.process_noise.as_dict(),
            "measurement_noise":  self.measurement_noise.as_dict(),
        }

@dataclass
class EKFInitialCovarianceConfig:
    p_position: float = 10.0
    p_size:     float = 10.0
    p_velocity: float = 100.0
    p_omega:    float = 1.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class EKFProcessNoiseConfig:
    q_position: float = 1.0
    q_size:     float = 1.0
    q_velocity: float = 0.1
    q_vel_size: float = 0.02
    q_omega:    float = 0.01

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtendedKalmanFilterConfig:
    initial_covariance: EKFInitialCovarianceConfig = field(
        default_factory=EKFInitialCovarianceConfig
    )
    process_noise:      EKFProcessNoiseConfig      = field(
        default_factory=EKFProcessNoiseConfig
    )
    measurement_noise:  MeasurementNoiseConfig     = field(
        default_factory=MeasurementNoiseConfig
    )

    def as_dict(self) -> dict:
        return {
            "initial_covariance": self.initial_covariance.as_dict(),
            "process_noise":      self.process_noise.as_dict(),
            "measurement_noise":  self.measurement_noise.as_dict(),
        }

@dataclass
class VisualizationConfig:
    rerun_enabled:           bool         = True
    rerun_app_id:            str          = "robotics_perception_pipeline"
    rerun_host:              str          = "auto"          # ← add
    rerun_port:              int          = 9876            # ← add
    rerun_save_path:         Optional[str]= None            # ← add
    show_bboxes:             bool         = True
    show_track_ids:          bool         = True
    show_velocity:           bool         = True
    show_covariance_ellipse: bool         = True
    show_nis:                bool         = False
    show_stats_overlay:      bool         = True
    show_track_history:      bool         = True
    bbox_thickness:          int          = 2
    velocity_arrow_scale:    float        = 0.5
    track_history_max_points: int         = 32

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkConfig:
    dataset_path: str                 = "data/MOT17"
    split:        str                 = "train"
    sequences:    Optional[List[str]] = None
    output_dir:   str                 = "data/benchmark_results"

    def as_dict(self) -> dict:
        return asdict(self)

@dataclass
class DepthConfig:
    """
    Depth backend selection.

    type
        'depth_anything' — DepthAnythingEstimator (monocular metric, GPU)
        'stereo_sgbm'    — StereoSGBMDepthEstimator (classical, CPU,
                           needs CameraFrame.right_image + intrinsics.baseline_m)
        'raft_stereo'    — RAFTStereoDepthEstimator (neural stereo, GPU,
                           same input contract as stereo_sgbm). Requires
                           the RAFT-Stereo repo cloned into
                           `raft_repo_dir` and weights at `raft_checkpoint`.
        'null'           — NullDepthEstimator (no depth)

    enabled
        Master switch retained for backwards compatibility — when False,
        NullDepthEstimator is used regardless of `type`. New configs
        can equivalently set `type: null`.

    SGBM tunables (only used when type == 'stereo_sgbm')
    RAFT-Stereo tunables (only used when type == 'raft_stereo')
    """
    enabled: bool = False
    type:    str  = "depth_anything"
    model:   str  = "ZoeD_N"
    device:  str  = "cuda"
    sgbm_min_disparity:   int = 0
    sgbm_num_disparities: int = 96   # must be divisible by 16
    sgbm_block_size:      int = 7    # must be odd
    # RAFT-Stereo (neural stereo). Defaults assume the repo is cloned
    # into third_party/RAFT-Stereo with the middlebury checkpoint.
    raft_repo_dir:   str = "third_party/RAFT-Stereo"
    raft_checkpoint: str = "third_party/RAFT-Stereo/models/raftstereo-middlebury.pth"
    raft_iters:      int = 16

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorldModelConfig:
    max_history:   int   = 30
    lost_timeout_s: float = 1.0
    # Name of the frame in which depth_estimate.position_3d is expressed.
    # Must match coordinate_frames.camera_frame for tree lookups to resolve.
    camera_frame:  str   = "camera_frame"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class StaticExtrinsicConfig:
    """
    Static SE(3) transform between two frames.

    R is a 3×3 rotation as a row-major list of 9 floats.
    t is a 3-vector translation in metres.
    """
    parent: str           = "base_link"
    child:  str           = "camera_frame"
    R:      List[float]   = field(default_factory=lambda: [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ])
    t:      List[float]   = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CoordinateFramesConfig:
    """
    TransformTree topology and static extrinsics.

    The pipeline builds a TransformTree(root_frame) at startup and
    registers each entry in `static_extrinsics` as a static edge.
    Dynamic edges (ego-motion) are pushed each frame by the pose
    estimator into camera_frame's parent edge.
    """
    enabled:           bool                          = False
    root_frame:        str                           = "map"
    camera_frame:      str                           = "camera_frame"
    static_extrinsics: List[StaticExtrinsicConfig]   = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "enabled":           self.enabled,
            "root_frame":        self.root_frame,
            "camera_frame":      self.camera_frame,
            "static_extrinsics": [e.as_dict() for e in self.static_extrinsics],
        }


@dataclass
class IMUConfig:
    """
    IMU backend selection + noise model + synthetic-source params.

    type
        'null'      — NullIMU (no IMU data; pre-integration returns identity)
        'synthetic' — SyntheticIMU (configurable motion preset; useful
                      for offline pre-integration validation)
        Real-hardware backends slot in here ('bno055', 'bmi088', etc.)
        when the project gets actual sensors.

    frame_id
        TF frame the IMU samples are expressed in. Static extrinsic
        base_link → imu_link defines the rigid mounting; pre-integrated
        measurements then sit naturally in base_link via the tree.

    Noise densities (sigma_*_n)
        From the IMU datasheet's noise spec ("rate noise density" for
        gyro, "noise density" for accel). Defaults match a good MEMS
        chip (Bosch BMI088 ballpark). The pre-integrator uses these
        for covariance propagation.

    Synthetic-only fields apply when type == 'synthetic'.
    """
    type:             str   = "null"
    frame_id:         str   = "imu_link"
    rate_hz:          float = 200.0
    sigma_gyro_n:     float = 1.7e-4   # rad/s/√Hz
    sigma_accel_n:    float = 2.0e-3   # m/s²/√Hz
    synthetic_motion: str   = "stationary_with_gravity"
    synthetic_noise_std_accel: float = 0.0
    synthetic_noise_std_gyro:  float = 0.0
    synthetic_seed:   int   = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class HealthMonitorConfig:
    """
    Per-stage latency budgets + breach thresholds. Each entry in
    stage_budgets_ms maps a stage name to its budget in milliseconds.
    Stages registered at runtime that aren't in this table fall back
    to `default_budget_ms`.
    """
    enabled:           bool                = True
    warn_after:        int                 = 3
    error_after:       int                 = 30
    stale_after_s:     float               = 5.0
    window:            int                 = 60
    default_budget_ms: float               = 50.0
    log_period_s:      float               = 5.0
    stage_budgets_ms:  Dict[str, float]    = field(default_factory=lambda: {
        # Defaults tuned for a 30 Hz pipeline (33.3 ms total budget).
        # `semantic` is generous (40 ms) because Mask2Former-Swin-T
        # alone consumes most of a 30 Hz frame budget — when it's
        # enabled the pipeline is effectively running slower; the
        # budget acknowledges that rather than spamming WARNs.
        "detector":   12.0,
        "depth":      20.0,
        "pose":       25.0,
        "tracker":    3.0,
        "scene_graph": 5.0,
        "appearance": 15.0,
        "semantic":   40.0,
        "frame_total": 40.0,
    })

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class AppearanceConfig:
    """
    Selects the AppearanceExtractor backend used for ReID embeddings.

    type
        'null'   — NullAppearanceExtractor (re-association disabled
                   or spatial-only)
        'dinov2' — Meta DINOv2 foundation features. Generalises across
                   the 80 COCO classes — unlike person-specific ReID.
    model
        HuggingFace model id. facebook/dinov2-small fits comfortably
        on a 12 GB GPU alongside the rest of the stack.
    """
    type:   str = "null"
    model:  str = "facebook/dinov2-small"
    device: str = "cuda"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class WorldMapConfig:
    """
    Long-term spatial memory of STATIC + SEMI_STATIC objects with
    appearance-based re-association on revisit.
    """
    enabled:              bool   = False
    spatial_gate_m:       float  = 1.5
    similarity_threshold: float  = 0.75
    allow_spatial_only:   bool   = True
    # Eviction (both off by default → monotonic-growth baseline).
    max_age_s:            float  = 0.0     # 0 disables aging
    max_entries:          int    = 0       # 0 disables capacity cap

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class OccupancyGridConfig:
    """
    Dynamic obstacle grid parameters. See world_model.OccupancyGridParams
    for the matching builder dataclass; this just exposes the same knobs
    in the validated config layer.
    """
    enabled:             bool             = False
    resolution_m:        float            = 0.05
    size_x_m:            float            = 20.0
    size_y_m:            float            = 20.0
    origin_x_m:          float            = -10.0
    origin_y_m:          float            = -10.0
    default_inflation_m: float            = 0.5
    min_inflation_m:     float            = 0.10
    per_class_inflation_m: Dict[str, float] = field(default_factory=lambda: {
        "person":     0.40,
        "bicycle":    0.60,
        "car":        2.00,
        "motorcycle": 0.80,
        "bus":        3.00,
        "truck":      2.50,
    })

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DrivableCostmapConfig:
    """
    Drivable-freespace costmap: top-down OccupancyGrid produced by
    projecting the semantic segmenter's drivable mask through the
    camera pose. Pure consumer-side config — the projector lives in
    world_model.drivable_projector. Reuses OccupancyGridConfig grid
    spec so the drivable costmap shares the dynamic-obstacle layer's
    frame and resolution (Nav2 fuses them cleanly).

    `use_depth=true` enables the depth-aware backend when a dense
    depth map is available on the current frame; falls back to
    flat-ground IPM otherwise. `z_ground_m` is the assumed ground
    plane in the world frame.
    """
    enabled:     bool  = False
    use_depth:   bool  = True
    z_ground_m:  float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RoomLayerConfigCfg:
    """
    Typed mirror of world_model.room_layer.RoomLayerConfig. Defaults
    match the standalone module's defaults.

    enabled
        Master switch for the room layer. v1 doesn't have live-pipeline
        wiring yet (matches the VIO EKF discipline) — when the wiring
        lands, this flag will gate it.
    erosion_m
        Half the maximum doorway width that should be closed during
        room separation. Tune for your environment: residential ~0.45,
        warehouse ~0.6, narrow apartment ~0.3.
    """
    enabled:            bool  = False
    erosion_m:          float = 0.45
    min_area_m2:        float = 1.0
    polygon_simplify_m: float = 0.05

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Occupancy3DConfig:
    """
    Sparse 3D obstacle grid parameters. See world_model.Occupancy3DParams
    for the matching builder dataclass; this just exposes the same knobs
    in the validated config layer.

    Defaults: 20×20×3 m volume centred on the robot in XY, spanning
    -0.5..+2.5 m in Z (workspace for a standing person / table-height
    manipulator), at 10 cm resolution. ~1.2 M voxels of extent, but the
    builder only stores the occupied ones.
    """
    enabled:             bool             = False
    resolution_m:        float            = 0.10
    size_x_m:            float            = 20.0
    size_y_m:            float            = 20.0
    size_z_m:            float            = 3.0
    origin_x_m:          float            = -10.0
    origin_y_m:          float            = -10.0
    origin_z_m:          float            = -0.5
    default_inflation_m: float            = 0.5
    min_inflation_m:     float            = 0.10
    per_class_inflation_m: Dict[str, float] = field(default_factory=lambda: {
        "person":     0.40,
        "bicycle":    0.60,
        "car":        2.00,
        "motorcycle": 0.80,
        "bus":        3.00,
        "truck":      2.50,
    })

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PoseEstimatorConfig:
    """
    Selects which PoseEstimator backend to instantiate.

    type
        'null'    — NullPoseEstimator (default; camera-frame-only mode)
        'orbslam' — ORB-SLAM3 (planned)
        'dpvo'    — Deep Patch Visual Odometry (implemented)

    DPVO-specific
    -------------
    checkpoint        : path to dpvo.pth weights
    stride            : run DPVO every Nth frame (1 = every frame).
                        stride=2 → pose at 15 Hz, leaves comfortable
                        budget on this hardware (see benchmark).
    patches_per_frame : DPVO's accuracy/speed knob (default 96)
    """
    type:              str           = "null"
    checkpoint:        Optional[str] = None
    stride:            int           = 2
    patches_per_frame: int           = 96

    def as_dict(self) -> dict:
        return asdict(self)

@dataclass
class SemanticConfig:
    """
    Semantic segmentation backend + per-dataset settings.

    type
        'null'       — NullSemanticSegmenter (default; no GPU cost)
        'mask2former'— Mask2FormerSemanticSegmenter (HuggingFace)
        Future backends slot in here ('oneformer', 'openseed', ...).

    dataset
        Hint for downstream helpers (drivable_mask,
        semantic_class_stability) so they can look up the right class
        tables. Should match the dataset the chosen `model` was
        fine-tuned on. 'cityscapes' | 'ade20k' covered out of the box.
    """
    enabled: bool = False
    type:    str  = "null"
    model:   str  = "facebook/mask2former-swin-tiny-cityscapes-semantic"
    device:  str  = "cuda"
    dataset: str  = "cityscapes"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class VIOConfig:
    """
    Visual-inertial error-state EKF tuning. Mirrors
    state_estimation.visual_inertial_ekf.VIOConfig — the dataclass is
    duplicated here so config validation lives next to all the other
    typed config and the EKF stays standalone.

    enabled
        Master switch; reserved for when the live pipeline wires the
        fuser in. The EKF class itself is usable without this — it's a
        pure-Python component.
    gravity_w
        World-frame gravity vector. Default Z-up world → -9.81 along Z.

    Validation: every std / random walk must be positive.
    """
    enabled:                  bool  = False
    init_position_std_m:      float = 0.10
    init_velocity_std_mps:    float = 0.10
    init_orientation_std_rad: float = 0.05
    init_bias_gyro_std:       float = 1e-3
    init_bias_accel_std:      float = 1e-2
    bias_gyro_random_walk:    float = 1e-5
    bias_accel_random_walk:   float = 1e-4
    visual_position_std_m:    float = 0.05
    visual_orientation_std_rad: float = 0.02
    gravity_w:                List[float] = field(
        default_factory=lambda: [0.0, 0.0, -9.81]
    )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipelineConfig:
    """
    Fully typed pipeline configuration.
    Construct only via load_config() — never directly.
    """
    pipeline:               PipelineSettings           = field(default_factory=PipelineSettings)
    camera:                 CameraConfig               = field(default_factory=CameraConfig)
    video:                  VideoConfig                = field(default_factory=VideoConfig)
    synthetic_camera:       SyntheticCameraConfig      = field(default_factory=SyntheticCameraConfig)
    detector:               DetectorConfig             = field(default_factory=DetectorConfig)
    tracker:                TrackerConfig              = field(default_factory=TrackerConfig)
    kalman_filter:          KalmanFilterConfig         = field(default_factory=KalmanFilterConfig)
    visualization:          VisualizationConfig        = field(default_factory=VisualizationConfig)
    benchmark:              BenchmarkConfig            = field(default_factory=BenchmarkConfig)
    extended_kalman_filter: ExtendedKalmanFilterConfig = field(default_factory=ExtendedKalmanFilterConfig)
    depth:                  DepthConfig                = field(default_factory=DepthConfig)
    world_model:            WorldModelConfig           = field(default_factory=WorldModelConfig)
    coordinate_frames:      CoordinateFramesConfig     = field(default_factory=CoordinateFramesConfig)
    pose_estimator:         PoseEstimatorConfig        = field(default_factory=PoseEstimatorConfig)
    occupancy_grid:         OccupancyGridConfig        = field(default_factory=OccupancyGridConfig)
    occupancy_3d:           Occupancy3DConfig          = field(default_factory=Occupancy3DConfig)
    drivable_costmap:       DrivableCostmapConfig      = field(default_factory=DrivableCostmapConfig)
    room_layer:             RoomLayerConfigCfg         = field(default_factory=RoomLayerConfigCfg)
    appearance:             AppearanceConfig           = field(default_factory=AppearanceConfig)
    world_map:              WorldMapConfig             = field(default_factory=WorldMapConfig)
    health_monitor:         HealthMonitorConfig        = field(default_factory=HealthMonitorConfig)
    imu:                    IMUConfig                  = field(default_factory=IMUConfig)
    vio:                    VIOConfig                  = field(default_factory=VIOConfig)
    semantic:               SemanticConfig             = field(default_factory=SemanticConfig)
    def as_dict(self) -> dict:
        """
        Full nested dict — structure matches the YAML exactly.
        Pass to any step 1-5 module:
            ByteTracker(cfg.as_dict())
            KalmanFilter(state, cfg.as_dict())
            SyntheticCamera(cfg.as_dict(), num_frames=90)
        """
        return {
            "pipeline":               self.pipeline.as_dict(),
            "camera":                 self.camera.as_dict(),
            "video":                  self.video.as_dict(),
            "synthetic_camera":       self.synthetic_camera.as_dict(),
            "detector":               self.detector.as_dict(),
            "tracker":                self.tracker.as_dict(),
            "kalman_filter":          self.kalman_filter.as_dict(),
            "visualization":          self.visualization.as_dict(),
            "benchmark":              self.benchmark.as_dict(),
            "extended_kalman_filter": self.extended_kalman_filter.as_dict(),
            "depth":                  self.depth.as_dict(),
            "world_model":            self.world_model.as_dict(),
            "coordinate_frames":      self.coordinate_frames.as_dict(),
            "pose_estimator":         self.pose_estimator.as_dict(),
            "occupancy_grid":         self.occupancy_grid.as_dict(),
            "occupancy_3d":           self.occupancy_3d.as_dict(),
            "room_layer":             self.room_layer.as_dict(),
            "appearance":             self.appearance.as_dict(),
            "world_map":              self.world_map.as_dict(),
            "health_monitor":         self.health_monitor.as_dict(),
            "imu":                    self.imu.as_dict(),
            "vio":                    self.vio.as_dict(),
            "semantic":               self.semantic.as_dict(),
        }

    def section_dict(self, section: str) -> dict:
        """Single section as dict — cfg.section_dict("tracker")."""
        return self.as_dict()[section]


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ConfigValidationError(Exception):
    """Raised when config validation fails. Always fatal."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_config(path: str | Path = "config/default.yaml") -> PipelineConfig:
    """
    Load, validate, and return a typed PipelineConfig.

    Steps:
        1. Read YAML
        2. Apply environment variable overrides
        3. Validate — collect ALL errors, raise once with full list
        4. Build and return PipelineConfig

    Raises
    ------
    FileNotFoundError     — config file does not exist
    ConfigValidationError — any parameter is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Run from the repo root directory."
        )

    with open(path) as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    _apply_env_overrides(raw)
    _validate(raw)
    return _build(raw)


# ---------------------------------------------------------------------------
# Environment overrides
# ---------------------------------------------------------------------------

def _set_nested(d: dict, dot_path: str, value: Any) -> None:
    keys = dot_path.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _apply_env_overrides(raw: dict) -> None:
    for env_var, (dot_path, cast) in OVERRIDES.items():
        value_str = os.environ.get(env_var)
        if value_str is None:
            continue
        try:
            value = value_str.lower() in ("true", "1", "yes") if cast is bool else cast(value_str)
        except (ValueError, TypeError) as exc:
            warnings.warn(
                f"Environment variable {env_var}={value_str!r} could not be "
                f"cast to {cast.__name__}: {exc}. Override ignored.",
                stacklevel=3,
            )
            continue
        _set_nested(raw, dot_path, value)
        log.info("Config override from env: %s = %r", dot_path, value)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require_in_range(
    errors: List[str], section: dict, full_key: str,
    lo: float, hi: float, exclusive: bool = False,
) -> None:
    key = full_key.split(".")[-1]
    val = section.get(key)
    if val is None:
        return
    if not isinstance(val, (int, float)):
        errors.append(f"{full_key}: expected a number, got {type(val).__name__}.")
        return
    ok = (lo < val < hi) if exclusive else (lo <= val <= hi)
    bounds = f"({'(' if exclusive else '['}{lo}, {hi}{')' if exclusive else ']'})"
    if not ok:
        errors.append(f"{full_key}={val} is out of valid range {bounds}.")


def _require_positive_float(errors: List[str], section: dict, full_key: str) -> None:
    key = full_key.split(".")[-1]
    val = section.get(key)
    if val is None:
        return
    if not isinstance(val, (int, float)):
        errors.append(f"{full_key}: expected a positive number, got {type(val).__name__}.")
    elif val <= 0.0:
        errors.append(
            f"{full_key}={val} must be positive. "
            "Zero or negative variance is physically meaningless."
        )


def _require_positive_int(errors: List[str], section: dict, full_key: str) -> None:
    key = full_key.split(".")[-1]
    val = section.get(key)
    if val is None:
        return
    if not isinstance(val, int):
        errors.append(f"{full_key}: expected int, got {type(val).__name__}.")
    elif val <= 0:
        errors.append(f"{full_key}={val} must be a positive integer.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(raw: dict) -> None:
    errors: List[str] = []

    # Required top-level sections
    for section in ["pipeline", "camera", "detector", "tracker",
                    "kalman_filter", "visualization"]:
        if section not in raw:
            errors.append(f"Missing required config section: '{section}'.")

    if errors:
        raise ConfigValidationError(
            "Config validation failed:\n"
            + "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        )

    # Pipeline
    pipe = raw.get("pipeline", {})
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if pipe.get("log_level", "INFO") not in valid_levels:
        errors.append(
            f"pipeline.log_level={pipe.get('log_level')!r} is invalid. "
            f"Choose from: {sorted(valid_levels)}."
        )
    hz = pipe.get("target_hz", 30)
    if not isinstance(hz, (int, float)) or hz <= 0:
        errors.append(f"pipeline.target_hz must be a positive number, got {hz!r}.")

    # Camera
    cam = raw.get("camera", {})
    _require_positive_int(errors, cam, "camera.width")
    _require_positive_int(errors, cam, "camera.height")
    _require_positive_int(errors, cam, "camera.fps")

    intr = cam.get("intrinsics_path")
    if intr is not None and not Path(intr).exists():
        warnings.warn(
            f"camera.intrinsics_path={intr!r} does not exist. "
            "Metric-space position estimates will be inaccurate. "
            "Run scripts/calibrate_camera.py or set to null.",
            stacklevel=4,
        )

    # Detector
    det = raw.get("detector", {})
    _require_in_range(errors, det, "detector.confidence_threshold", 0.0, 1.0, exclusive=True)
    _require_in_range(errors, det, "detector.iou_threshold",        0.0, 1.0, exclusive=True)
    _require_positive_int(errors, det, "detector.max_detections")
    _require_positive_int(errors, det, "detector.img_size")

    img = det.get("img_size")
    if isinstance(img, int) and img % 32 != 0:
        errors.append(
            f"detector.img_size={img} must be a multiple of 32 "
            "(required by YOLO's feature pyramid network)."
        )

    cf = det.get("class_filter")
    if cf is not None:
        if not isinstance(cf, list):
            errors.append("detector.class_filter must be a list of ints or null.")
        elif not all(isinstance(c, int) and 0 <= c <= 79 for c in cf):
            errors.append(
                "detector.class_filter must contain integers in [0, 79] (COCO class IDs)."
            )

    # Tracker
    trk = raw.get("tracker", {})
    _require_in_range(errors, trk, "tracker.high_thresh",      0.0, 1.0, exclusive=True)
    _require_in_range(errors, trk, "tracker.low_thresh",       0.0, 1.0, exclusive=True)
    _require_in_range(errors, trk, "tracker.new_track_thresh", 0.0, 1.0, exclusive=True)
    _require_in_range(errors, trk, "tracker.iou_threshold",    0.0, 1.0, exclusive=True)
    _require_positive_int(errors, trk, "tracker.max_age")
    _require_positive_int(errors, trk, "tracker.min_hits")
    _require_in_range(errors, trk, "tracker.appearance_weight", 0.0, 1.0)
    _require_in_range(errors, trk, "tracker.appearance_ema",    0.0, 1.0)

    lo = trk.get("low_thresh")
    hi = trk.get("high_thresh")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo >= hi:
        errors.append(
            f"tracker.low_thresh={lo} must be strictly less than "
            f"tracker.high_thresh={hi}."
        )

    # Kalman filter
    kf = raw.get("kalman_filter", {})
    ic = kf.get("initial_covariance", {})
    pn = kf.get("process_noise", {})
    mn = kf.get("measurement_noise", {})

    for key in ("p_position", "p_size", "p_velocity"):
        _require_positive_float(errors, ic, f"kalman_filter.initial_covariance.{key}")
    for key in ("q_position", "q_size", "q_velocity", "q_vel_size"):
        _require_positive_float(errors, pn, f"kalman_filter.process_noise.{key}")
    for key in ("r_center", "r_size"):
        _require_positive_float(errors, mn, f"kalman_filter.measurement_noise.{key}")

    # Coordinate frames
    cf_raw = raw.get("coordinate_frames", {})
    for i, ext in enumerate(cf_raw.get("static_extrinsics", []) or []):
        prefix = f"coordinate_frames.static_extrinsics[{i}]"
        if "parent" not in ext or "child" not in ext:
            errors.append(f"{prefix}: 'parent' and 'child' are required.")
        R = ext.get("R")
        if R is not None:
            if not isinstance(R, list) or len(R) != 9:
                errors.append(
                    f"{prefix}.R must be a list of 9 floats (row-major 3×3)."
                )
            else:
                # Cheap orthogonality check: ||RᵀR − I||_F should be small.
                import numpy as _np
                Rm = _np.asarray(R, dtype=_np.float64).reshape(3, 3)
                err = _np.linalg.norm(Rm.T @ Rm - _np.eye(3), ord="fro")
                if err > 1e-3:
                    errors.append(
                        f"{prefix}.R is not orthonormal "
                        f"(||RᵀR − I||={err:.3e}, expected <1e-3). "
                        "Check the matrix values."
                    )
        t = ext.get("t")
        if t is not None and (not isinstance(t, list) or len(t) != 3):
            errors.append(f"{prefix}.t must be a list of 3 floats (metres).")

    # Pose estimator
    pe_raw = raw.get("pose_estimator", {})
    pe_type = pe_raw.get("type", "null")
    if pe_type not in ("null", "orbslam", "dpvo"):
        errors.append(
            f"pose_estimator.type={pe_type!r} is invalid. "
            "Supported: 'null', 'dpvo' (implemented), 'orbslam' (planned)."
        )

    # Depth estimator
    de_raw = raw.get("depth", {})
    de_type = de_raw.get("type", "depth_anything")
    if de_type not in ("null", "depth_anything", "stereo_sgbm", "raft_stereo"):
        errors.append(
            f"depth.type={de_type!r} is invalid. "
            "Supported: 'null', 'depth_anything', 'stereo_sgbm', 'raft_stereo'."
        )
    raft_iters = de_raw.get("raft_iters")
    if raft_iters is not None:
        if not isinstance(raft_iters, int) or raft_iters <= 0:
            errors.append(
                f"depth.raft_iters={raft_iters!r} must be a positive int."
            )
    num_disp = de_raw.get("sgbm_num_disparities")
    if num_disp is not None:
        if not isinstance(num_disp, int) or num_disp <= 0:
            errors.append(
                f"depth.sgbm_num_disparities={num_disp!r} must be a positive int."
            )
        elif num_disp % 16 != 0:
            errors.append(
                f"depth.sgbm_num_disparities={num_disp} must be divisible by 16 "
                "(cv2.StereoSGBM requirement)."
            )
    bs = de_raw.get("sgbm_block_size")
    if bs is not None:
        if not isinstance(bs, int) or bs <= 0:
            errors.append(
                f"depth.sgbm_block_size={bs!r} must be a positive int."
            )
        elif bs % 2 == 0:
            errors.append(
                f"depth.sgbm_block_size={bs} must be odd "
                "(cv2.StereoSGBM requirement)."
            )

    # Appearance extractor
    ap_raw = raw.get("appearance", {})
    ap_type = ap_raw.get("type", "null")
    if ap_type not in ("null", "dinov2"):
        errors.append(
            f"appearance.type={ap_type!r} is invalid. "
            "Supported: 'null', 'dinov2'."
        )

    # IMU
    imu_raw = raw.get("imu", {})
    imu_type = imu_raw.get("type", "null")
    if imu_type not in ("null", "synthetic"):
        errors.append(
            f"imu.type={imu_type!r} is invalid. "
            "Supported: 'null', 'synthetic'."
        )
    _require_positive_float(errors, imu_raw, "imu.rate_hz")
    _require_positive_float(errors, imu_raw, "imu.sigma_gyro_n")
    _require_positive_float(errors, imu_raw, "imu.sigma_accel_n")

    # VIO error-state EKF
    vio_raw = raw.get("vio", {})
    for k in ("init_position_std_m", "init_velocity_std_mps",
              "init_orientation_std_rad", "init_bias_gyro_std",
              "init_bias_accel_std", "bias_gyro_random_walk",
              "bias_accel_random_walk", "visual_position_std_m",
              "visual_orientation_std_rad"):
        _require_positive_float(errors, vio_raw, f"vio.{k}")
    g_w = vio_raw.get("gravity_w")
    if g_w is not None and (not isinstance(g_w, list) or len(g_w) != 3):
        errors.append("vio.gravity_w must be a list of 3 floats (world-frame gravity).")

    # Semantic segmentation
    sem_raw = raw.get("semantic", {})
    sem_type = sem_raw.get("type", "null")
    if sem_type not in ("null", "mask2former"):
        errors.append(
            f"semantic.type={sem_type!r} is invalid. "
            "Supported: 'null', 'mask2former'."
        )
    sem_ds = sem_raw.get("dataset", "cityscapes")
    if sem_ds not in ("cityscapes", "ade20k"):
        # Not fatal — downstream helpers gracefully fall back when the
        # dataset is unknown — but flag it so config typos surface.
        warnings.warn(
            f"semantic.dataset={sem_ds!r} is not in the built-in helper "
            "tables ('cityscapes', 'ade20k'). drivable_mask + "
            "semantic_class_stability will return empty/SEMI_STATIC for "
            "this dataset until those tables are extended.",
            stacklevel=4,
        )

    # World map
    wmap_raw = raw.get("world_map", {})
    _require_positive_float(errors, wmap_raw, "world_map.spatial_gate_m")
    sim_thr = wmap_raw.get("similarity_threshold")
    if sim_thr is not None and not (-1.0 <= float(sim_thr) <= 1.0):
        errors.append(
            f"world_map.similarity_threshold={sim_thr} must lie in [-1, 1] "
            "(cosine similarity range)."
        )
    # 0 means "disabled" for both eviction knobs; reject negatives.
    max_age = wmap_raw.get("max_age_s")
    if max_age is not None and float(max_age) < 0.0:
        errors.append(
            f"world_map.max_age_s={max_age} must be >= 0 (0 disables aging)."
        )
    max_ent = wmap_raw.get("max_entries")
    if max_ent is not None and int(max_ent) < 0:
        errors.append(
            f"world_map.max_entries={max_ent} must be >= 0 (0 disables cap)."
        )
    _require_positive_int(errors, pe_raw, "pose_estimator.stride")
    _require_positive_int(errors, pe_raw, "pose_estimator.patches_per_frame")

    # Occupancy grid
    og_raw = raw.get("occupancy_grid", {})
    _require_positive_float(errors, og_raw, "occupancy_grid.resolution_m")
    _require_positive_float(errors, og_raw, "occupancy_grid.size_x_m")
    _require_positive_float(errors, og_raw, "occupancy_grid.size_y_m")
    _require_positive_float(errors, og_raw, "occupancy_grid.default_inflation_m")
    _require_positive_float(errors, og_raw, "occupancy_grid.min_inflation_m")

    # 3D occupancy
    o3_raw = raw.get("occupancy_3d", {})
    _require_positive_float(errors, o3_raw, "occupancy_3d.resolution_m")
    _require_positive_float(errors, o3_raw, "occupancy_3d.size_x_m")
    _require_positive_float(errors, o3_raw, "occupancy_3d.size_y_m")
    _require_positive_float(errors, o3_raw, "occupancy_3d.size_z_m")
    _require_positive_float(errors, o3_raw, "occupancy_3d.default_inflation_m")
    _require_positive_float(errors, o3_raw, "occupancy_3d.min_inflation_m")

    # Room layer
    rl_raw = raw.get("room_layer", {})
    _require_positive_float(errors, rl_raw, "room_layer.erosion_m")
    _require_positive_float(errors, rl_raw, "room_layer.min_area_m2")
    _require_positive_float(errors, rl_raw, "room_layer.polygon_simplify_m")

    # Visualization
    vis = raw.get("visualization", {})
    vas = vis.get("velocity_arrow_scale", 0.5)
    if isinstance(vas, (int, float)) and vas <= 0:
        errors.append(f"visualization.velocity_arrow_scale={vas} must be positive.")

    if errors:
        raise ConfigValidationError(
            f"Config validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        )


# ---------------------------------------------------------------------------
# Build typed config from validated raw dict
# ---------------------------------------------------------------------------

def _build(raw: dict) -> PipelineConfig:
    p   = raw.get("pipeline", {})
    c   = raw.get("camera", {})
    v   = raw.get("video", {})
    s   = raw.get("synthetic_camera", {})
    d   = raw.get("detector", {})
    t   = raw.get("tracker", {})
    kf  = raw.get("kalman_filter", {})
    ic  = kf.get("initial_covariance", {})
    pn  = kf.get("process_noise", {})
    mn  = kf.get("measurement_noise", {})
    vis = raw.get("visualization", {})
    b   = raw.get("benchmark", {})
    ekf = raw.get("extended_kalman_filter", {})
    ekf_ic = ekf.get("initial_covariance", {})
    ekf_pn = ekf.get("process_noise", {})
    ekf_mn = ekf.get("measurement_noise", {})
    de = raw.get("depth", {})
    wm = raw.get("world_model", {})
    cf = raw.get("coordinate_frames", {})
    pe = raw.get("pose_estimator", {})
    og = raw.get("occupancy_grid", {})
    o3 = raw.get("occupancy_3d",   {})
    dc = raw.get("drivable_costmap", {})
    rl = raw.get("room_layer",     {})
    ap = raw.get("appearance", {})
    wmap = raw.get("world_map", {})
    hm = raw.get("health_monitor", {})
    imu = raw.get("imu", {})
    vio = raw.get("vio", {})
    sem = raw.get("semantic", {})

    static_ext_raw = cf.get("static_extrinsics", []) or []
    static_extrinsics = [
        StaticExtrinsicConfig(
            parent=str(e.get("parent", "base_link")),
            child= str(e.get("child",  "camera_frame")),
            R=     [float(x) for x in e.get(
                "R", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            )],
            t=     [float(x) for x in e.get("t", [0.0, 0.0, 0.0])],
        )
        for e in static_ext_raw
    ]

    return PipelineConfig(
        pipeline=PipelineSettings(
            name=              str(p.get("name", "robotics-perception-pipeline")),
            log_level=         str(p.get("log_level", "INFO")),
            target_hz=         float(p.get("target_hz", 30.0)),
            output_video_path= p.get("output_video_path"),
            seed=              int(p.get("seed", 42)),
        ),
        camera=CameraConfig(
            device_index=    int(c.get("device_index", 0)),
            width=           int(c.get("width", 1280)),
            height=          int(c.get("height", 720)),
            fps=             int(c.get("fps", 30)),
            intrinsics_path= c.get("intrinsics_path"),
        ),
        video=VideoConfig(
            input_path=   str(v.get("input_path", "data/test_clip.mp4")),
            output_path=  str(v.get("output_path", "data/output.mp4")),
            playback_fps= float(v["playback_fps"]) if v.get("playback_fps") else None,
            loop=         bool(v.get("loop", False)),
        ),
        synthetic_camera=SyntheticCameraConfig(
            width=       int(s.get("width", 640)),
            height=      int(s.get("height", 480)),
            num_frames=  int(s.get("num_frames", 300)),
            fps=         float(s.get("fps", 30.0)),
            num_objects= int(s.get("num_objects", 3)),
            seed=        int(s.get("seed", 42)),
        ),
        detector=DetectorConfig(
            model=                str(d.get("model", "yolov8n.pt")),
            confidence_threshold= float(d.get("confidence_threshold", 0.25)),
            iou_threshold=        float(d.get("iou_threshold", 0.45)),
            device=               str(d.get("device", "cuda:0")),
            half_precision=       bool(d.get("half_precision", True)),
            class_filter=         [int(x) for x in d["class_filter"]]
                                  if d.get("class_filter") else None,
            max_detections=       int(d.get("max_detections", 100)),
            img_size=             int(d.get("img_size", 640)),
        ),
        tracker=TrackerConfig(
            high_thresh=       float(t.get("high_thresh", 0.50)),
            low_thresh=        float(t.get("low_thresh", 0.10)),
            new_track_thresh=  float(t.get("new_track_thresh", 0.50)),
            iou_threshold=     float(t.get("iou_threshold", 0.30)),
            max_age=           int(t.get("max_age", 30)),
            min_hits=          int(t.get("min_hits", 1)),
            use_ekf=           bool(t.get("use_ekf", False)),
            use_appearance=    bool(t.get("use_appearance", False)),
            appearance_weight= float(t.get("appearance_weight", 0.25)),
            appearance_ema=    float(t.get("appearance_ema", 0.9)),
        ),
        kalman_filter=KalmanFilterConfig(
            initial_covariance=InitialCovarianceConfig(
                p_position= float(ic.get("p_position", 10.0)),
                p_size=     float(ic.get("p_size", 10.0)),
                p_velocity= float(ic.get("p_velocity", 100.0)),
            ),
            process_noise=ProcessNoiseConfig(
                q_position= float(pn.get("q_position", 1.0)),
                q_size=     float(pn.get("q_size", 1.0)),
                q_velocity= float(pn.get("q_velocity", 0.1)),
                q_vel_size= float(pn.get("q_vel_size", 0.02)),
            ),
            measurement_noise=MeasurementNoiseConfig(
                r_center= float(mn.get("r_center", 1.0)),
                r_size=   float(mn.get("r_size", 1.0)),
            ),
        ),
        extended_kalman_filter=ExtendedKalmanFilterConfig(
            initial_covariance=EKFInitialCovarianceConfig(
                p_position= float(ekf_ic.get("p_position", 10.0)),
                p_size=     float(ekf_ic.get("p_size",     10.0)),
                p_velocity= float(ekf_ic.get("p_velocity", 100.0)),
                p_omega=    float(ekf_ic.get("p_omega",    1.0)),
            ),
            process_noise=EKFProcessNoiseConfig(
                q_position= float(ekf_pn.get("q_position", 1.0)),
                q_size=     float(ekf_pn.get("q_size",     1.0)),
                q_velocity= float(ekf_pn.get("q_velocity", 0.1)),
                q_vel_size= float(ekf_pn.get("q_vel_size", 0.02)),
                q_omega=    float(ekf_pn.get("q_omega",    0.01)),
            ),
            measurement_noise=MeasurementNoiseConfig(
                r_center= float(ekf_mn.get("r_center", 1.0)),
                r_size=   float(ekf_mn.get("r_size",   1.0)),
            ),
        ),
        depth=DepthConfig(
            enabled=bool(de.get("enabled", False)),
            type=str(de.get("type", "depth_anything")),
            model=str(de.get("model", "ZoeD_N")),
            device=str(de.get("device", "cuda")),
            sgbm_min_disparity=int(de.get("sgbm_min_disparity", 0)),
            sgbm_num_disparities=int(de.get("sgbm_num_disparities", 96)),
            sgbm_block_size=int(de.get("sgbm_block_size", 7)),
            raft_repo_dir=  str(de.get("raft_repo_dir",
                                       "third_party/RAFT-Stereo")),
            raft_checkpoint=str(de.get("raft_checkpoint",
                "third_party/RAFT-Stereo/models/raftstereo-middlebury.pth")),
            raft_iters=     int(de.get("raft_iters", 16)),
        ),
        visualization=VisualizationConfig(
            rerun_enabled=           bool(vis.get("rerun_enabled", True)),
            rerun_app_id=            str(vis.get("rerun_app_id", "robotics_perception_pipeline")),
            rerun_host=              str(vis.get("rerun_host", "auto")),
            rerun_port=              int(vis.get("rerun_port", 9876)),
            rerun_save_path=         vis.get("rerun_save_path"),
            show_bboxes=             bool(vis.get("show_bboxes", True)),
            show_track_ids=          bool(vis.get("show_track_ids", True)),
            show_velocity=           bool(vis.get("show_velocity", True)),
            show_covariance_ellipse= bool(vis.get("show_covariance_ellipse", True)),
            show_nis=                bool(vis.get("show_nis", False)),
            show_stats_overlay=      bool(vis.get("show_stats_overlay", True)),
            bbox_thickness=          int(vis.get("bbox_thickness", 2)),
            velocity_arrow_scale=    float(vis.get("velocity_arrow_scale", 0.5)),
        ),
        benchmark=BenchmarkConfig(
            dataset_path= str(b.get("dataset_path", "data/MOT17")),
            split=        str(b.get("split", "train")),
            sequences=    b.get("sequences"),
            output_dir=   str(b.get("output_dir", "data/benchmark_results")),
        ),
        world_model=WorldModelConfig(
            max_history=    int(wm.get("max_history", 30)),
            lost_timeout_s= float(wm.get("lost_timeout_s", 1.0)),
            camera_frame=   str(wm.get("camera_frame", "camera_frame")),
        ),
        coordinate_frames=CoordinateFramesConfig(
            enabled=           bool(cf.get("enabled", False)),
            root_frame=        str(cf.get("root_frame", "map")),
            camera_frame=      str(cf.get("camera_frame", "camera_frame")),
            static_extrinsics= static_extrinsics,
        ),
        pose_estimator=PoseEstimatorConfig(
            type=              str(pe.get("type", "null")),
            checkpoint=        pe.get("checkpoint"),
            stride=            int(pe.get("stride", 2)),
            patches_per_frame= int(pe.get("patches_per_frame", 96)),
        ),
        occupancy_grid=OccupancyGridConfig(
            enabled=               bool(og.get("enabled", False)),
            resolution_m=          float(og.get("resolution_m", 0.05)),
            size_x_m=              float(og.get("size_x_m", 20.0)),
            size_y_m=              float(og.get("size_y_m", 20.0)),
            origin_x_m=            float(og.get("origin_x_m", -10.0)),
            origin_y_m=            float(og.get("origin_y_m", -10.0)),
            default_inflation_m=   float(og.get("default_inflation_m", 0.5)),
            min_inflation_m=       float(og.get("min_inflation_m", 0.10)),
            per_class_inflation_m={
                str(k): float(v) for k, v in
                (og.get("per_class_inflation_m") or {}).items()
            } or OccupancyGridConfig().per_class_inflation_m,
        ),
        occupancy_3d=Occupancy3DConfig(
            enabled=               bool(o3.get("enabled", False)),
            resolution_m=          float(o3.get("resolution_m", 0.10)),
            size_x_m=              float(o3.get("size_x_m", 20.0)),
            size_y_m=              float(o3.get("size_y_m", 20.0)),
            size_z_m=              float(o3.get("size_z_m", 3.0)),
            origin_x_m=            float(o3.get("origin_x_m", -10.0)),
            origin_y_m=            float(o3.get("origin_y_m", -10.0)),
            origin_z_m=            float(o3.get("origin_z_m", -0.5)),
            default_inflation_m=   float(o3.get("default_inflation_m", 0.5)),
            min_inflation_m=       float(o3.get("min_inflation_m", 0.10)),
            per_class_inflation_m={
                str(k): float(v) for k, v in
                (o3.get("per_class_inflation_m") or {}).items()
            } or Occupancy3DConfig().per_class_inflation_m,
        ),
        drivable_costmap=DrivableCostmapConfig(
            enabled=    bool(dc.get("enabled", False)),
            use_depth=  bool(dc.get("use_depth", True)),
            z_ground_m= float(dc.get("z_ground_m", 0.0)),
        ),
        room_layer=RoomLayerConfigCfg(
            enabled=            bool(rl.get("enabled", False)),
            erosion_m=          float(rl.get("erosion_m",          0.45)),
            min_area_m2=        float(rl.get("min_area_m2",        1.0)),
            polygon_simplify_m= float(rl.get("polygon_simplify_m", 0.05)),
        ),
        appearance=AppearanceConfig(
            type=   str(ap.get("type", "null")),
            model=  str(ap.get("model", "facebook/dinov2-small")),
            device= str(ap.get("device", "cuda")),
        ),
        world_map=WorldMapConfig(
            enabled=              bool(wmap.get("enabled", False)),
            spatial_gate_m=       float(wmap.get("spatial_gate_m", 1.5)),
            similarity_threshold= float(wmap.get("similarity_threshold", 0.75)),
            allow_spatial_only=   bool(wmap.get("allow_spatial_only", True)),
            max_age_s=            float(wmap.get("max_age_s", 0.0)),
            max_entries=          int(wmap.get("max_entries", 0)),
        ),
        health_monitor=HealthMonitorConfig(
            enabled=           bool(hm.get("enabled", True)),
            warn_after=        int(hm.get("warn_after", 3)),
            error_after=       int(hm.get("error_after", 30)),
            stale_after_s=     float(hm.get("stale_after_s", 5.0)),
            window=            int(hm.get("window", 60)),
            default_budget_ms= float(hm.get("default_budget_ms", 50.0)),
            log_period_s=      float(hm.get("log_period_s", 5.0)),
            stage_budgets_ms={
                str(k): float(v) for k, v in
                (hm.get("stage_budgets_ms") or {}).items()
            } or HealthMonitorConfig().stage_budgets_ms,
        ),
        imu=IMUConfig(
            type=             str(imu.get("type", "null")),
            frame_id=         str(imu.get("frame_id", "imu_link")),
            rate_hz=          float(imu.get("rate_hz", 200.0)),
            sigma_gyro_n=     float(imu.get("sigma_gyro_n",  1.7e-4)),
            sigma_accel_n=    float(imu.get("sigma_accel_n", 2.0e-3)),
            synthetic_motion= str(imu.get("synthetic_motion",
                                          "stationary_with_gravity")),
            synthetic_noise_std_accel=
                              float(imu.get("synthetic_noise_std_accel", 0.0)),
            synthetic_noise_std_gyro=
                              float(imu.get("synthetic_noise_std_gyro", 0.0)),
            synthetic_seed=   int(imu.get("synthetic_seed", 0)),
        ),
        vio=VIOConfig(
            enabled=                  bool(vio.get("enabled", False)),
            init_position_std_m=      float(vio.get("init_position_std_m",      0.10)),
            init_velocity_std_mps=    float(vio.get("init_velocity_std_mps",    0.10)),
            init_orientation_std_rad= float(vio.get("init_orientation_std_rad", 0.05)),
            init_bias_gyro_std=       float(vio.get("init_bias_gyro_std",       1e-3)),
            init_bias_accel_std=      float(vio.get("init_bias_accel_std",      1e-2)),
            bias_gyro_random_walk=    float(vio.get("bias_gyro_random_walk",    1e-5)),
            bias_accel_random_walk=   float(vio.get("bias_accel_random_walk",   1e-4)),
            visual_position_std_m=    float(vio.get("visual_position_std_m",    0.05)),
            visual_orientation_std_rad=
                                      float(vio.get("visual_orientation_std_rad", 0.02)),
            gravity_w=                [float(x) for x in vio.get(
                "gravity_w", [0.0, 0.0, -9.81]
            )],
        ),
        semantic=SemanticConfig(
            enabled= bool(sem.get("enabled", False)),
            type=    str(sem.get("type",    "null")),
            model=   str(sem.get("model",
                                 "facebook/mask2former-swin-tiny-cityscapes-semantic")),
            device=  str(sem.get("device",  "cuda")),
            dataset= str(sem.get("dataset", "cityscapes")),
        ),
    )
