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
    high_thresh:      float = 0.50
    low_thresh:       float = 0.10
    new_track_thresh: float = 0.50
    iou_threshold:    float = 0.30
    max_age:          int   = 30
    min_hits:         int   = 1

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
class VisualizationConfig:
    rerun_enabled:           bool  = True
    rerun_app_id:            str   = "robotics_perception_pipeline"
    show_bboxes:             bool  = True
    show_track_ids:          bool  = True
    show_velocity:           bool  = True
    show_covariance_ellipse: bool  = True
    show_nis:                bool  = False
    show_stats_overlay:      bool  = True
    bbox_thickness:          int   = 2
    velocity_arrow_scale:    float = 0.5

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
class PipelineConfig:
    """
    Fully typed pipeline configuration.
    Construct only via load_config() — never directly.
    """
    pipeline:         PipelineSettings      = field(default_factory=PipelineSettings)
    camera:           CameraConfig          = field(default_factory=CameraConfig)
    video:            VideoConfig           = field(default_factory=VideoConfig)
    synthetic_camera: SyntheticCameraConfig = field(default_factory=SyntheticCameraConfig)
    detector:         DetectorConfig        = field(default_factory=DetectorConfig)
    tracker:          TrackerConfig         = field(default_factory=TrackerConfig)
    kalman_filter:    KalmanFilterConfig    = field(default_factory=KalmanFilterConfig)
    visualization:    VisualizationConfig   = field(default_factory=VisualizationConfig)
    benchmark:        BenchmarkConfig       = field(default_factory=BenchmarkConfig)

    def as_dict(self) -> dict:
        """
        Full nested dict — structure matches the YAML exactly.
        Pass to any step 1-5 module:
            ByteTracker(cfg.as_dict())
            KalmanFilter(state, cfg.as_dict())
            SyntheticCamera(cfg.as_dict(), num_frames=90)
        """
        return {
            "pipeline":         self.pipeline.as_dict(),
            "camera":           self.camera.as_dict(),
            "video":            self.video.as_dict(),
            "synthetic_camera": self.synthetic_camera.as_dict(),
            "detector":         self.detector.as_dict(),
            "tracker":          self.tracker.as_dict(),
            "kalman_filter":    self.kalman_filter.as_dict(),
            "visualization":    self.visualization.as_dict(),
            "benchmark":        self.benchmark.as_dict(),
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
            high_thresh=      float(t.get("high_thresh", 0.50)),
            low_thresh=       float(t.get("low_thresh", 0.10)),
            new_track_thresh= float(t.get("new_track_thresh", 0.50)),
            iou_threshold=    float(t.get("iou_threshold", 0.30)),
            max_age=          int(t.get("max_age", 30)),
            min_hits=         int(t.get("min_hits", 1)),
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
        visualization=VisualizationConfig(
            rerun_enabled=           bool(vis.get("rerun_enabled", True)),
            rerun_app_id=            str(vis.get("rerun_app_id", "robotics_perception_pipeline")),
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
    )
