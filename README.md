# Robotics Perception Pipeline

![CI](https://github.com/peclem/robotics-perception-pipeline/actions/workflows/ci.yml/badge.svg)

A modular, robotics-grade multi-object tracking system built on a standard camera,
designed for integration into real robotics hardware.

---

## Problem Statement

How does a robot build a probabilistic model of its dynamic environment from raw sensor data?

This system answers that through a full perception stack:
raw frames → object detection → multi-object tracking → Kalman-filtered state estimation → probabilistic world model.

Every component is designed with the same constraints as production robotics systems:
typed interfaces, uncertainty-aware outputs, modular architecture, and quantitative validation.

---

## System Architecture

    Camera Input
        │
        ▼
    Detection (YOLOv8n)
        │  bbox + confidence
        ▼
    ByteTrack Two-Stage Association
        │  track ID + detection
        ▼
    Kalman Filter / EKF State Estimation
        │  [cx, cy, w, h, vx, vy, vw, vh] + covariance
        ▼
    World Model / Scene Graph
        │  ObjectState: position, covariance, velocity, trajectory
        ▼
    Visualization (Rerun.io + OpenCV)

Each module exposes a clean Python ABC. Swapping any component — detector model,
filter type, camera backend — requires changing one class with zero cascading changes.

---

## Key Features

**Uncertainty-aware state estimation** — every track output includes a full covariance
matrix. The 2-sigma position ellipse shrinks as the filter converges and grows during
occlusion. Visually demonstrable in Rerun.io.

**Two-stage ByteTrack association** — high-confidence detections for primary assignment,
low-confidence detections for occlusion rescue. Tracks survive partial occlusion without
ID switches.

**Extended Kalman Filter with constant turn rate model** — nonlinear CTR motion model
with analytical Jacobians validated against numerical finite differences.
NEES diagnostic available for simulation consistency testing.

**NIS filter consistency validation** — Normalised Innovation Squared logged per track.
A consistent filter produces NIS ~ chi-squared(4) with mean 4.0.
Standard aerospace/robotics filter validation diagnostic.

**Probabilistic world model** — scene graph with per-object trajectory history,
query_nearby(position, radius) spatial interface, and Mahalanobis-distance queries.
Designed as the interface a path planner or costmap generator would call.

**Modular, interface-driven architecture** — all modules depend on ABCs.
ROS2 node wrappers are thin adapters over the core logic.

**Configuration-driven** — all parameters live in config/default.yaml.
Environment variable overrides for CI and deployment. No magic numbers in source.

---

## Technical Stack

| Component        | Library                | Robotics Rationale                                      |
|------------------|------------------------|---------------------------------------------------------|
| Detection        | YOLOv8n (Ultralytics)  | ONNX/TensorRT exportable for embedded deployment        |
| Tracking         | ByteTrack              | Two-stage occlusion recovery, production-grade          |
| State estimation | NumPy KF/EKF (scratch) | Full transparency — every matrix multiply explainable   |
| Config           | PyYAML + dataclasses   | Mirrors ROS2 parameter server pattern                   |
| Visualization    | Rerun.io + OpenCV      | Robotics-native: 3D, time-series, transforms            |
| Testing          | pytest + filterpy      | Deterministic, hardware-free, CI-ready                  |

---

## Robotics Relevance

| This project                      | Real robotics system                              |
|-----------------------------------|---------------------------------------------------|
| KalmanFilter / ExtendedKF         | EKF in autonomous vehicle localisation            |
| ByteTracker Hungarian assignment  | Multi-object SLAM data association                |
| SceneGraph.query_nearby()         | Dynamic obstacle query for motion planner         |
| CameraFrame typed DTO             | sensor_msgs/Image + sensor_msgs/CameraInfo        |
| config/default.yaml               | ROS2 parameter server                             |
| ros2_nodes/ stubs                 | Production ROS2 deployment adapters               |
| NIS diagnostic                    | Standard aerospace filter consistency test        |
| NEES diagnostic                   | Simulation-based filter validation                |
| Covariance ellipse                | Uncertainty propagation in probabilistic roadmaps |
| Track history buffer              | Trajectory prediction input for planners          |

The KF covariance output is a first-class deliverable — not an internal variable.
Every confirmed track exposes its full 8x8 (or 9x9 EKF) covariance matrix.
A planner consuming this system receives a probability distribution over object states.

---

## Benchmark Results

### MOT17 — train split (21 sequences, YOLOv8n fp16, RTX 4070Ti)

| Metric | Value  |
|--------|--------|
| MOTA   | 23.6%  |
| MOTP   | 75.9%  |
| FP     | 45228  |
| FN     | 211167 |
| IDSW   | 2463   |
| Hz     | 132.4  |

MOTP 75.9% confirms that when objects are detected, localisation and association
quality are consistent across all 21 sequences. The primary bottleneck is detector
recall — YOLOv8n is a 6MB model trained on COCO without domain fine-tuning.
IDSW/frame = 0.53 demonstrates stable track identity under the two-stage
ByteTrack association. Throughput of 132.4 Hz provides 4x real-time headroom at 30fps.

---

## Quick Start

    git clone https://github.com/peclem/robotics-perception-pipeline
    cd robotics-perception-pipeline

    python3.10 -m venv .venv && source .venv/bin/activate

    pip install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu121
    pip install ultralytics opencv-python-headless filterpy \
      scipy numpy pyyaml pytest rerun-sdk
    pip install -e .

    # Run on synthetic camera (no hardware needed)
    RERUN_ENABLED=false python3 launch.py --source synthetic

    # Run on a video file
    RERUN_ENABLED=false python3 launch.py --source video --input data/your_video.mp4

---

## Run Tests

    # All unit tests (no GPU, no hardware required)
    python3 -m pytest tests/ -m "not integration" -v

    # Integration tests (requires CUDA GPU)
    python3 -m pytest tests/ -m integration -v

248 unit tests across 6 modules. All tests use SyntheticCamera or synthetic data.
No hardware required for CI.

---

## Rerun.io Visualization

    # 1. Download Rerun viewer:
    #    https://rerun.io/docs/getting-started/installing-viewer

    # 2. Open the viewer (it waits for connections)

    # 3. Run the pipeline — it auto-connects via TCP
    python3 launch.py --source video --input data/your_video.mp4

    # Or save a recording for offline review
    python3 launch.py --source synthetic --rerun-save data/recording.rrd

The viewer shows camera image, detection boxes, track boxes colour-coded by ID,
2-sigma covariance ellipses, velocity arrows, and FPS/latency time-series plots.

---

## Camera Calibration

    # Live webcam calibration (requires 9x6 checkerboard)
    python3 scripts/calibrate_camera.py \
      --device 0 --rows 9 --cols 6 \
      --out config/camera_intrinsics.yaml

    # From a video of a checkerboard
    python3 scripts/calibrate_camera.py \
      --input data/calib.mp4 \
      --out config/camera_intrinsics.yaml

Calibration enables metric-space position estimates from the Kalman filter.

---

## Benchmark

    # Single sequence
    python3 scripts/benchmark.py \
      --dataset data/MOT17 --split train \
      --sequences MOT17-04-FRCNN --out data/mot17_results

    # Full train split
    python3 scripts/benchmark.py \
      --dataset data/MOT17 --split train --out data/mot17_results

---

## Project Structure

    perception/          Camera interface, YOLOv8 detector, config loader
    tracking/            ByteTrack, Hungarian assignment, Track dataclass
    state_estimation/    Kalman Filter, Extended KF (CTR), NIS/NEES diagnostics
    world_model/         Scene graph, ObjectState, spatial queries
    visualization/       Rerun.io logger, OpenCV annotator
    ros2_nodes/          ROS2 adapter stubs (Phase 3)
    scripts/             Calibration, benchmark
    tests/               248 unit tests, all hardware-free
    config/              YAML configuration — all parameters externalised
    docs/                Benchmark results

---

## Future Improvements

### Robot Integration (Phase 3)

ROS2 Humble node wrappers for each module are stubbed in ros2_nodes/.
Each adapter is approximately 50 lines — the core module logic is framework-agnostic.
Upgrade path: WebcamCamera becomes a ROS2ImageSubscriber with the same CameraFrame output.

### SLAM Extension

SceneGraph coordinate frame is designed to accept ORB-SLAM3 map output.
Replace pixel coordinates with metric 3D positions from the SLAM map frame.

### Planning / Control Interface

SceneGraph.query_nearby(robot_position, radius) is the planner interface.
Connect to Nav2 costmap or a custom potential field planner:
dynamic obstacles → inflated costmap → path replanning.

### Multi-Sensor Fusion

ExtendedKalmanFilter is ready for IMU fusion via error-state EKF.
Fuse angular velocity into the turn rate state omega in the CTR motion model.

### Detector Fine-tuning

The Detector ABC means swapping yolov8n.pt for a domain-specific checkpoint
is a one-class change with zero other modifications required.

---

## Research Foundation

| Paper                                        | Used in                                      |
|----------------------------------------------|----------------------------------------------|
| Kalman (1960) — optimal linear filter        | state_estimation/kalman_filter.py            |
| Welch and Bishop — KF tutorial               | state_estimation/kalman_filter.py            |
| Bewley et al. SORT (2016)                    | tracking/tracker.py, tracking/association.py |
| Zhang et al. ByteTrack (2022)                | tracking/tracker.py                          |
| Wan and van der Merwe — UKF (2000)           | EKF upgrade path                             |
| Thrun, Burgard, Fox — Probabilistic Robotics | state_estimation/, world_model/              |
| Campos et al. ORB-SLAM3 (2021)               | Phase 3 integration target                   |

---

## Hardware

- CPU: AMD Ryzen 7 7700
- GPU: NVIDIA RTX 4070 Ti
- OS: Ubuntu 22.04 (WSL2)
- Python: 3.10
