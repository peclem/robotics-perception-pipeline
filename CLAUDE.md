# Robotics Perception Pipeline — agent instructions

Modular camera-based perception stack for mobile robotics. Detection,
multi-object tracking, Kalman/EKF state estimation, monocular depth,
probabilistic world model. Designed as a robotics-ready perception layer.

For the full architecture, benchmark results, and gap analysis, read
README.md before answering architecture questions.

## Commands

    # Run tests (must stay green)
    python3 -m pytest tests/ -m "not integration" -v

    # Run pipeline
    RERUN_ENABLED=false python3 launch.py --source synthetic
    RERUN_ENABLED=false python3 launch.py --source video --input data/clip.mp4

    # MOT17 benchmark
    python3 scripts/benchmark.py --dataset data/MOT17 --split train --out data/results

    # Detector fine-tuning
    python3 scripts/mot17_to_yolo.py --mot17 data/MOT17 --out data/mot17_yolo
    python3 scripts/train_detector.py --data data/mot17_yolo/mot17.yaml

## Architecture rules

- Every component lives behind a Python ABC with a Null/Synthetic fallback.
- Adding a new backend (camera, detector, depth, pose) must require zero
  changes outside its own file plus the config.
- Config goes through `perception/config_loader.py` PipelineConfig dataclasses.
  No new YAML key without a typed config field and validation.
- CameraFrame is the canonical sensor DTO. Do not pass raw numpy arrays
  between modules.

## Code style

- Use type hints on all public functions and dataclass fields.
- Numpy arrays specify dtype=np.float64 unless there's a reason otherwise.
- Kalman filter math uses Joseph form for the update step.
- Tests use SyntheticCamera or synthetic data — never real hardware in CI.
- All timestamps are `time.monotonic()` floats in seconds.

## What is NOT validated yet

These features exist but lack benchmarks. Do not claim accuracy numbers
in code comments, docs, or commits unless they come from a benchmark run:

- Camera motion compensation effectiveness (no moving-camera benchmark)
- Depth Anything V2 accuracy on this hardware (no TUM RGB-D evaluation)
- EKF vs KF comparative benefit (no comparison benchmark)
- Fine-tuned detector generalisation outside MOT17

## What does NOT exist yet

Do not reference these as if they were implemented:

- ros2_nodes/ — directory does not exist, no ROS2 integration
- Real ego-pose estimation — only NullPoseEstimator scaffold exists
- Static occupancy grid
- Hardware timestamp synchronisation
- Watchdog / health monitor
- ReID appearance features
- Stereo depth, IMU fusion, VIO — Meta glasses integration not started

## Implementation directive

For every new component, implement the SOTA approach that fits the
target hardware (RTX 4070Ti 12GB, Ryzen 7 7700, WSL2 Ubuntu 22.04).
"Fits" means: runs at real-time-relevant Hz, fits in 12 GB VRAM
alongside the existing detector + depth model, and has a maintained
reference implementation. Prefer recent (≤2 years) methods with
published weights over classical baselines unless the classical
option is genuinely competitive on this hardware.

Justify the SOTA choice in the commit message (one line: method name,
why it wins on this hardware vs the obvious alternative).

## Roadmap

### Phase 1 — Validate the unvalidated claims (DEFERRED)

Hold for later. CMC on DanceTrack, Depth Anything V2 on TUM RGB-D,
EKF vs KF on MOT17, fine-tuned detector cross-domain eval.

### Phase 2 — Close the coordinate-frame gap

Recent commits added the `CameraPose` / `PoseEstimator` scaffold and
`ObjectState.position_world`. Only `NullPoseEstimator` exists; this
phase makes it real.

1. Real ego-pose estimator (SOTA monocular VIO/SLAM for 12 GB VRAM)
2. tf2-style transform tree: `camera → base_link → odom → map`
3. World-frame `SceneGraph.query_nearby` with map-frame input

Exit criteria: synthetic moving-camera scenario produces stable
world-frame `ObjectState.position_world` values.

### Phase 3 — ROS2 integration (DONE 2026-05-17)

Built 5 adapter nodes in `ros2_ws/src/robotics_perception_ros2/`:
camera_publisher, detection, tracking, pose, scene_graph. Launch
file orchestrates the full graph. Throughput ~6 Hz at 1280×720
(vs 30 Hz standalone); IPC + multi-process CUDA overhead documented
in README's "Honest performance notes" section. Phase 4 should
address via image_transport + intra-process composition.

### Phase 4 — Production robustness

4. Watchdog / health monitor with per-module latency budgets and
   graceful degradation
5. Hardware timestamp sync (prerequisite for any multi-sensor work)
6. ~~Static occupancy grid~~ → done as **dynamic obstacle grid**
   (2026-05-17). nav_msgs/OccupancyGrid via OccupancyGridBuilder +
   occupancy_grid_node. Static layer needs SLAM and stays open.
7. ReID appearance features behind an `AppearanceExtractor` ABC,
   wired into the existing cosine cost slot in
   `tracking/association.py`. SOTA backbone for 12 GB VRAM.

### Phase 5 — Multi-sensor fusion (Meta glasses direction)

8. IMU pre-integration (error-state EKF, fuse with existing CTR EKF)
9. Stereo depth as drop-in `DepthEstimator` (SOTA stereo network)
10. Visual-inertial SLAM (replaces Phase 2 ego-pose if hardware permits)
11. Semantic segmentation for drivable-surface and class priors

### Cross-cutting (every phase)

- Tests stay green
- Conventional Commits
- No accuracy claim without a benchmark
- New backend = new ABC implementation + config field, zero
  cascading changes

## Required for every change

- Tests stay green: `python3 -m pytest tests/ -m "not integration" -v`
- New modules ship with unit tests, no exceptions
- Conventional Commits: feat(scope), fix(scope), docs(scope), refactor(scope)
- Never claim performance improvements without a benchmark result

## File map

    perception/         camera_interface, detector, depth_estimator,
                        pose_estimator, config_loader
    tracking/           track, tracker, association, motion_compensation
    state_estimation/   kalman_filter, extended_kf, filter_utils
    world_model/        scene_graph, object_state
    visualization/      rerun_logger, debug_vis
    scripts/            calibrate_camera, benchmark, mot17_to_yolo,
                        train_detector
    tests/              248 unit tests, hardware-free
    config/default.yaml master config
    README.md           architecture diagram, benchmark results
