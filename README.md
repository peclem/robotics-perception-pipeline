# Robotics Perception Pipeline

![CI](https://github.com/peclem/robotics-perception-pipeline/actions/workflows/ci.yml/badge.svg)

A modular perception stack for mobile robotics. Implements camera-based
multi-object detection, tracking, and state estimation, with a probabilistic
world model designed as a perception layer for ROS2 navigation systems.

---

## System architecture

A complete robotics perception architecture. Implemented components are
marked. Unimplemented components and their integration points are identified.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  SENSING                                                         ║
    ║                                                                  ║
    ║  [✓] Monocular RGB        CameraInterface ABC                    ║
    ║  [✓] Camera calibration   scripts/calibrate_camera.py           ║
    ║  [ ] Stereo camera        CameraInterface ABC — drop-in backend  ║
    ║  [ ] IMU                  error-state EKF input                  ║
    ║  [ ] LiDAR                point cloud processing                 ║
    ╚══════════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║  SENSOR PREPROCESSING                                            ║
    ║                                                                  ║
    ║  [✓] Distortion correction    load_intrinsics() + OpenCV        ║
    ║  [✓] Frame timestamping       time.monotonic(), CameraFrame     ║
    ║  [ ] Hardware timestamp sync  multi-sensor clock alignment      ║
    ╚══════════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║  PERCEPTION                                                      ║
    ║                                                                  ║
    ║  [✓] Object detection         YOLOv8n fine-tuned on MOT17       ║
    ║  [✓] Multi-object tracking    ByteTrack two-stage association    ║
    ║  [✓] KF state estimation      Joseph form, 8D state, NIS        ║
    ║  [✓] EKF — constant turn rate 9D state + ω, analytical Jac.    ║
    ║  [✓] Camera motion comp.      LK optical flow + affine RANSAC   ║
    ║  [✓] Monocular depth          Depth Anything V2, metric         ║
    ║  [ ] ReID embeddings          OSNet, IoU + cosine cost          ║
    ║  [ ] Stereo depth             StereoDepthEstimator ABC          ║
    ║  [ ] IMU pre-integration      error-state EKF fusion            ║
    ║  [ ] Semantic segmentation    —                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║  COORDINATE FRAMES                                               ║
    ║                                                                  ║
    ║  [ ] camera_frame → base_link → odom → map transform tree      ║
    ║  [ ] Ego-pose estimation      monocular VIO or stereo + IMU    ║
    ║                                                                  ║
    ║  Object positions are currently in the camera frame.            ║
    ║  A planning layer requires positions in a fixed world frame.    ║
    ║  Integration point: SceneGraph.update() accepts a camera pose   ║
    ║  parameter; all ObjectState positions are then expressed in     ║
    ║  the map frame automatically.                                   ║
    ║                                                                  ║
    ║  Camera motion compensation (implemented) corrects 2D track     ║
    ║  states for ego-motion but does not produce a full SE(3) pose.  ║
    ╚══════════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║  WORLD MODEL                                                     ║
    ║                                                                  ║
    ║  [✓] Dynamic object tracking  SceneGraph, ObjectState           ║
    ║  [✓] Per-object covariance    full 8x8 / 9x9 matrix            ║
    ║  [✓] Trajectory history       bounded KFSnapshot deque          ║
    ║  [✓] Metric 3D positions      position_3d from depth            ║
    ║  [✓] Spatial queries          query_nearby(pos, radius)         ║
    ║  [✓] Uncertainty queries      Mahalanobis distance              ║
    ║  [ ] Static occupancy grid    prerequisite for Nav2 costmap     ║
    ║  [ ] Persistent map           SLAM integration                  ║
    ╚══════════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║  PLANNING INTERFACE                                              ║
    ║                                                                  ║
    ║  SceneGraph.query_nearby(robot_position, radius)                 ║
    ║      → List[(distance, ObjectState)] sorted, with covariance    ║
    ║      → costmap inflation by 2σ ellipse per object               ║
    ║      → local planner input (DWA / TEB / MPC)                   ║
    ║                                                                  ║
    ║  [ ] Global planner           Nav2 / RRT / A*                   ║
    ║  [ ] Local planner            DWA / TEB / MPC                   ║
    ║  [ ] Behaviour tree           BehaviorTree.CPP                  ║
    ╚══════════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║  CONTROL                                                         ║
    ║                                                                  ║
    ║  [ ] ros2_control             hardware interface layer           ║
    ║  [ ] PID / LQR / MPC          motor control loops               ║
    ║  [ ] Safety monitor           watchdog, graceful degradation     ║
    ╚══════════════════════════════════════════════════════════════════╝

    ROS2 adapter nodes: ros2_ws/src/robotics_perception_ros2/
    wraps each implemented module in thin sensor_msgs / vision_msgs /
    nav_msgs interfaces. Launch with
        ros2 launch robotics_perception_ros2 perception_pipeline.launch.py
    See "ROS2 integration" section below for topic graph + setup.

---

## Data flow

    CameraFrame (image, timestamp, intrinsics)
        │
        ├──▶ YOLOv8n ──▶ Detection[] (bbox, confidence, class)
        │         │
        │         └──▶ Depth Anything V2 ──▶ depth_m, position_3d
        │
        ├──▶ CameraMotionCompensator
        │         LK optical flow on background keypoints
        │         Affine RANSAC → homography H
        │         H⁻¹ applied to track states before association
        │
        ▼
    ByteTracker
        Stage 1: D_high ↔ all tracks    (Hungarian, 1-IoU cost)
        Stage 2: D_low  ↔ lost tracks   (occlusion rescue)
        Per track:
            KF.predict(dt)   →  x̂ = F x,  P = F P Fᵀ + Q
            KF.update(z)     →  x = x̂ + K(z − Hx̂),  Joseph form
            NIS = yᵀ S⁻¹ y  ~  χ²(4),  bounds [0.711, 9.488]
        │
        ▼
    SceneGraph
        ObjectState per confirmed track:
            position    (cx, cy)  pixels — camera frame
            position_3d (X, Y, Z) metres — camera frame [¹]
            covariance  8×8 full matrix
            velocity    (vx, vy, vw, vh) pixels/s
            trajectory  bounded KFSnapshot history
        │
        └──▶ query_nearby(position, radius) → planner interface

    [¹] Camera-frame metric positions. World-frame positions require
        ego-pose estimation and a coordinate frame transform tree.

---

## Benchmark results

Evaluated on MOT17 train split (21 sequences) on RTX 4070Ti.
Public detection track — zero-shot and domain-fine-tuned variants.

    Detector                  MOTA    MOTP    FP      FN       IDSW   Hz
    ──────────────────────────────────────────────────────────────────────
    YOLOv8n  zero-shot        23.6%   75.9%   45228   211167   2463   132
    YOLOv8n  fine-tuned †     44.7%   76.5%   23160   142461   3651   110

    † Fine-tuned: 50 epochs, imgsz=1280, trained on 5 MOT17 sequences.
      Validation sequences (MOT17-09, MOT17-11, unseen during training):
      MOTA 51.6%  MOTP 77.7%  Hz 118

The gap to the published ByteTrack result (MOTA 80.3%) is explained by
detector scale and training data. The published result uses YOLOX-X
(94M parameters, private detections fine-tuned on MOT17 test sequences).
This implementation uses YOLOv8n (3.2M parameters, public detections).
The tracker association and Kalman filter are architecturally equivalent.

FP dropped 49% and FN dropped 33% with fine-tuning, confirming that the
tracker association is not the performance bottleneck.

---

## Engineering notes

**Filter consistency.** The Kalman filter is validated beyond prediction
accuracy. NIS (Normalised Innovation Squared) is computed per track and
per frame. A consistent filter produces NIS ~ χ²(4) with mean ≈ 4.0 —
the standard aerospace and robotics filter validation diagnostic from
Bar-Shalom et al. (2001). The NEES diagnostic is also implemented for
simulation-based validation when ground truth is available.

**Uncertainty as a first-class output.** Every confirmed track exposes
its full 8×8 (or 9×9 EKF) covariance matrix. The 2σ position ellipse
is rendered in Rerun.io, shrinking as the filter converges and growing
during predict-only periods (occlusion). A downstream planner can use
the covariance to inflate a costmap proportionally to position uncertainty
rather than using a fixed inflation radius.

**Camera motion compensation.** The LK optical flow + affine RANSAC
approach follows BoT-SORT section 3.2. Background keypoints are detected
and explicitly masked inside object bounding boxes — a moving object
would otherwise corrupt the homography estimate. The inverse homography
is applied to track states before association, removing apparent motion
caused by camera ego-motion.

**Modular architecture.** Every component implements a Python ABC.
Swapping any module — detector checkpoint, filter variant, camera
backend, depth estimator — requires changing one class with zero
cascading changes downstream. The fine-tuning experiment demonstrates
this: the only change between zero-shot and fine-tuned evaluations is
the model path in config/default.yaml.

---

## Technical stack

    Component                Library / method           Latency (4070Ti)
    ───────────────────────────────────────────────────────────────────
    Object detection         YOLOv8n (Ultralytics)      ~5 ms
    Multi-object tracking    ByteTrack                  ~0.5 ms
    State estimation         KF / EKF (NumPy)           ~0.1 ms
    Depth estimation         Depth Anything V2           ~10 ms
    Camera motion comp.      OpenCV LK + RANSAC         ~1 ms
    World model              Custom scene graph          ~0.2 ms
    Total (depth disabled)                              ~7 ms  (143 Hz)
    Total (depth enabled)                               ~17 ms  (58 Hz)

---

## Quick start

    git clone https://github.com/peclem/robotics-perception-pipeline
    cd robotics-perception-pipeline

    python3.10 -m venv .venv && source .venv/bin/activate

    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
    pip install ultralytics opencv-python-headless filterpy \
        scipy numpy pyyaml pytest rerun-sdk transformers accelerate
    pip install -e .

    RERUN_ENABLED=false python3 launch.py --source synthetic
    RERUN_ENABLED=false python3 launch.py --source video --input data/clip.mp4

---

## ROS2 integration

Standalone Python pipeline AND a parallel ROS2 graph. Same modules
behind both — the ROS2 layer is an adapter, not a port.

### Topic graph

    camera_publisher_node ──┬──▶ /perception/image_raw    (sensor_msgs/Image)
                            ├──▶ /perception/camera_info  (sensor_msgs/CameraInfo)

    /perception/image_raw  ──▶ detection_node  ──▶ /perception/detections
                                                   (vision_msgs/Detection2DArray)

    /perception/detections + image ──▶ tracking_node  ──▶ /perception/tracks
                                                          (vision_msgs/Detection2DArray
                                                           with track IDs)

    /perception/image_raw  ──▶ pose_node  ──┬──▶ /perception/odom (nav_msgs/Odometry)
                                            └──▶ /tf  (map → camera_frame)

    /perception/tracks + /perception/odom ──▶ scene_graph_node ──▶ /perception/scene
                                                                   (vision_msgs/Detection3DArray
                                                                    with covariance)

### Setup

    # 1. Install ROS2 Humble (see https://docs.ros.org/en/humble/Installation.html)
    # 2. Source ROS2 + project venv together (ORDER MATTERS — venv first):
    source .venv/bin/activate
    source /opt/ros/humble/setup.bash

    # 3. Build the workspace
    cd ros2_ws
    colcon build --packages-select robotics_perception_ros2

    # 4. Source the install overlay
    source install/setup.bash

### Launch

    ros2 launch robotics_perception_ros2 perception_pipeline.launch.py \
        source:=video \
        video_path:=/abs/path/to/clip.mp4 \
        config_path:=/abs/path/to/config/default.yaml

    # Verify in another shell:
    ros2 topic list                            # 7 /perception/* topics
    ros2 topic hz /perception/scene            # ~6 Hz on this hardware
    ros2 topic echo /perception/scene --once   # single Detection3DArray

### Honest performance notes

The standalone Python pipeline runs end-to-end at ~30 Hz. The ROS2
graph runs at ~6 Hz at 1280×720 due to:

- DDS serialisation of 1280×720 BGR images (~2.7 MB / msg / topic)
- Per-process CUDA context overhead (detection + pose both load torch)
- Synchronous callback chains across 5 processes

The graph is a faithful adapter, not a tuned production deployment.
Knock-on optimisations (image_transport compression, intra-process
composition via component containers, shared CUDA context) are
deferred to Phase 4 production robustness work.

### Lossiness across the ROS boundary

`scene_graph_node` reconstructs minimal Track state from
`Detection2DArray` because ROS standard messages don't carry the full
8×8 KF covariance. Position covariance survives via
`Detection3D.results[0].pose.covariance` (a 6×6 block). Velocity
covariance is approximated. For a production deployment, define a
custom message carrying the full filter state.

### Backend selection

Both the standalone and ROS2 paths read the same `config/default.yaml`.
`pose_estimator.type: dpvo` switches the pose source from
`NullPoseEstimator` to the real `DPVOPoseEstimator`. Build the DPVO
extension first (see `third_party/DPVO/`).

---

## Tests

    python3 -m pytest tests/ -m "not integration" -v

248 unit tests across detection, tracking, state estimation, world model,
and visualisation. All tests use SyntheticCamera or synthetic data —
no hardware required. Integration tests (GPU, real model) are marked
and excluded from CI.

---

## Configuration

All parameters are externalised in config/default.yaml.

    detector:
        model: "runs/detect/mot17_finetune/weights/best.pt"
        confidence_threshold: 0.25

    tracker:
        use_ekf: false      # true = ExtendedKalmanFilter (CTR model)
        use_cmc: false      # true = camera motion compensation

    depth:
        enabled: false      # true = Depth Anything V2 metric depth
        model: "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

Environment variable overrides: DEVICE=cpu, RERUN_ENABLED=false.

---

## Benchmark scripts

    python3 scripts/mot17_to_yolo.py --mot17 data/MOT17 --out data/mot17_yolo
    python3 scripts/train_detector.py --data data/mot17_yolo/mot17.yaml
    python3 scripts/benchmark.py \
        --dataset data/MOT17 --split train --out data/mot17_results

    python3 scripts/calibrate_camera.py \
        --device 0 --rows 9 --cols 6 \
        --out config/camera_intrinsics.yaml

---

## Project structure

    perception/          Camera interface, detector, depth estimator, config
    tracking/            ByteTrack, association, motion compensation, Track
    state_estimation/    Kalman Filter, Extended KF, NIS/NEES diagnostics
    world_model/         SceneGraph, ObjectState, spatial queries
    visualization/       Rerun.io logger, OpenCV annotator
    ros2_ws/             ROS2 colcon workspace
                          src/robotics_perception_ros2/
                            camera_publisher_node    Image + CameraInfo
                            detection_node           Detection2DArray
                            tracking_node            tracked Detection2DArray
                            pose_node                Odometry + tf broadcast
                            scene_graph_node         Detection3DArray
    scripts/             Calibration, benchmark, detector training
    tests/               248 unit tests — all hardware-free
    config/              YAML configuration

---

## Extensions

**Coordinate frame management.** Object positions are currently expressed
in the camera frame. Expressing them in a fixed world frame requires
ego-pose estimation (monocular VIO, or stereo + IMU) and a tf2-style
transform tree. The integration point is SceneGraph.update(), which
accepts an optional camera pose parameter.

**ReID appearance features.** The association cost matrix in
tracking/association.py accepts an additional cosine distance term.
Adding an OSNet or FastReID backbone as an AppearanceExtractor module
enables track re-identification after long occlusion without changes
to the tracker or world model.

**IMU fusion.** The ExtendedKalmanFilter CTR model includes a turn rate
state ω that is directly observable from a gyroscope. Adding an IMU
measurement function closes the loop between the motion model and
physical sensor data, improving velocity estimates under camera motion.

**Stereo depth.** The DepthEstimator ABC accepts a StereoDepthEstimator
as a drop-in replacement for DepthAnythingEstimator. Stereo triangulation
produces metric depth without the scale ambiguity inherent to monocular
estimation, with no additional inference cost at runtime.

---

## References

    Kalman, R.E. (1960)              Optimal linear filter
    Welch & Bishop (2006)            Kalman filter tutorial
    Bar-Shalom et al. (2001)         Estimation with applications to
                                     tracking and navigation
    Bewley et al. (2016)             SORT — tracking-by-detection
    Zhang et al. (2022)              ByteTrack — arXiv:2110.06864
    Aharon et al. (2022)             BoT-SORT — arXiv:2206.14651
    Thrun, Burgard & Fox (2005)      Probabilistic Robotics
    Yang et al. (2024)               Depth Anything V2 — arXiv:2406.09414
    Campos et al. (2021)             ORB-SLAM3 — arXiv:2007.11898

---

## Hardware

    CPU    AMD Ryzen 7 7700
    GPU    NVIDIA RTX 4070 Ti  (12 GB VRAM)
    OS     Ubuntu 22.04 (WSL2)
    Python 3.10
