# Robotics Perception Pipeline

![CI](https://github.com/peclem/robotics-perception-pipeline/actions/workflows/ci.yml/badge.svg)

A modular perception stack for mobile robotics. Implements camera-based
multi-object detection, tracking, monocular ego-pose, and state estimation,
with a probabilistic world model exposed both as a standalone Python pipeline
and as a set of ROS2 adapter nodes for direct integration into Nav2-style
navigation stacks.

---

## System architecture

A complete robotics perception architecture. Implemented components are
marked. Unimplemented components and their integration points are identified.

    ╔══════════════════════════════════════════════════════════════════╗
    ║  SENSING                                                         ║
    ║                                                                  ║
    ║  [✓] Monocular RGB        CameraInterface ABC                    ║
    ║  [✓] Camera calibration   scripts/calibrate_camera.py           ║
    ║  [✓] IMU (synthetic)      IMUInterface ABC + SyntheticIMU,       ║
    ║                           Forster pre-integration with           ║
    ║                           uncertainty Jacobians.                 ║
    ║                           Real-hardware backend pending.         ║
    ║  [ ] Stereo camera        CameraInterface ABC — drop-in backend  ║
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
    ║  [✓] Monocular ego-pose       DPVO (deep patch VO), 15 Hz       ║
    ║  [ ] ReID embeddings          OSNet, IoU + cosine cost          ║
    ║  [✓] Stereo depth             StereoSGBMDepthEstimator          ║
    ║                               (cv2.StereoSGBM). Drop-in under   ║
    ║                               the existing DepthEstimator ABC.  ║
    ║                               Neural backend (IGEV / Foundation- ║
    ║                               Stereo) pending.                   ║
    ║  [✓] IMU pre-integration      Forster (2017) ΔR/Δv/Δp +         ║
    ║                               covariance + bias Jacobians.      ║
    ║  [✓] VIO fuser                Error-state EKF (15-D), Joseph    ║
    ║                               update; consumes preint + visual  ║
    ║                               pose. Loose coupling, no live     ║
    ║                               pipeline wiring yet.              ║
    ║  [ ] Semantic segmentation    —                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║  COORDINATE FRAMES                                               ║
    ║                                                                  ║
    ║  [✓] TransformTree            map ← odom ← base_link ←         ║
    ║                               camera_frame, static + dynamic     ║
    ║                               edges, lookup via common ancestor  ║
    ║  [✓] Ego-pose (monocular)     DPVOPoseEstimator wraps DPVO,      ║
    ║                               stride-decoupled (15 Hz pose at    ║
    ║                               30 Hz camera). NullPoseEstimator   ║
    ║                               fallback for camera-frame-only     ║
    ║                               mode.                              ║
    ║  [ ] Metric scale anchor      monocular VO has unobservable      ║
    ║                               scale; anchor against Depth        ║
    ║                               Anything V2 (deferred to Phase 1   ║
    ║                               validation work)                   ║
    ║  [ ] SLAM / loop closure      DPV-SLAM extension for drift       ║
    ║                                                                  ║
    ║  ObjectState.position_world (X, Y, Z) metres in the map frame   ║
    ║  is populated when ego-pose is available. SceneGraph.update()    ║
    ║  routes via the transform tree; query_nearby(frame='world')      ║
    ║  is the planner-facing metric query.                             ║
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
    ║  [✓] Dynamic obstacle grid    nav_msgs/OccupancyGrid,           ║
    ║                               2σ-covariance inflation per       ║
    ║                               object, depth-projected when      ║
    ║                               available                         ║
    ║  [✓] 3D occupancy             Sparse voxel grid →               ║
    ║                               sensor_msgs/PointCloud2           ║
    ║                               (always) + octomap_msgs/Octomap   ║
    ║                               (when octomap+_msgs installed)    ║
    ║  [✓] Per-class spatial memory STATIC / SEMI_STATIC / DYNAMIC    ║
    ║                               classification (class prior +     ║
    ║                               motion override). STATIC objects  ║
    ║                               persist indefinitely; DYNAMIC     ║
    ║                               decay in seconds.                 ║
    ║  [✓] WorldMap + ReID          DINOv2 foundation-model           ║
    ║                               embeddings, spatial gate +        ║
    ║                               cosine similarity re-association  ║
    ║                               on revisit. ObjectState carries   ║
    ║                               a stable persistent_id across     ║
    ║                               ByteTracker ID resets.            ║
    ║  [✓] Health monitor           per-stage LatencyTracker + topic  ║
    ║                               inter-arrival, OK/WARN/ERROR/     ║
    ║                               STALE on /diagnostics             ║
    ║  [ ] Static obstacle layer    pre-mapped walls; needs SLAM      ║
    ╚══════════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║  PLANNING INTERFACE                                              ║
    ║                                                                  ║
    ║  Python API:                                                     ║
    ║    SceneGraph.query_nearby(robot_position, radius,               ║
    ║                            frame='camera' | 'world')             ║
    ║      → List[(distance, ObjectState)] sorted, with covariance    ║
    ║                                                                  ║
    ║  ROS2 API:                                                       ║
    ║    /perception/scene    vision_msgs/Detection3DArray             ║
    ║    /perception/costmap  nav_msgs/OccupancyGrid (dynamic layer)   ║
    ║    /perception/voxels   sensor_msgs/PointCloud2 (3D occupancy)   ║
    ║    /perception/octomap  octomap_msgs/Octomap   (when installed)  ║
    ║    /tf                  map → camera_frame (when ego-pose on)    ║
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
    DPVOPoseEstimator (when pose_estimator.type == 'dpvo')
        Lazy DPVO init on first frame; stride-based rate decoupling
        (default stride=2 → 15 Hz pose at 30 Hz camera). Returns
        CameraPose(R, t) in world ← camera convention.
        │
        ▼
    TransformTree
        Pushes camera_pose into map ← camera_frame edge each frame.
        Resolves arbitrary (target, source) lookups via the common
        ancestor on the directed tree.
        │
        ▼
    SceneGraph
        ObjectState per confirmed track:
            position       (cx, cy)  pixels — camera frame
            position_3d    (X, Y, Z) metres — camera frame
            position_world (X, Y, Z) metres — world (map) frame [¹]
            covariance     8×8 full matrix
            velocity       (vx, vy, vw, vh) pixels/s
            trajectory     bounded KFSnapshot history
        │
        └──▶ query_nearby(pos, radius, frame='camera' | 'world')
             → planner interface (pixel or metric)

    [¹] position_world is None when no ego-pose is available
        (NullPoseEstimator, or during DPVO's bootstrap window). The
        scale is up to a monocular ambiguity until anchored against
        Depth Anything V2 (deferred).

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
    Depth estimation         Depth Anything V2          ~10 ms
    Camera motion comp.      OpenCV LK + RANSAC         ~1 ms
    Monocular ego-pose       DPVO @ stride 2            ~17 ms / call
                                                         (every 2nd frame
                                                          → ~8 ms amortised
                                                          at 640×480)
    World model              Custom scene graph         ~0.2 ms
    Total (depth disabled, pose disabled)              ~7 ms  (143 Hz)
    Total (depth enabled,  pose disabled)              ~17 ms  (58 Hz)
    Total (depth + DPVO at stride 2)                   ~25 ms  (40 Hz)

    Measured on real video (data/sample.mp4). Synthetic random-noise
    frames overstate DPVO latency 2× — DPVO inserts keyframes constantly
    without temporal coherence.

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

To enable DPVO ego-pose, see "DPVO setup" below. To launch the ROS2
graph, see "ROS2 integration" further down.

---

## DPVO setup (optional, for `pose_estimator.type: dpvo`)

DPVO is a custom CUDA-extension PyTorch package. On the project's
target stack (PyTorch 2.11 + CUDA 13.0, WSL2 Ubuntu 22.04) the
upstream sources need three mechanical patches; the build is otherwise
clean.

    # 1. CUDA toolkit 13.0 (provides nvcc matching torch's cu130)
    sudo apt install cuda-toolkit-13-0      # via NVIDIA's WSL apt repo

    # 2. Clone DPVO and pull Eigen
    mkdir -p third_party && cd third_party
    git clone --recursive https://github.com/princeton-vl/DPVO.git
    cd DPVO
    wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    unzip -q eigen-3.4.0.zip -d thirdparty

    # 3. Python deps (matching torch 2.11 + cu130)
    pip install wheel ninja numba einops pypose kornia yacs plyfile evo
    pip install torch-scatter \
        -f https://data.pyg.org/whl/torch-2.11.0+cu130.html

    # 4. Patch deprecated PyTorch API in DPVO sources
    find dpvo -name "*.cpp" -o -name "*.cu" -o -name "*.h" \
        | xargs sed -i 's/\.type(), "/\.scalar_type(), "/g'
    sed -i 's|::detail::scalar_type(the_type)|the_type|' \
        dpvo/lietorch/include/dispatch.h

    # 5. Build the CUDA extensions (force CUDA 13.0 on PATH)
    CUDA_HOME=/usr/local/cuda-13.0 \
        PATH=/usr/local/cuda-13.0/bin:$PATH \
        pip install . --no-build-isolation

    # 6. Download the pretrained checkpoint
    wget "https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip?dl=1" \
        -O models.zip
    unzip -o models.zip -d models/

Verify:

    python3 -c "from dpvo.dpvo import DPVO; print('DPVO OK')"
    python3 scripts/benchmark_dpvo_latency.py    # ~17 ms median at 640×480

Then set `pose_estimator.type: dpvo` in config/default.yaml.

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

    /perception/scene  ──▶ occupancy_grid_node ──▶ /perception/costmap
                                                   (nav_msgs/OccupancyGrid,
                                                    20×20 m @ 5 cm/cell,
                                                    2σ-inflated dynamic layer)

    /perception/scene  ──▶ occupancy_3d_node   ──┬─▶ /perception/voxels
                                                  │   (sensor_msgs/PointCloud2,
                                                  │    sparse occupied centres)
                                                  └─▶ /perception/octomap
                                                      (octomap_msgs/Octomap,
                                                       only if octomap +
                                                       octomap_msgs installed)

    all /perception/* topics ──▶ health_monitor_node ──▶ /diagnostics
                                  (diagnostic_msgs/DiagnosticArray @ 1 Hz,
                                   per-stage OK/WARN/ERROR/STALE based on
                                   topic inter-arrival vs per-stage budgets)

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

**Phase 4 perf attempts (honestly reported):**

- *Image downscale to 640×480* — `publish_width:=640 publish_height:=480`
  reduces DDS payload ~3×. On this stack (Python rclpy + Fast-DDS +
  WSL2): negligible throughput improvement (~6 Hz unchanged).
  Bandwidth wasn't the bottleneck. Likely useful on lower-spec
  hardware or constrained network deployments.
- *Composite launcher* — `perception_pipeline_composite.launch.py`
  runs all nodes in a single Python process under a
  `MultiThreadedExecutor`. Goal: shared CUDA context + zero-copy IPC.
  Reality: Python `rclpy` does **not** expose
  `use_intra_process_comms` (C++ `ComposableNodeContainer` only), so
  messages still serialise. Combined with GIL contention, throughput
  is slightly *worse* (~5 Hz) than the multi-process variant.

The real fix on this stack is a C++ rewrite of the adapter nodes as
composable components. Deferred — multi-process Python at 6 Hz is
adequate for demonstrating the graph; real production deployments
would invest in the C++ port.

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
extension first — see "DPVO setup" above.

---

## Tests

    python3 -m pytest tests/ -m "not integration" -v

535 unit tests across detection, tracking, state estimation (including
bias-aware IMU pre-integration with numerical Jacobian verification),
world model, coordinate frames (TransformTree), DPVO wrapper, mono +
stereo depth, occupancy grid, stability classification, appearance
extractor, WorldMap, health monitor, IMU interface, visualisation, and
benchmarks. All tests use SyntheticCamera / synthetic IMU / synthetic
data — no hardware required. Integration tests (real GPU, live DPVO /
DINOv2 models) are marked and excluded from CI; run with
`pytest -m integration`.

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
        enabled: false              # master switch (false → NullDepthEstimator)
        type: depth_anything        # 'null' | 'depth_anything' | 'stereo_sgbm'
        model: "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
        # stereo_sgbm-only tunables — needs frame.right_image + intrinsics.baseline_m
        sgbm_num_disparities: 96    # must be divisible by 16
        sgbm_block_size: 7          # must be odd

    pose_estimator:
        type: "null"        # 'null' | 'dpvo'
        stride: 2           # DPVO every Nth frame → pose at 30/stride Hz
        patches_per_frame: 96

    coordinate_frames:
        enabled: false      # set true to use TransformTree for position_world
        root_frame: map
        camera_frame: camera_frame
        static_extrinsics: []   # parent→child SE(3) edges, e.g. base_link → camera_frame

    occupancy_grid:
        enabled: false      # publish nav_msgs/OccupancyGrid from the scene graph
        resolution_m: 0.05  # 5 cm per cell (Nav2 default)
        size_x_m: 20.0
        size_y_m: 20.0
        origin_x_m: -10.0   # grid centred at world origin
        origin_y_m: -10.0
        default_inflation_m: 0.5

    occupancy_3d:
        enabled: false      # publish sparse 3D voxels (PointCloud2 + optional Octomap)
        resolution_m: 0.10  # 10 cm per voxel
        size_z_m: 3.0       # vertical span (standing person / arm workspace)
        origin_z_m: -0.5    # 0.5 m below ground → ceiling at +2.5 m
        per_class_inflation_m:
            person: 0.40
            car:    2.00

    stability:
        timeouts_s:         # per-class memory durations (seconds)
            DYNAMIC:     1.5
            SEMI_STATIC: 60.0
            # STATIC defaults to inf (never auto-pruned)
        class_overrides: {}     # COCO class → STATIC/SEMI_STATIC/DYNAMIC
        demote_speed_px_s:  100.0   # observed motion → demote to DYNAMIC
        demote_frames:      90      # sustained for N frames
        promote_speed_px_s: 10.0    # stationary → promote DYNAMIC objects
        promote_frames:     900     # sustained for N frames

    appearance:                 # ReID embedding backend
        type: "null"            # 'null' | 'dinov2'
        model: "facebook/dinov2-small"
        device: "cuda"

    world_map:                  # long-term spatial memory
        enabled: false
        spatial_gate_m: 1.5     # re-association candidates within this distance
        similarity_threshold: 0.75   # cosine similarity threshold
        allow_spatial_only: true     # fall back when embeddings missing

    health_monitor:             # per-stage latency budgets + diagnostics
        enabled: true
        warn_after:  3              # consecutive budget breaches → WARN
        error_after: 30             # consecutive budget breaches → ERROR
        log_period_s: 5.0
        stage_budgets_ms:
            detector:    12.0
            depth:       20.0
            pose:        25.0
            tracker:     3.0
            scene_graph: 5.0
            appearance:  15.0
            frame_total: 40.0

Environment variable overrides: DEVICE=cpu, RERUN_ENABLED=false.

---

## Benchmark scripts

    # MOT17 → YOLO format + fine-tuning
    python3 scripts/mot17_to_yolo.py --mot17 data/MOT17 --out data/mot17_yolo
    python3 scripts/train_detector.py --data data/mot17_yolo/mot17.yaml

    # MOTA / MOTP / IDSW benchmark on MOT17 train split
    python3 scripts/benchmark.py \
        --dataset data/MOT17 --split train --out data/mot17_results

    # DPVO per-frame latency sweep (resolution × patches × stride)
    python3 scripts/benchmark_dpvo_latency.py

    # Intrinsics calibration from a checkerboard
    python3 scripts/calibrate_camera.py \
        --device 0 --rows 9 --cols 6 \
        --out config/camera_intrinsics.yaml

---

## Project structure

    perception/          Camera interface, detector, depth estimator,
                          pose estimator (Null + DPVO), appearance
                          extractor (Null + DINOv2), IMU interface
                          (Null + Synthetic), TransformTree, typed config
    tracking/            ByteTrack, association, motion compensation, Track
    state_estimation/    Kalman Filter, Extended KF, NIS/NEES,
                          IMU pre-integration, visual-inertial EKF
                          diagnostics, IMU pre-integration (Forster 2017)
    world_model/         SceneGraph, ObjectState (+ position_world,
                          stability, persistent_id), spatial queries
                          (camera- and world-frame), OccupancyGridBuilder
                          (dynamic obstacle layer), stability classification,
                          WorldMap (long-term spatial memory + ReID
                          re-association)
    visualization/       Rerun.io logger, OpenCV annotator
    ros2_ws/             ROS2 colcon workspace
                          src/robotics_perception_ros2/
                            camera_publisher_node    Image + CameraInfo
                            detection_node           Detection2DArray
                            tracking_node            tracked Detection2DArray
                            pose_node                Odometry + tf broadcast
                            scene_graph_node         Detection3DArray
                            occupancy_grid_node      OccupancyGrid (Nav2 costmap-ready)
                            occupancy_3d_node        PointCloud2 + optional Octomap
                            health_monitor_node      DiagnosticArray
                            composite_node           single-process bundle (perf experiment)
    scripts/             Calibration, benchmark (MOT17, DPVO latency),
                          detector training
    third_party/         External clones (DPVO + bundled Pangolin / DBoW2)
                          — not committed; see DPVO setup in README
    tests/               575 unit tests — all hardware-free; integration
                          tests marked separately
    config/              YAML configuration

---

## Extensions

**Metric scale anchoring for DPVO.** Monocular VO has unobservable
absolute scale — DPVO's translations are in arbitrary units until
anchored. Depth Anything V2 already provides metric depth, so the
calibration is feasible: at init (or via a continuous low-pass
filter), compute the scale ratio between DPVO's reported depth and
the median metric depth. Deferred until Phase 1 validation work
(TUM RGB-D) provides ground truth to measure against.

**Loop closure (DPV-SLAM).** DPVO drifts over long runs. Princeton's
DPV-SLAM extension adds long-term loop closure on top of DPVO with
the same wrapper API. Drop-in upgrade when drift becomes a concern.

**Eviction policy for WorldMap.** WorldMap currently grows
monotonically — entries are never auto-evicted. A production
deployment should add (a) a max-age cap, (b) a memory-size cap with
LRU eviction, and (c) "revisited and not seen" eviction (robot
returns to a region and doesn't observe a remembered entry → mark
for verification, evict after K confirmations). Deferred because
the right policy is deployment-specific.

**ReID appearance features.** The association cost matrix in
tracking/association.py accepts an additional cosine distance term.
Adding an OSNet or FastReID backbone as an AppearanceExtractor module
enables track re-identification after long occlusion without changes
to the tracker or world model.

**IMU-VIO fusion.** Full pipe: `IMUInterface` ABC + `SyntheticIMU`
backend, Forster (2017) pre-integration with `PreintegratedMeasurement`
(ΔR, Δv, Δp + 9×9 covariance + bias Jacobians for first-order
correction), and now the *fuser* — a loosely-coupled error-state EKF
(`state_estimation/visual_inertial_ekf.py`). 15-D error state
[δp, δv, δθ, δb_g, δb_a]; predict consumes one `PreintegratedMeasurement`
between visual frames; update consumes a 6-DOF `CameraPose` from any
`PoseEstimator` (DPVO ships today; OpenVINS / VINS-Fusion / ORB-SLAM3
slot into the same ABC). Joseph-form covariance update for PSD safety.
Camera = IMU body in v1 (extrinsic foldable via TransformTree). What's
NOT in this revision: live pipeline orchestration (IMU producer + visual
loop sync + ROS2 fused-pose publisher), gravity-aligned initialiser,
chi-square outlier rejection on the visual update.

**Neural stereo backend.** `StereoSGBMDepthEstimator` (classical
cv2.StereoSGBM) ships today and is genuinely the right pick for CPU-
constrained or GPU-tight deployments. A neural stereo backend
(IGEV-Stereo, FoundationStereo, CREStereo) slots into the same
`DepthEstimator` ABC — better accuracy on textureless / low-feature
scenes but adds a 1–3 GB GPU model and a CUDA build dependency.
Deferred until a use case shows SGBM's accuracy is the bottleneck.

**Bias-aware IMU pre-integration.** Already implemented — the
PreintegratedMeasurement carries five 3×3 bias Jacobians and
`correct_for_bias()` applies first-order corrections in O(1).
What's still missing is the *fuser* that consumes these
measurements: an error-state EKF or factor graph that estimates
biases online and feeds the corrected pre-integration back into
DPVO's pose updates.

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
    Teed, Lipson & Deng (2023)       DPVO — arXiv:2208.04726
    Foote (2013)                     tf: the transform library (ICRA)
    Forster et al. (2017)            On-Manifold Pre-integration for
                                     VIO — arXiv:1512.02363

---

## Hardware

    CPU    AMD Ryzen 7 7700
    GPU    NVIDIA RTX 4070 Ti  (12 GB VRAM)
    OS     Ubuntu 22.04 (WSL2)
    Python 3.10
