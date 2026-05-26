# Robotics Perception Pipeline

![CI](https://github.com/peclem/robotics-perception-pipeline/actions/workflows/ci.yml/badge.svg)

A modular perception stack for mobile robotics. Implements camera-based
multi-object detection, tracking, monocular ego-pose, and state estimation,
with a probabilistic world model exposed both as a standalone Python pipeline
and as a set of ROS2 adapter nodes for direct integration into Nav2-style
navigation stacks.

---

## System architecture

    ╔══════════════════════════════════════════════════════════════════╗
    ║  SENSING                                                         ║
    ║                                                                  ║
    ║  Monocular RGB        CameraInterface ABC                        ║
    ║  Camera calibration   scripts/calibrate_camera.py                ║
    ║  IMU                  IMUInterface ABC + SyntheticIMU + CODa     ║
    ║                       (replay). Forster pre-integration with     ║
    ║                       uncertainty Jacobians.                     ║
    ╚══════════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║  SENSOR PREPROCESSING                                            ║
    ║                                                                  ║
    ║  Distortion correction    load_intrinsics() + OpenCV             ║
    ║  Frame timestamping       time.monotonic(), CameraFrame          ║
    ╚══════════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║  PERCEPTION                                                      ║
    ║                                                                  ║
    ║  Object detection         YOLOv8n fine-tuned on MOT17            ║
    ║  Multi-object tracking    ByteTrack two-stage association,       ║
    ║                           optional DINOv2 appearance blend       ║
    ║                           (StrongSORT/Deep OC-SORT-style)        ║
    ║  KF state estimation      Joseph form, 8D state, NIS             ║
    ║  EKF — constant turn rate 9D state + ω, analytical Jac.         ║
    ║  Camera motion comp.      LK optical flow + affine RANSAC        ║
    ║  Monocular depth          Depth Anything V2, metric              ║
    ║  Monocular ego-pose       DPVO (deep patch VO) / DPV-SLAM        ║
    ║                           (DPVO + loop closure)                  ║
    ║  ReID embeddings          DINOv2 (foundation, class-agnostic).   ║
    ║                           IoU + cosine cost, StrongSORT-style    ║
    ║                           blend.                                 ║
    ║  Stereo depth             Classical (cv2.StereoSGBM) + neural    ║
    ║                           (RAFT-Stereo). Drop-in under the       ║
    ║                           existing DepthEstimator ABC.           ║
    ║  IMU pre-integration      Forster (2017) ΔR/Δv/Δp + covariance + ║
    ║                           bias Jacobians.                        ║
    ║  VIO fuser                Loosely-coupled error-state EKF (16-D, ║
    ║                           scale state default-off — see          ║
    ║                           "Outdoor-pose Stage 1" note), Joseph   ║
    ║                           update, ZUPT, first-frame anchor.      ║
    ║  VIO live orchestration   VIOPoseEstimator wraps visual + IMU    ║
    ║                           behind PoseEstimator ABC.              ║
    ║                           vio.enabled flag in config.            ║
    ║  Semantic segmentation    Mask2Former (Swin-T, Cityscapes),      ║
    ║                           drivable_mask + class-stability        ║
    ║                           helpers. Wired into launch.py with     ║
    ║                           per-frame stage + SceneGraph stability ║
    ║                           refinement.                            ║
    ╚══════════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║  COORDINATE FRAMES                                               ║
    ║                                                                  ║
    ║  TransformTree            map ← odom ← base_link ← camera_frame, ║
    ║                           static + dynamic edges, lookup via     ║
    ║                           common ancestor.                       ║
    ║  Ego-pose (monocular)     DPVOPoseEstimator wraps DPVO, stride-  ║
    ║                           decoupled. NullPoseEstimator fallback  ║
    ║                           for camera-frame-only mode.            ║
    ║  SLAM / loop closure      DPV-SLAM — DPVO + proximity + DBoW2    ║
    ║                           loop closure. Drift-corrected on       ║
    ║                           revisit. pose_estimator.type='dpv_slam'║
    ║                                                                  ║
    ║  ObjectState.position_world (X, Y, Z) metres in the map frame    ║
    ║  is populated when ego-pose is available. SceneGraph.update()    ║
    ║  routes via the transform tree; query_nearby(frame='world')      ║
    ║  is the planner-facing metric query.                             ║
    ╚══════════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║  WORLD MODEL                                                     ║
    ║                                                                  ║
    ║  Dynamic object tracking  SceneGraph, ObjectState                ║
    ║  Per-object covariance    full 8x8 / 9x9 matrix                  ║
    ║  Trajectory history       bounded KFSnapshot deque               ║
    ║  Metric 3D positions      position_3d from depth                 ║
    ║  Spatial queries          query_nearby(pos, radius)              ║
    ║  Uncertainty queries      Mahalanobis distance                   ║
    ║  Dynamic obstacle grid    nav_msgs/OccupancyGrid, 2σ-covariance  ║
    ║                           inflation per object, depth-projected. ║
    ║  3D occupancy             Sparse voxel grid →                    ║
    ║                           sensor_msgs/PointCloud2 (always) +     ║
    ║                           octomap_msgs/Octomap (when installed). ║
    ║  Room layer (3D scene-    Morphological-erosion clustering on    ║
    ║  graph hierarchy)         occupancy → labelled polygons with     ║
    ║                           object membership.                     ║
    ║  Per-class spatial memory STATIC / SEMI_STATIC / DYNAMIC         ║
    ║                           classification (class prior + motion   ║
    ║                           override).                             ║
    ║  WorldMap + ReID          DINOv2 foundation-model embeddings,    ║
    ║                           spatial gate + cosine similarity       ║
    ║                           re-association on revisit. ObjectState ║
    ║                           carries a stable persistent_id across  ║
    ║                           ByteTracker ID resets.                 ║
    ║  Semantic SLAM            Persistent voxel-hashed metric-        ║
    ║                           semantic map (SemanticMap / Kimera-    ║
    ║                           Semantics back-end). Per-voxel         ║
    ║                           occupancy log-odds + fused semantic    ║
    ║                           class distribution. ROS2 +             ║
    ║                           Rerun viewers.                         ║
    ║  Health monitor           per-stage LatencyTracker + topic       ║
    ║                           inter-arrival, OK/WARN/ERROR/STALE     ║
    ║                           on /diagnostics.                       ║
    ║  Drivable freespace       IPM projection of the semantic         ║
    ║  costmap                  drivable mask onto the ground.         ║
    ║                           Depth-aware backend + flat-ground      ║
    ║                           fallback. nav_msgs/OccupancyGrid       ║
    ║                           (0=drive, -1=unknown) — fuses with the ║
    ║                           obstacle layer in Nav2.                ║
    ╚══════════════════════════════════════════════════════════════════╝
                          │
                          ▼
    ╔══════════════════════════════════════════════════════════════════╗
    ║  PLANNING INTERFACE                                              ║
    ║                                                                  ║
    ║  Python API:                                                     ║
    ║    SceneGraph.query_nearby(robot_position, radius,               ║
    ║                            frame='camera' | 'world')             ║
    ║      → List[(distance, ObjectState)] sorted, with covariance     ║
    ║                                                                  ║
    ║  ROS2 API:                                                       ║
    ║    /perception/scene             vision_msgs/Detection3DArray    ║
    ║    /perception/costmap           nav_msgs/OccupancyGrid          ║
    ║                                  (dynamic obstacle layer)        ║
    ║    /perception/drivable_costmap  nav_msgs/OccupancyGrid          ║
    ║                                  (drivable freespace, IPM-       ║
    ║                                  projected from semantic mask)   ║
    ║    /perception/drivable_mask     sensor_msgs/Image (mono8)       ║
    ║    /perception/voxels            sensor_msgs/PointCloud2         ║
    ║    /perception/octomap           octomap_msgs/Octomap            ║
    ║    /perception/semantic_map      sensor_msgs/PointCloud2 XYZRGB  ║
    ║    /perception/depth             sensor_msgs/Image (32FC1)       ║
    ║    /perception/odom              nav_msgs/Odometry (fused VIO    ║
    ║                                  when vio.enabled, else visual)  ║
    ║    /tf                           map → camera_frame              ║
    ╚══════════════════════════════════════════════════════════════════╝

    ROS2 adapter nodes: ros2_ws/src/robotics_perception_ros2/ wraps each
    module in thin sensor_msgs / vision_msgs / nav_msgs interfaces.
    Launch with
        ros2 launch robotics_perception_ros2 perception_pipeline.launch.py
    See "ROS2 integration" section below for topic graph + setup.

---

## Data flow

    CameraFrame (image, timestamp, intrinsics, optional right_image)
        │
        ├──▶ YOLOv8n ──────────────▶ Detection[]
        │                              (bbox, confidence, class)
        │
        ├──▶ Depth Anything V2 / ─▶ DepthEstimate[] + dense depth_map
        │    SGBM / RAFT-Stereo
        │
        ├──▶ Mask2Former (Swin-T) ─▶ SemanticMask
        │                              (drivable_mask, surface-class
        │                               stability prior)
        │
        ├──▶ DINOv2 ──────────────▶ per-detection embeddings (L2-norm)
        │    (gated on use_appearance OR world_map.enabled)
        │
        ├──▶ CameraMotionCompensator
        │      LK optical flow on background keypoints (object-bbox
        │      masked) + affine RANSAC → homography H⁻¹ applied to
        │      track states before association
        │
        ▼
    ByteTracker
        Stage 1: D_high ↔ all tracks    (Hungarian, IoU + α·cosine)
        Stage 2: D_low  ↔ lost tracks   (occlusion rescue)
        Per track:
            KF.predict(dt)   →  x̂ = F x,  P = F P Fᵀ + Q
            KF.update(z)     →  x = x̂ + K(z − Hx̂),  Joseph form
            NIS = yᵀ S⁻¹ y  ~  χ²(4),  bounds [0.711, 9.488]
            embedding ← EMA(0.9 * old, 0.1 * new)
        │
        ▼
    Visual ego-pose: DPVOPoseEstimator   (pose_estimator.type='dpvo')
                  or DPVSLAMPoseEstimator (pose_estimator.type='dpv_slam')
        Lazy DPVO init on first frame; stride-based rate decoupling
        (default stride=2 → 15 Hz pose at 30 Hz camera). Returns
        CameraPose(R, t) in world ← camera convention. With 'dpv_slam',
        loop closure drift-corrects the trajectory on revisit — the
        corrected pose can jump at a loop event (absorbed by the
        map ← odom transform edge).
        │
        ▼ (when vio.enabled)
    VIOPoseEstimator (wraps visual + IMU)
        Pulls IMU samples since last visual frame → Forster
        pre-integration → 16-D error-state EKF predict; visual pose
        as 6-DOF measurement → EKF update (Joseph form). Output is
        the fused CameraPose with the same world ← camera signature.
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
            track_id       session-local — ByteTracker assignment
            persistent_id  stable across re-associations (WorldMap)
            position       (cx, cy)  pixels — camera frame
            position_3d    (X, Y, Z) metres — camera frame
            position_world (X, Y, Z) metres — world (map) frame [¹]
            covariance     8×8 full matrix
            velocity       (vx, vy, vw, vh) pixels/s
            stability      STATIC / SEMI_STATIC / DYNAMIC
                           (class prior + motion override +
                            optional semantic surface refinement)
            history        bounded KFSnapshot deque
        │
        ├──▶ WorldMap re-association
        │      Spatial gate (≤spatial_gate_m) + cosine appearance
        │      gate (≥similarity_threshold) → adopts persistent_id
        │      on revisit. Opt-in eviction: max_age_s + max_entries.
        │
        ├──▶ OccupancyGridBuilder ─▶ nav_msgs/OccupancyGrid
        │      2σ position covariance → metric inflation per object.
        │
        ├──▶ Occupancy3DBuilder ──▶ sparse {(i,j,k): occ} →
        │                            PointCloud2 + optional Octomap.
        │
        ├──▶ RoomLayer (when room_layer.enabled)
        │      Morphological erode → CC → contour on 2D grid →
        │      Room polygons with object membership.
        │
        ├──▶ project_drivable_to_grid (when drivable_costmap.enabled)
        │      IPM (flat-ground or depth-aware) from SemanticMask's
        │      drivable_mask + CameraPose → nav_msgs/OccupancyGrid
        │      with 0=drivable, -1=unknown.
        │
        └──▶ query_nearby(pos, radius, frame='camera' | 'world')
              → planner interface (pixel or metric)

    [¹] position_world is None when no ego-pose is available
        (NullPoseEstimator, or during DPVO's bootstrap window). The
        scale is up to a monocular ambiguity until anchored against
        Depth Anything V2 (deferred to Phase 1 validation).

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

### Appearance-aware matching — MOT17 ablation (honest negative)

Held-out validation pair (MOT17-09, MOT17-11, SDP detector track),
fine-tuned YOLOv8n, otherwise identical config. Toggling
`tracker.use_appearance` blends DINOv2 cosine distance into the
Hungarian cost (StrongSORT-style; weight 0.25, EMA 0.9).

    Configuration                Mean MOTA   FP     FN     IDSW   Mean Hz
    ─────────────────────────────────────────────────────────────────────
    IoU only (baseline)             50.2%   1744   5533   115     66
    IoU + DINOv2 appearance         50.7%   1724   5523   116     31

The architectural change is in (`tracker.use_appearance: true`,
DINOv2 embeddings threaded through the matcher), but on this
benchmark the measurable lift is within noise — +0.5 pp MOTA driven
almost entirely by MOT17-09 (+1.6 pp); MOT17-11 regressed -0.5 pp.
IDSW didn't drop. Throughput halved as expected from the added
DINOv2 forward per detection.

The reasonable interpretation: MOT17 is mostly well-separated upright
pedestrians where IoU + the Kalman filter already disambiguates most
matches. The detector is the dominant error source (FP+FN ≫ IDSW).
Appearance-based ReID has its biggest payoff on DanceTrack-style
benchmarks (similar appearances, dense crowds, frequent crossovers)
or on long occlusion gaps where IoU is uninformative. The
infrastructure is in place for those benchmarks when they're worth
running.

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

    Component                  Library / method            Latency (4070Ti)
    ────────────────────────────────────────────────────────────────────────
    Object detection           YOLOv8n (Ultralytics)       ~5 ms
    Multi-object tracking      ByteTrack                   ~0.5 ms
    State estimation           KF / EKF (NumPy)            ~0.1 ms
    Camera motion comp.        OpenCV LK + RANSAC          ~1 ms
    Depth (monocular)          Depth Anything V2           ~10 ms
    Depth (stereo, classical)  cv2.StereoSGBM (CPU)        ~30 ms @ 640×480
    Depth (stereo, neural)     RAFT-Stereo (Princeton-VL)  ~25 ms / 7 iters
    Appearance / ReID          DINOv2-small (HF)           ~6 ms (batched)
    Semantic segmentation      Mask2Former-Swin-T (HF)     ~25 ms
    Monocular ego-pose         DPVO @ stride 2             ~17 ms / call
                                                            (every 2nd frame
                                                             → ~8 ms amortised
                                                             at 640×480)
    Visual-inertial fusion     Error-state EKF (16-D)      ~0.3 ms / step
    World model                SceneGraph + KFSnapshot     ~0.2 ms
    Occupancy 2D / 3D          Numpy sphere stamping       ~0.5 ms / 1.0 ms
    Drivable costmap (IPM)     Vectorised projector        ~0.8 ms
    Room layer                 cv2 erode + CC + contour    ~1 ms (100×100 grid)

    Total (detection + tracking only — minimal)            ~6 ms  (165 Hz)
    Total (with depth, appearance off, pose off)           ~17 ms (58 Hz)
    Total (with depth + DPVO @ stride 2)                   ~25 ms (40 Hz)
    Total (with depth + DPVO + semantic, no appearance)    ~50 ms (20 Hz)
    Total (everything on — heavy showcase)                 ~60 ms (16 Hz)

    Measured on real video (data/sample.mp4). Synthetic random-noise
    frames overstate DPVO latency 2× — DPVO inserts keyframes constantly
    without temporal coherence. Semantic latency dominates the "everything
    on" budget; disable it for higher Hz when you only need geometry.

---

## Quick start

    git clone https://github.com/peclem/robotics-perception-pipeline
    cd robotics-perception-pipeline

    python3.10 -m venv .venv && source .venv/bin/activate

    # Install CUDA-matching torch FIRST (skipped if you have CPU-only torch
    # already installed). Adjust the index URL to match your CUDA — this
    # repo's reference stack runs cu130; the public PyPI default works
    # on CPU.
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu130

    # Editable install pulls the core runtime (numpy, scipy, opencv,
    # pyyaml) and lets you pick optional backends via extras. `[all]`
    # is the convenience target — ultralytics + transformers +
    # accelerate + rerun-sdk for the full pipeline. Pick narrower
    # extras (`detect`, `depth`, `semantic`, `appearance`, `viz`,
    # `dev`) if you only want a subset.
    pip install -e ".[all,dev]"

    RERUN_ENABLED=false python3 launch.py --source synthetic
    RERUN_ENABLED=false python3 launch.py --source video --input data/clip.mp4

To enable DPVO ego-pose, see "DPVO setup" below. To launch the ROS2
graph, see "ROS2 integration" further down.

---

## Recording a demo / showing off the perception

Two artefacts to share: a **Rerun `.rrd` recording** (interactive
3D scene + tracks + uncertainty, shareable via Rerun's hosted viewer
at https://rerun.io/viewer) and an **annotated `.mp4`** (works
everywhere — GitHub READMEs, social media, slides).

Both are produced by a single run.

### From a video file

    # Annotated mp4 (always) + .rrd (when --rerun-save is passed)
    python3 launch.py \
        --source video --input data/your_clip.mp4 \
        --output data/your_clip_annotated.mp4 \
        --rerun-save data/your_clip.rrd

Drop `your_clip.rrd` into https://rerun.io/viewer to share a link;
upload `your_clip_annotated.mp4` to YouTube, GitHub release, or
embed via README.

### From a USB webcam

    python3 launch.py --source webcam --rerun-save data/webcam_demo.rrd

Pipe through `--output` if you also want the mp4.

### From an iPhone (DroidCam / Larix Broadcaster / OBS NDI)

Install a "phone as IP camera" app on the phone, point it at your
PC's IP, then treat the resulting stream URL like a video file:

    python3 launch.py --source video --input rtsp://<phone-ip>:1935/live \
        --rerun-save data/iphone_demo.rrd

### Recording shortcuts

`scripts/record_demo.py` is a thin wrapper that turns on Rerun
recording with sensible defaults:

    python3 scripts/record_demo.py --input data/your_clip.mp4 \
        --out data/demo

writes `data/demo.rrd` + `data/demo.mp4`.

### What's visible in the recording

By default: camera image, 2D detection + track boxes, KF velocity
arrows, 2σ position-covariance ellipses. With the right config
flags, additionally:

- **`occupancy_3d.enabled: true`** → 3D voxel cloud at
  `world/occupancy_3d` (Points3D, voxel-sized radii)
- **`room_layer.enabled: true`** → room polygons + labels at
  `world/rooms/polygons` + `world/rooms/labels` (LineStrips3D in
  world XY at z=0)
- **`semantic.enabled: true` + `type: mask2former`** → per-pixel
  semantic-class overlay at `world/camera/semantic` (SegmentationImage
  + AnnotationContext for the class table)
- **`pose_estimator.type: dpvo` (or VIO)** → camera image rendered as
  a 3D frustum positioned in the world via `rr.Pinhole` + `rr.Transform3D`,
  plus the rolling ego-trajectory as `LineStrips3D` at
  `world/ego_trajectory`. With this enabled, the image, voxels,
  rooms, and tracks all live in one 3D world view in the Rerun viewer.
- **Tracked-object world markers** at `world/scene/objects` —
  `rr.Points3D` for each `ObjectState.position_world`, labelled
  `#track_id class p:persistent_id` so WorldMap re-association is
  visible in the viewer. Always logged (cheap when no objects).
- Per-track persistent_id (WorldMap re-association), metric depth,
  frame-level metrics (detector / tracker latency, FPS, lost-track count)

Enable everything for a maximal showcase by adding the layer flags
to your config and running `scripts/record_demo.py` as above.

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

## RAFT-Stereo setup (optional, for `depth.type: raft_stereo`)

RAFT-Stereo (Princeton-VL, 3DV 2021) is the neural-stereo backend
alongside the classical `stereo_sgbm`. Pure PyTorch — no custom CUDA
extensions to compile, unlike DPVO or IGEV-Stereo. Install is just a
clone + weights download.

    # 1. Clone the upstream repo
    cd third_party
    git clone https://github.com/princeton-vl/RAFT-Stereo.git
    cd RAFT-Stereo

    # 2. Download the published checkpoints (~100 MB total)
    bash download_models.sh

    # 3. (optional) RAFT-Stereo brings a couple of pip deps you may
    # not have yet
    pip install opt_einsum

Verify:

    python3 -c "from perception.depth_estimator import \
        RAFTStereoDepthEstimator; \
        e = RAFTStereoDepthEstimator(); print('ready =', e.is_ready)"

Then set in config/default.yaml:

    depth:
      enabled: true
      type: raft_stereo

The default checkpoint `raftstereo-middlebury.pth` is the strongest
generalist on indoor scenes. For outdoor low-texture, use
`raftstereo-eth3d.pth`; for sub-10 ms inference, use
`raftstereo-realtime.pth`. Wire via `depth.raft_checkpoint`.

If the load fails for any reason (missing repo, upstream API drift,
state-dict mismatch), the wrapper logs a warning and falls back to
zero depth — the rest of the pipeline keeps running.

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
                                            (fused VIO pose when vio.enabled,
                                             bare visual pose otherwise)

    /perception/image_raw  ──▶ depth_node  ──▶ /perception/depth
                                               (sensor_msgs/Image 32FC1,
                                                dense metric depth)

    /perception/image_raw  ──▶ drivable_mask_node  ──▶ /perception/drivable_mask
                                                       (sensor_msgs/Image mono8,
                                                        Mask2Former drivable
                                                        surface classes)

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

    /perception/drivable_mask + /perception/depth + tf
                       ──▶ drivable_costmap_node ──▶ /perception/drivable_costmap
                                                     (nav_msgs/OccupancyGrid,
                                                      IPM-projected drivable
                                                      freespace; depth-aware
                                                      when /perception/depth
                                                      is available, flat-
                                                      ground fallback)

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
    ros2 topic list                            # 10 /perception/* topics
    ros2 topic hz /perception/scene            # ~6 Hz on this hardware
    ros2 topic echo /perception/scene --once   # single Detection3DArray

### Honest performance notes

The standalone Python pipeline runs end-to-end at ~30 Hz. The ROS2
graph runs at ~6 Hz at 1280×720 due to:

- DDS serialisation of 1280×720 BGR images (~2.7 MB / msg / topic)
- Per-process CUDA context overhead (detection + pose both load torch)
- Synchronous callback chains across 10 processes

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

A C++ rewrite of the adapter nodes as composable components would
get true zero-copy IPC, but it is **explicitly off the roadmap** —
the project's bottleneck is perception-layer accuracy (outdoor pose,
tightly-coupled VIO), not ROS2 transport. 6 Hz multi-process Python
is adequate for the demo role ROS2 plays here.

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

959 unit tests across detection, tracking, state estimation (including
bias-aware IMU pre-integration with numerical Jacobian verification +
the visual-inertial EKF with scale state, ZUPT, anchoring), world
model, coordinate frames (TransformTree), DPVO wrapper, mono + stereo
depth, occupancy grid (2D + 3D), drivable freespace IPM projector,
stability classification, appearance extractor, WorldMap (with opt-in
eviction), room layer, semantic map (Kimera-Semantics), health
monitor, IMU interface, pose-estimator factory, visualisation, and
benchmarks. All tests use SyntheticCamera / synthetic IMU / synthetic
data — no hardware required. Integration tests (real GPU, live DPVO /
DINOv2 / Mask2Former models) are marked and excluded from CI; run
with `pytest -m integration`. CI runs the unit suite on every push
and PR to main (see badge at the top).

---

## Configuration

All parameters are externalised in `config/default.yaml`. Highlights
below — the YAML itself is the canonical reference.

    detector:
        model: "runs/detect/mot17_finetune/weights/best.pt"
        confidence_threshold: 0.25

    tracker:
        use_ekf: false              # true = ExtendedKalmanFilter (CTR model)
        use_cmc: false              # true = camera motion compensation
        use_appearance: false       # blend DINOv2 cosine distance into matcher
        appearance_weight: 0.25     # 0 = IoU only, 1 = appearance only
        appearance_ema: 0.9         # EMA alpha on the track's stored embedding

    depth:
        enabled: false              # master switch (false → NullDepthEstimator)
        type: depth_anything        # 'null' | 'depth_anything'
                                    #       | 'stereo_sgbm' | 'raft_stereo'
        model: "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
        # stereo_sgbm-only — needs frame.right_image + intrinsics.baseline_m
        sgbm_num_disparities: 96    # must be divisible by 16
        sgbm_block_size: 7          # must be odd
        # raft_stereo-only — pure-PyTorch neural stereo backend
        raft_repo_dir: "third_party/RAFT-Stereo"
        raft_checkpoint: "models/raftstereo-middlebury.pth"
        raft_iters: 12

    pose_estimator:
        type: "null"        # 'null' | 'dpvo' | 'dpv_slam'
        stride: 2           # DPVO every Nth frame → pose at 30/stride Hz
        patches_per_frame: 96
        # dpv_slam-only — DPVO + loop closure (drift-corrected on revisit)
        loop_closure: true          # proximity loop closure (no extra deps)
        backend_thresh: 64.0
        global_opt_freq: 15         # global optimisation every N keyframes
        classic_loop_closure: false # DBoW2 long-term LC (needs DBoW2 build)

    imu:
        type: "null"        # 'null' | 'synthetic'  (HW backends slot in here)
        rate_hz: 200.0      # typical MEMS rate
        sigma_gyro_n:  1.7e-4   # rad/s/√Hz (BMI088 datasheet)
        sigma_accel_n: 2.0e-3   # m/s²/√Hz
        synthetic_motion: "stationary_with_gravity"
        synthetic_noise_std_accel: 0.0
        synthetic_noise_std_gyro:  0.0
        synthetic_seed: 0

    vio:
        # 16-D error-state EKF (state_estimation/visual_inertial_ekf.py).
        # Predicts with pre-integrated IMU, updates with the visual pose.
        # The visual-scale state is DEFAULT-DISABLED (init_scale_std≈0,
        # scale_random_walk=0) — see the inline comment in default.yaml.
        # When enabled, wraps the visual estimator in VIOPoseEstimator;
        # downstream consumers see the fused pose unchanged.
        enabled: false
        init_position_std_m:        0.10
        init_velocity_std_mps:      0.10
        init_orientation_std_rad:   0.05
        init_bias_gyro_std:         1.0e-3
        init_bias_accel_std:        1.0e-2
        init_scale_std:             1.0e-8     # ≈ 0; freezes s at 1.0
        bias_gyro_random_walk:      1.0e-5
        bias_accel_random_walk:     1.0e-4
        scale_random_walk:          0.0        # 0 ⇒ scale never moves
        visual_position_std_m:      0.05
        visual_orientation_std_rad: 0.02
        zupt_velocity_std:          0.02       # ZUPT measurement 1σ (m/s)
        gravity_w: [0.0, 0.0, -9.81]

    semantic:
        enabled: false
        type: "null"        # 'null' | 'mask2former'
        model: "facebook/mask2former-swin-tiny-cityscapes-semantic"
        device: "cuda"
        dataset: "cityscapes"   # 'cityscapes' | 'ade20k'

    coordinate_frames:
        enabled: false      # use TransformTree for position_world
        root_frame: map
        camera_frame: camera_frame
        static_extrinsics: []   # parent→child SE(3) edges

    occupancy_grid:
        enabled: false      # publish nav_msgs/OccupancyGrid (dynamic obstacles)
        resolution_m: 0.05  # 5 cm per cell (Nav2 default)
        size_x_m: 20.0
        size_y_m: 20.0
        origin_x_m: -10.0
        origin_y_m: -10.0
        default_inflation_m: 0.5
        min_inflation_m: 0.10
        per_class_inflation_m:
            person:     0.40
            bicycle:    0.60
            car:        2.00
            motorcycle: 0.80
            bus:        3.00
            truck:      2.50

    occupancy_3d:
        enabled: false      # publish sparse 3D voxels (PointCloud2 + optional Octomap)
        resolution_m: 0.10  # 10 cm per voxel
        size_z_m: 3.0
        origin_z_m: -0.5    # ceiling at +2.5 m
        per_class_inflation_m:
            person: 0.40
            car:    2.00

    drivable_costmap:
        # Top-down OccupancyGrid produced by IPM-projecting the semantic
        # drivable mask. Reuses occupancy_grid's grid spec so the two
        # layers fuse cleanly in Nav2.
        enabled: false
        use_depth: true     # back-project via dense depth when available
        z_ground_m: 0.0     # assumed ground plane z in world frame

    room_layer:
        # Top-of-graph room hierarchy (morphological-erosion clustering).
        enabled: false
        erosion_m: 0.45             # closes doorways up to 0.9 m wide
        min_area_m2: 1.0
        polygon_simplify_m: 0.05

    stability:
        timeouts_s:                 # per-class memory durations
            DYNAMIC:     1.5
            SEMI_STATIC: 60.0
            # STATIC defaults to inf (never auto-pruned)
        class_overrides: {}         # COCO class → STATIC/SEMI_STATIC/DYNAMIC
        demote_speed_px_s:  100.0   # observed motion → demote to DYNAMIC
        demote_frames:      90      # sustained for N frames
        promote_speed_px_s: 10.0    # stationary → promote DYNAMIC
        promote_frames:     900

    appearance:                     # ReID embedding backend
        type: "null"                # 'null' | 'dinov2'
        model: "facebook/dinov2-small"
        device: "cuda"

    world_map:                      # long-term spatial memory
        enabled: false
        spatial_gate_m: 1.5
        similarity_threshold: 0.75
        allow_spatial_only: true
        # Opt-in eviction (both default 0 = disabled → backwards-compat).
        max_age_s: 0.0              # >0: drops entries unseen this long
        max_entries: 0              # >0: LRU-evicts to keep len ≤ max_entries

    health_monitor:                 # per-stage latency budgets + diagnostics
        enabled: true
        warn_after:  3              # consecutive breaches → WARN
        error_after: 30             # consecutive breaches → ERROR
        stale_after_s: 5.0
        window: 60                  # rolling sample retention
        log_period_s: 5.0
        stage_budgets_ms:
            detector:    12.0
            depth:       20.0
            pose:        25.0
            tracker:     3.0
            scene_graph: 5.0
            appearance:  15.0
            semantic:    40.0
            frame_total: 40.0

Environment variable overrides: `DEVICE=cpu`, `RERUN_ENABLED=false`.

---

## Benchmark scripts

    # MOT17 → YOLO format + fine-tuning
    python3 scripts/mot17_to_yolo.py --mot17 data/MOT17 --out data/mot17_yolo
    python3 scripts/train_detector.py --data data/mot17_yolo/mot17.yaml

    # MOTA / MOTP / IDSW benchmark on MOT17 train split
    python3 scripts/benchmark.py \
        --dataset data/MOT17 --split train --out data/mot17_results

    # Cross-domain detector eval on VisDrone
    python3 scripts/benchmark_visdrone.py \
        --dataset data/VisDrone2019-MOT-train --out data/visdrone_results

    # Depth / ego-pose accuracy on TUM RGB-D (indoor) or CODa (outdoor)
    python3 scripts/eval_dataset.py --dataset tum  \
        --sequence data/rgbd_dataset_freiburg1_room --out data/eval
    python3 scripts/eval_dataset.py --dataset coda \
        --sequence data/coda/seq0 --out data/eval

    # DPVO per-frame latency sweep (resolution × patches × stride)
    python3 scripts/benchmark_dpvo_latency.py

    # Intrinsics calibration from a checkerboard
    python3 scripts/calibrate_camera.py \
        --device 0 --rows 9 --cols 6 \
        --out config/camera_intrinsics.yaml

---

## Project structure

    perception/          Camera interface, detector, depth estimator
                          (Depth Anything V2 / SGBM / RAFT-Stereo),
                          pose estimator (Null + DPVO + DPV-SLAM + VIO),
                          appearance extractor (Null + DINOv2), IMU
                          interface (Null + Synthetic + CODa replay),
                          semantic segmenter (Mask2Former), TransformTree,
                          typed config
    tracking/            ByteTrack, association, motion compensation, Track
    state_estimation/    Kalman Filter, Extended KF, NIS/NEES diagnostics,
                          IMU pre-integration (Forster 2017), visual-
                          inertial EKF (16-D error state, Joseph update,
                          ZUPT, stationary init)
    world_model/         SceneGraph, ObjectState (+ position_world,
                          stability, persistent_id), spatial queries
                          (camera- and world-frame), OccupancyGridBuilder
                          (dynamic obstacle layer), Occupancy3DBuilder
                          (sparse voxel grid), drivable_projector (IPM),
                          RoomLayer (morphological clustering), WorldMap
                          (long-term spatial memory + ReID re-association),
                          SemanticMap (Kimera-Semantics back-end)
    visualization/       Rerun.io logger, OpenCV annotator
    ros2_ws/             ROS2 colcon workspace
                          src/robotics_perception_ros2/
                            camera_publisher_node    Image + CameraInfo
                            detection_node           Detection2DArray
                            tracking_node            tracked Detection2DArray
                            pose_node                Odometry + tf broadcast
                                                     (fused VIO when vio.enabled)
                            depth_node               Image 32FC1 (dense metric depth)
                            scene_graph_node         Detection3DArray
                            occupancy_grid_node      OccupancyGrid (Nav2 costmap-ready)
                            occupancy_3d_node        PointCloud2 + optional Octomap
                            drivable_mask_node       Image mono8 (semantic drivable mask)
                            drivable_costmap_node    OccupancyGrid (IPM-projected
                                                     drivable freespace, Nav2 fusable)
                            semantic_map_node        PointCloud2 XYZRGB (class-coloured
                                                     metric-semantic map)
                            health_monitor_node      DiagnosticArray
                            composite_node           single-process bundle (perf experiment)
    scripts/             Calibration, MOT17/VisDrone benchmark, DPVO
                          latency, TUM/CODa dataset eval, detector training
    third_party/         External clones (DPVO, RAFT-Stereo) — not
                          committed; see setup sections above
    tests/               959 unit tests — all hardware-free; integration
                          tests marked separately
    config/              YAML configuration

---

## Extensions

### Shipped (was deferred in earlier revisions)

**WorldMap eviction.** Opt-in via `world_map.max_age_s` (drops entries
whose `last_seen` is older than `now − max_age_s`) and
`world_map.max_entries` (LRU cap on total entries; re-association
refreshes `last_seen` so revisited entries survive). Both default to
0 (disabled) → backwards-compatible monotonic-growth behaviour. The
"revisited and not seen" eviction (robot returns to a region and
doesn't observe a remembered entry) remains deferred — it needs
sensor-FOV knowledge that belongs in the planner layer.

**ReID appearance features.** DINOv2-small (`facebook/dinov2-small`)
plugged in as a class-agnostic foundation embedding backend.
Embeddings flow into both the per-frame ByteTracker matcher
(StrongSORT-style 0.25 cosine blend) AND the WorldMap re-association
gate. MOT17 ablation is in the README — small but honest positive on
pedestrian-only data; bigger payoff expected on DanceTrack and on
WorldMap revisit scenarios (not yet benchmarked).

**Full IMU-VIO live orchestration.** The 16-D error-state EKF
(`state_estimation/visual_inertial_ekf.py`) is wired through
`VIOPoseEstimator` and selected via the shared
`perception/pose_estimator_factory.py`. Both `launch.py` (standalone)
and the ROS2 `pose_node` use the same factory — `/perception/odom`
and `/tf` publish the fused pose when `vio.enabled=true`. Pre-
integration + bias Jacobians + Joseph update + ZUPT + first-frame
anchor are all live; the CODa replay backend supplies real-IMU
samples for outdoor evaluation. The 16-D scale state and the camera-
IMU body-frame transform are gated off pending the
position↔orientation Jacobian work (see Deferred).

**Neural stereo backend (RAFT-Stereo).** Pure-PyTorch backend slotted
in alongside `StereoSGBMDepthEstimator` under the existing
`DepthEstimator` ABC. `git clone` + `bash download_models.sh` is the
entire setup — no custom CUDA build like DPVO. See "RAFT-Stereo setup"
section above. IGEV-Stereo / FoundationStereo remain candidates if
RAFT's accuracy turns out to be the bottleneck.

**Drivable freespace + IPM projector.** Mask2Former drivable mask
goes through `world_model.drivable_projector.project_drivable_to_grid`
to produce a top-down `nav_msgs/OccupancyGrid`. Two backends behind
the same contract: flat-ground IPM (no depth dep) and depth-aware
back-projection (handles stairs / slopes). Both pipelines (standalone
Rerun + ROS2 graph) emit the costmap.

**SLAM — loop closure (DPV-SLAM).** `DPVSLAMPoseEstimator`
(`perception/dpv_slam_pose_estimator.py`) is DPVO + loop closure:
proximity-based pose-graph loop edges (always on, no extra deps) plus
optional DBoW2 long-term place recognition. Drift is corrected on
revisit instead of accumulating monotonically. Drop-in via
`pose_estimator.type: dpv_slam`; the whole downstream stack
(TransformTree, SceneGraph, ROS2 pose_node) consumes the loop-
corrected pose unchanged. The loop-closure jump is absorbed by the
`map ← odom` transform edge.

**Semantic SLAM (metric-semantic map).** `world_model/semantic_map.py`
ships a persistent voxel-hashed metric-semantic map (`SemanticMap` +
`SemanticVoxel`) — Kimera-Semantics mapping back-end (Rosinol 2020).
Per-voxel occupancy log-odds + per-class vote-fused semantic
distribution. CPU voxel hashing (zero extra VRAM). ROS2 publisher
streams it as an XYZRGB `PointCloud2`; Rerun viewer renders a class-
coloured cloud. Free-space ray-carving and loop-closure map
deformation deferred.

### Deferred

**Position↔orientation Jacobian for the VIO body-frame transform.**
The camera-IMU extrinsic transform is loaded and ready, but the
measurement-model Jacobian `H[0:3, T_IDX] = -[R·p_imu_cam]_× / s`
is not wired in. Without it, every orientation update injects a
fictitious position step via the lever arm — validated as an 8× ATE
regression on CODa seq0 (default-off until fixed). One-flag flip on
`VIOPoseEstimator._apply_body_frame` re-activates the path once the
Jacobian lands. The visual-scale state observability depends on the
same fix.

**Tightly-coupled VIO backbone.** The shipped fuser is loosely-
coupled (consumes the visual estimator's 6-DOF pose as a measurement).
The SOTA accuracy ceiling lives with tightly-coupled VIO — OpenVINS,
VINS-Fusion, ORB-SLAM3+IMU. All slot into the same `PoseEstimator`
ABC.

**Persistent room IDs across frames.** RoomLayer v1 rebuilds rooms
each frame; IDs reshuffle when geometry changes. Persistent IDs need
Hungarian assignment on Jaccard polygon overlap + a SLAM-grade global
ego pose.

**Free-space ray-carving for the semantic map.** Currently voxels
accumulate occupancy hits but never free; the map can't unsee a
parked car that drives away.

---

## References

    Filtering & estimation
    Kalman, R.E. (1960)              Optimal linear filter
    Welch & Bishop (2006)            Kalman filter tutorial
    Bar-Shalom et al. (2001)         Estimation with applications to
                                     tracking and navigation
    Thrun, Burgard & Fox (2005)      Probabilistic Robotics
    Sola (2017)                      Quaternion kinematics for the
                                     error-state KF — arXiv:1711.02508

    Detection & tracking
    Bewley et al. (2016)             SORT — tracking-by-detection
    Zhang et al. (2022)              ByteTrack — arXiv:2110.06864
    Aharon et al. (2022)             BoT-SORT — arXiv:2206.14651
    Du et al. (2023)                 StrongSORT — arXiv:2202.13514
    Sun et al. (2022)                DanceTrack — arXiv:2111.14690

    Visual / depth backbones
    Yang et al. (2024)               Depth Anything V2 — arXiv:2406.09414
    Lipson, Teed & Deng (2021)       RAFT-Stereo — arXiv:2109.07547
    Oquab et al. (2024)              DINOv2 — arXiv:2304.07193
    Cheng et al. (2022)              Mask2Former — arXiv:2112.01527
    Teed, Lipson & Deng (2023)       DPVO — arXiv:2208.04726
    Campos et al. (2021)             ORB-SLAM3 — arXiv:2007.11898

    VIO / mapping
    Forster et al. (2017)            On-Manifold Pre-integration for
                                     VIO — arXiv:1512.02363
    Geneva et al. (2020)             OpenVINS — ICRA 2020
    Rosinol et al. (2020)            Kimera-Semantics — RA-L 2020
    Foote (2013)                     tf: the transform library — ICRA

    Datasets
    Milan et al. (2016)              MOT16 — arXiv:1603.00831
    Sturm et al. (2012)              TUM RGB-D — IROS 2012
    Zhang et al. (2024)              CODa: UT Campus Object Dataset
                                     — IJRR 2024
