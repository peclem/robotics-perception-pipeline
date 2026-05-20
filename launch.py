"""
Robotics Perception Pipeline — main entry point.

Usage
-----
    python3 launch.py --source synthetic
    python3 launch.py --source video --input data/test_clip.mp4
    python3 launch.py --source webcam
    RERUN_ENABLED=false python3 launch.py --source synthetic
    python3 launch.py --source synthetic --rerun-save data/recording.rrd
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from perception.camera_interface import (
    CameraFrame,
    SyntheticCamera,
    VideoFileCamera,
    WebcamCamera,
)
from perception.config_loader import load_config, PipelineConfig
from perception.detector import YOLOv8Detector
from tracking.tracker import ByteTracker
from visualization.debug_vis import DebugVisualizer
from world_model.scene_graph import SceneGraph
from perception.depth_estimator import DepthEstimator
from perception.appearance_extractor import (
    AppearanceExtractor, NullAppearanceExtractor,
)
from perception.semantic_segmenter import (
    SemanticSegmenter, NullSemanticSegmenter, drivable_mask,
)
from perception.depth_estimator_factory import build_depth_estimator
from world_model.occupancy_grid import (
    OccupancyGridBuilder, OccupancyGridParams,
)
from world_model.drivable_projector import (
    DrivableProjectorParams, project_drivable_to_grid,
)
from world_model.occupancy_3d import (
    Occupancy3DBuilder, Occupancy3DParams,
)
from world_model.room_layer import RoomLayer, RoomLayerConfig
from world_model.semantic_map import SemanticMap, SemanticMapParams
from perception.health_monitor import HealthMonitor, HealthStatus
from world_model.world_map import WorldMap
from perception.pose_estimator import CameraPose
from perception.imu_interface import IMUInterface
from perception.pose_estimator_factory import (
    build_imu, build_pose_estimator, build_visual_pose_estimator,
)
from perception.transform_tree import TransformTree

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


log = logging.getLogger("launch")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:

    def __init__(
        self,
        cfg:        PipelineConfig,
        source:     str,
        input_path: Optional[str],
    ) -> None:
        self._cfg        = cfg
        self._source     = source
        self._input_path = input_path
        self._running    = False

        self._frame_times: list[float] = []
        self._target_dt = 1.0 / cfg.pipeline.target_hz

        self._camera     = None
        self._detector   = None
        self._tracker    = None
        self._visualizer = None
        self._writer     = None
        self._scene_graph: Optional[SceneGraph] = None
        self._depth_estimator = None
        self._pose_estimator = None
        self._transform_tree: Optional[TransformTree] = None
        self._semantic_map: Optional[SemanticMap] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._open()
        self._running = True

        log.info("Pipeline running — press Ctrl+C to stop.")
        log.info(
            "Source: %s | Device: %s | Target: %.0f Hz",
            self._source,
            self._cfg.detector.device,
            self._cfg.pipeline.target_hz,
        )

        try:
            self._loop()
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — shutting down.")
        finally:
            self._release()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _open(self) -> None:
        cfg = self._cfg
        raw = cfg.as_dict()

        self._camera = self._build_camera(raw)
        self._camera.open()

        log.info(
            "Loading detector: %s on %s ...",
            cfg.detector.model,
            cfg.detector.device,
        )
        self._detector = YOLOv8Detector(raw)
        self._detector.warmup()
        log.info(
            "Detector ready. Mean warmup latency: %.1f ms",
            self._detector.mean_inference_ms,
        )
        self._depth_estimator = self._build_depth_estimator()
        self._pose_estimator = self._build_pose_estimator(raw)

        # Coordinate frame manager. Disabled by default — when enabled,
        # registers static extrinsics from config and is updated each
        # frame from camera_pose. The SceneGraph reads position_world
        # via the tree when present, falling back to direct CameraPose
        # application otherwise.
        self._transform_tree = self._build_transform_tree()

        # Appearance extractor + long-term spatial memory.
        # Both opt-in; default config keeps them off so existing
        # deployments are unaffected.
        self._appearance_extractor = self._build_appearance_extractor()
        self._semantic_segmenter = self._build_semantic_segmenter()
        self._world_map = self._build_world_map()
        # Spatial-memory hierarchy layers — None when their config flag
        # is off; the per-frame loop checks for None and skips the work.
        self._occupancy_grid_builder = self._build_occupancy_grid_builder()
        self._occupancy_3d_builder   = self._build_occupancy_3d_builder()
        self._room_layer             = self._build_room_layer()
        self._semantic_map           = self._build_semantic_map()
        self._drivable_projector_params = self._build_drivable_projector_params()

        # Health monitor: per-stage latency budgets + degraded-mode
        # signalling. Always built (cheap); reporting is enabled when
        # health_monitor.enabled is true in config.
        self._health = self._build_health_monitor()
        self._next_health_log_s = 0.0

        self._tracker    = ByteTracker(raw)
        self._scene_graph = SceneGraph(
            raw,
            transform_tree=self._transform_tree,
            world_map=self._world_map,
        )
        self._visualizer = DebugVisualizer(cfg)
        self._visualizer.connect_rerun()
        self._writer     = None

    def _build_depth_estimator(self) -> DepthEstimator:
        """Delegates to perception.depth_estimator_factory."""
        return build_depth_estimator(self._cfg)

    def _build_occupancy_grid_builder(self) -> Optional[OccupancyGridBuilder]:
        """
        2D obstacle grid. Built when occupancy_grid.enabled OR
        room_layer.enabled (rooms need the 2D grid as input).
        """
        og = self._cfg.occupancy_grid
        if not (og.enabled or self._cfg.room_layer.enabled):
            return None
        params = OccupancyGridParams(
            resolution_m=og.resolution_m,
            size_x_m=og.size_x_m, size_y_m=og.size_y_m,
            origin_x_m=og.origin_x_m, origin_y_m=og.origin_y_m,
            default_inflation_m=og.default_inflation_m,
            min_inflation_m=og.min_inflation_m,
            per_class_inflation_m=dict(og.per_class_inflation_m),
        )
        return OccupancyGridBuilder(params, intrinsics=None)

    def _build_occupancy_3d_builder(self) -> Optional[Occupancy3DBuilder]:
        """Sparse 3D voxel grid. Built when occupancy_3d.enabled."""
        o3 = self._cfg.occupancy_3d
        if not o3.enabled:
            return None
        params = Occupancy3DParams(
            resolution_m=o3.resolution_m,
            size_x_m=o3.size_x_m, size_y_m=o3.size_y_m, size_z_m=o3.size_z_m,
            origin_x_m=o3.origin_x_m, origin_y_m=o3.origin_y_m,
            origin_z_m=o3.origin_z_m,
            default_inflation_m=o3.default_inflation_m,
            min_inflation_m=o3.min_inflation_m,
            per_class_inflation_m=dict(o3.per_class_inflation_m),
        )
        return Occupancy3DBuilder(params, intrinsics=None)

    def _build_room_layer(self) -> Optional[RoomLayer]:
        """3D scene-graph room hierarchy. Built when room_layer.enabled."""
        rl = self._cfg.room_layer
        if not rl.enabled:
            return None
        return RoomLayer(RoomLayerConfig(
            erosion_m=rl.erosion_m,
            min_area_m2=rl.min_area_m2,
            polygon_simplify_m=rl.polygon_simplify_m,
        ))

    def _build_semantic_map(self) -> Optional[SemanticMap]:
        """
        Persistent metric-semantic voxel map (semantic SLAM mapping layer).
        Built when semantic_map.enabled. Integration additionally needs a
        live semantic segmenter, a dense depth map and a pose estimate;
        the per-frame loop skips integration whenever one is missing, so
        this factory only gates on its own flag — but it does warn when
        the segmenter is Null, since the map would then collect geometry
        with no labels.
        """
        sm = self._cfg.semantic_map
        if not sm.enabled:
            return None
        if isinstance(self._semantic_segmenter, NullSemanticSegmenter):
            log.warning(
                "semantic_map.enabled but the semantic segmenter is Null — "
                "the voxel map would collect geometry with no labels. "
                "Enable semantic.* to get a useful map."
            )
        log.info(
            "SemanticMap enabled (voxel=%.2f m, range=%.1f-%.1f m, "
            "stride=%d, max_voxels=%d)",
            sm.voxel_size_m, sm.min_range_m, sm.max_range_m,
            sm.pixel_stride, sm.max_voxels,
        )
        return SemanticMap(SemanticMapParams(
            voxel_size_m=sm.voxel_size_m,
            min_range_m=sm.min_range_m,
            max_range_m=sm.max_range_m,
            pixel_stride=sm.pixel_stride,
            occupancy_hit_logodds=sm.occupancy_hit_logodds,
            occupancy_clamp=sm.occupancy_clamp,
            observation_weight=sm.observation_weight,
            max_voxels=sm.max_voxels,
        ))

    def _build_drivable_projector_params(self) -> Optional[DrivableProjectorParams]:
        """
        DrivableProjectorParams for the standalone drivable-costmap
        layer, built once. Reuses the occupancy_grid block's grid spec
        regardless of occupancy_grid.enabled — that block is the
        canonical grid definition; the `enabled` flag only gates the
        dynamic-obstacle builder, not the spec itself.
        """
        dc = self._cfg.drivable_costmap
        if not dc.enabled:
            return None
        og = self._cfg.occupancy_grid
        log.info(
            "DrivableCostmap enabled (use_depth=%s, z_ground=%.2f, "
            "grid=%dx%d m at %.2f m/cell)",
            dc.use_depth, dc.z_ground_m,
            og.size_x_m, og.size_y_m, og.resolution_m,
        )
        return DrivableProjectorParams(
            grid_params=OccupancyGridParams(
                resolution_m=og.resolution_m,
                size_x_m=og.size_x_m, size_y_m=og.size_y_m,
                origin_x_m=og.origin_x_m, origin_y_m=og.origin_y_m,
            ),
            z_ground_m=dc.z_ground_m,
        )

    def _build_health_monitor(self) -> HealthMonitor:
        """Construct a HealthMonitor with stages from config."""
        hm_cfg = self._cfg.health_monitor
        m = HealthMonitor()
        for name, budget in hm_cfg.stage_budgets_ms.items():
            m.register(
                name, budget_ms=float(budget),
                window=hm_cfg.window,
                warn_after=hm_cfg.warn_after,
                error_after=hm_cfg.error_after,
                stale_after_s=hm_cfg.stale_after_s,
            )
        if hm_cfg.enabled:
            log.info(
                "HealthMonitor enabled (%d stages, log every %.1fs)",
                len(hm_cfg.stage_budgets_ms), hm_cfg.log_period_s,
            )
        return m

    def _build_appearance_extractor(self) -> AppearanceExtractor:
        """Factory for AppearanceExtractor keyed on appearance.type."""
        ap_type = self._cfg.appearance.type
        if ap_type == "null":
            return NullAppearanceExtractor()
        if ap_type == "dinov2":
            from perception.appearance_extractor import DINOv2AppearanceExtractor
            ext = DINOv2AppearanceExtractor(
                model_name=self._cfg.appearance.model,
                device=self._cfg.appearance.device,
            )
            ext.warmup()
            log.info("AppearanceExtractor: %r", ext)
            return ext
        raise ValueError(
            f"Unknown appearance.type={ap_type!r}. Supported: 'null', 'dinov2'."
        )

    def _build_semantic_segmenter(self) -> SemanticSegmenter:
        """Factory for SemanticSegmenter keyed on semantic.type / .enabled."""
        cfg = self._cfg.semantic
        if not cfg.enabled or cfg.type == "null":
            return NullSemanticSegmenter()
        if cfg.type == "mask2former":
            from perception.semantic_segmenter import (
                Mask2FormerSemanticSegmenter,
            )
            seg = Mask2FormerSemanticSegmenter(
                device=cfg.device, model_name=cfg.model, dataset=cfg.dataset,
            )
            seg.warmup()
            log.info("SemanticSegmenter: Mask2Former on %s (dataset=%s)",
                     cfg.device, cfg.dataset)
            return seg
        raise ValueError(
            f"Unknown semantic.type={cfg.type!r}. Supported: 'null', 'mask2former'."
        )

    def _build_world_map(self) -> Optional[WorldMap]:
        """Construct WorldMap from config; returns None when disabled."""
        wm = self._cfg.world_map
        if not wm.enabled:
            return None
        log.info(
            "WorldMap enabled (gate=%.2f m, sim_thr=%.2f, "
            "max_age=%.0f s, max_entries=%d)",
            wm.spatial_gate_m, wm.similarity_threshold,
            wm.max_age_s, wm.max_entries,
        )
        return WorldMap(
            spatial_gate_m=wm.spatial_gate_m,
            similarity_threshold=wm.similarity_threshold,
            allow_spatial_only=wm.allow_spatial_only,
            max_age_s=wm.max_age_s,
            max_entries=wm.max_entries,
        )

    def _build_pose_estimator(self, raw: dict):
        """Delegates to perception.pose_estimator_factory."""
        return build_pose_estimator(self._cfg, raw)

    def _build_visual_pose_estimator(self, raw: dict):
        """Delegates to perception.pose_estimator_factory."""
        return build_visual_pose_estimator(self._cfg, raw)

    def _build_imu(self) -> IMUInterface:
        """Delegates to perception.pose_estimator_factory."""
        return build_imu(self._cfg)

    def _build_transform_tree(self) -> Optional[TransformTree]:
        """Construct TransformTree from config, or None when disabled."""
        cf = self._cfg.coordinate_frames
        if not cf.enabled:
            return None
        tree = TransformTree(root_frame=cf.root_frame)
        for ext in cf.static_extrinsics:
            R = np.asarray(ext.R, dtype=np.float64).reshape(3, 3)
            t = np.asarray(ext.t, dtype=np.float64)
            tree.set_static(parent=ext.parent, child=ext.child, R=R, t=t)
        log.info(
            "TransformTree enabled (root=%s, %d static extrinsics)",
            cf.root_frame, len(cf.static_extrinsics),
        )
        return tree

    def _build_camera(self, raw: dict):
        cfg = self._cfg
        if self._source == "webcam":
            return WebcamCamera(
                raw,
                device_index=cfg.camera.device_index,
                intrinsics_path=cfg.camera.intrinsics_path,
            )
        elif self._source == "video":
            path = self._input_path or cfg.video.input_path
            return VideoFileCamera(
                raw,
                video_path=path,
                intrinsics_path=cfg.camera.intrinsics_path,
                loop=cfg.video.loop,
                playback_fps=cfg.video.playback_fps,
            )
        elif self._source == "synthetic":
            return SyntheticCamera(
                raw,
                width=cfg.synthetic_camera.width,
                height=cfg.synthetic_camera.height,
                num_frames=cfg.synthetic_camera.num_frames,
                fps=cfg.synthetic_camera.fps,
                num_objects=cfg.synthetic_camera.num_objects,
                seed=cfg.synthetic_camera.seed,
            )
        else:
            raise ValueError(
                f"Unknown source: {self._source!r}. "
                "Use: webcam / video / synthetic"
            )

    def _open_writer(self, frame: CameraFrame) -> Optional[cv2.VideoWriter]:
        out_path = (
            self._cfg.pipeline.output_video_path
            or self._cfg.video.output_path
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_path),
            fourcc,
            float(self._cfg.camera.fps),
            (frame.width, frame.height),
        )
        if not writer.isOpened():
            log.warning(
                "Could not open VideoWriter at %s — no output file.", out_path
            )
            return None

        log.info("Writing annotated output to: %s", out_path)
        return writer

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        frame_idx = 0

        while self._running:
            t_frame_start = time.monotonic()

            frame = self._camera.get_frame()
            if frame is None:
                log.info("Source exhausted after %d frames.", frame_idx)
                break

            if frame_idx == 0 and self._writer is None:
                self._writer = self._open_writer(frame)

            with self._health.stage("detector"):
                t0          = time.monotonic()
                detections  = self._detector.detect(frame)
                t_detect    = time.monotonic() - t0

            # Depth estimation
            with self._health.stage("depth"):
                depth_estimates_list = self._depth_estimator.estimate(frame, detections)

            # Per-DETECTION appearance embeddings. Computed upfront so
            # the tracker can blend cosine distance into the per-frame
            # cost matrix (Deep OC-SORT / StrongSORT-style). The same
            # embeddings flow through to the matched tracks via
            # Track.update_embedding inside ByteTracker — no second
            # extraction needed for WorldMap. Gated on whichever
            # consumer is enabled to avoid wasted GPU time.
            det_embeddings = None
            needs_appearance = (
                detections and not isinstance(
                    self._appearance_extractor, NullAppearanceExtractor,
                ) and (
                    self._cfg.tracker.use_appearance
                    or self._world_map is not None
                )
            )
            if needs_appearance:
                with self._health.stage("appearance"):
                    bboxes = [d.bbox_xyxy for d in detections]
                    det_embeddings = self._appearance_extractor.extract(
                        frame.image, bboxes,
                    )

            # Build track_id → depth estimate mapping after tracker update
            # (tracker assigns IDs, then we match detections to tracks by position)
            with self._health.stage("tracker"):
                t0               = time.monotonic()
                confirmed_tracks = self._tracker.update(
                    detections, frame,
                    detection_embeddings=det_embeddings,
                )
                t_track          = time.monotonic() - t0

            depth_map = {}
            for track in confirmed_tracks:
                # Match track to detection by closest centroid
                if detections and depth_estimates_list:
                    track_cx, track_cy = track.position
                    best_idx = None
                    best_dist = float("inf")
                    for det_idx, det in enumerate(detections):
                        x1, y1, x2, y2 = det.bbox_xyxy
                        det_cx = (x1 + x2) / 2.0
                        det_cy = (y1 + y2) / 2.0
                        dist = ((track_cx - det_cx)**2 + (track_cy - det_cy)**2)**0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = det_idx
                    if best_idx is not None and best_dist < 50:
                        depth_map[track.track_id] = depth_estimates_list[best_idx]

            with self._health.stage("pose"):
                camera_pose = self._pose_estimator.estimate(frame)
                if self._transform_tree is not None and camera_pose is not None:
                    self._transform_tree.update_from_camera_pose(
                        camera_pose,
                        camera_frame=self._cfg.coordinate_frames.camera_frame,
                    )

            # Per-track appearance embeddings for WorldMap re-association.
            # The tracker already EMA-updated each matched track's
            # embedding from `det_embeddings` above, so just read it off
            # the confirmed tracks. Tracks with no embedding (never
            # matched to a detection that had one) propagate None and
            # the WorldMap falls back to spatial-only re-association.
            appearance_map = {}
            if self._world_map is not None:
                for tr in confirmed_tracks:
                    appearance_map[tr.track_id] = tr.embedding

            # Semantic segmentation — feeds SceneGraph stability refinement.
            # Skip the work entirely when the backend is the Null one;
            # NullSemanticSegmenter would return None anyway, but the
            # HealthMonitor stage entry would still account a frame budget.
            semantic_mask = None
            if not isinstance(self._semantic_segmenter, NullSemanticSegmenter):
                with self._health.stage("semantic"):
                    semantic_mask = self._semantic_segmenter.segment(frame)

            with self._health.stage("scene_graph"):
                self._scene_graph.update(
                    confirmed_tracks=confirmed_tracks,
                    lost_tracks=self._tracker.lost_tracks,
                    timestamp=frame.timestamp,
                    depth_estimates=depth_map,
                    appearance_embeddings=appearance_map,
                    camera_pose=camera_pose,
                    semantic_mask=semantic_mask,
                )

            # Persistent metric-semantic voxel map (semantic SLAM mapping
            # layer). Folds this frame's dense depth + per-pixel labels +
            # camera world-pose into the global voxel map. Skipped on any
            # frame missing labels, a pose, or a dense depth map — the
            # SemanticMap consumer degrades silently, like the others.
            if (self._semantic_map is not None
                    and semantic_mask is not None
                    and camera_pose is not None):
                with self._health.stage("semantic_map"):
                    # Reuse the dense map the depth stage already produced
                    # (every DepthEstimate carries it); only fall back to
                    # a second forward pass when there were no detections
                    # this frame and the list is therefore empty.
                    dense_depth = next(
                        (e.depth_map for e in depth_estimates_list
                         if e.depth_map is not None),
                        None,
                    )
                    if dense_depth is None:
                        dense_depth = self._depth_estimator.dense_depth_map(
                            frame,
                        )
                    if dense_depth is not None:
                        self._semantic_map.integrate(
                            dense_depth, semantic_mask,
                            camera_pose, frame.intrinsics,
                        )
                    prune_age = self._cfg.semantic_map.prune_age_s
                    if prune_age > 0.0:
                        self._semantic_map.prune(frame.timestamp, prune_age)

            # Spatial-memory layers — fed to Rerun for the showcase.
            # All gated on their own config flags; None when disabled
            # so the visualizer skips the corresponding draw call.
            occ_grid_arr = None
            occ_3d       = None
            rooms        = None
            scene_objects = self._scene_graph.all_objects()
            if self._occupancy_grid_builder is not None:
                occ_grid_arr = self._occupancy_grid_builder.build(scene_objects)
            if self._occupancy_3d_builder is not None:
                occ_3d = self._occupancy_3d_builder.build(scene_objects)
            if self._room_layer is not None and self._occupancy_grid_builder is not None:
                self._room_layer.update(
                    occ_grid_arr, self._occupancy_grid_builder.params,
                    scene_objects, frame_idx,
                )
                rooms = self._room_layer.rooms

            # Drivable freespace costmap. Needs the semantic mask (for
            # the drivable bool image), camera pose (for the world
            # transform), and optionally a depth map (for slope/stair
            # handling). Silently skipped when any of those are missing.
            drivable_costmap = None
            if (self._drivable_projector_params is not None
                    and semantic_mask is not None
                    and camera_pose is not None):
                d_mask = drivable_mask(semantic_mask)
                if d_mask.any():
                    use_depth = self._cfg.drivable_costmap.use_depth
                    depth_arr = next(
                        (e.depth_map for e in depth_estimates_list
                         if e.depth_map is not None),
                        None,
                    ) if use_depth else None
                    # Mask + depth resolutions can disagree if the
                    # segmenter resampled; in that case fall back to
                    # flat-ground rather than raise.
                    if depth_arr is not None and depth_arr.shape != d_mask.shape:
                        depth_arr = None
                    drivable_costmap = project_drivable_to_grid(
                        d_mask, frame.intrinsics, camera_pose,
                        self._drivable_projector_params,
                        depth_map=depth_arr,
                    )

            annotated = self._visualizer.draw(
                frame=frame,
                detections=detections,
                tracks=confirmed_tracks,
                detect_ms=t_detect * 1000,
                track_ms=t_track * 1000,
            )

            if self._writer is not None:
                self._writer.write(annotated)

            fps = (
                1.0 / float(np.mean(self._frame_times[-10:]) or 1.0)
                if self._frame_times else 0.0
            )
            self._visualizer.log_rerun(
                frame=frame,
                detections=detections,
                tracks=confirmed_tracks,
                detect_ms=t_detect * 1000,
                track_ms=t_track * 1000,
                fps=fps,
                n_lost=len(self._tracker.lost_tracks),
                occupancy_3d=occ_3d,
                rooms=rooms,
                semantic_mask=semantic_mask,
                camera_pose=camera_pose,
                scene_objects=scene_objects,
                world_map=self._world_map,
                occupancy_grid_2d=(
                    (occ_grid_arr, self._occupancy_grid_builder.params)
                    if (occ_grid_arr is not None
                        and self._occupancy_grid_builder is not None)
                    else None
                ),
                drivable_costmap=(
                    (drivable_costmap, self._drivable_projector_params.grid_params)
                    if (drivable_costmap is not None
                        and self._drivable_projector_params is not None)
                    else None
                ),
                depth_map=next(
                    (e.depth_map for e in depth_estimates_list
                     if e.depth_map is not None),
                    None,
                ),
                health_monitor=self._health,
                vio_ekf=getattr(self._pose_estimator, "ekf", None),
            )

            t_total = time.monotonic() - t_frame_start
            self._frame_times.append(t_total)
            self._health.get("frame_total").observe(t_total * 1000.0)

            # Periodic health summary — and a WARN log line whenever
            # any stage has escalated to ERROR.
            hm_cfg = self._cfg.health_monitor
            if hm_cfg.enabled and time.monotonic() >= self._next_health_log_s:
                overall = self._health.overall_status()
                if overall == HealthStatus.OK:
                    log.debug("Health OK: %s", self._health.summary_line())
                else:
                    log.warning(
                        "Health %s: %s",
                        overall.name, self._health.summary_line(),
                    )
                self._next_health_log_s = time.monotonic() + hm_cfg.log_period_s

            if frame_idx % 30 == 0:
                fps_log = (
                    1.0 / np.mean(self._frame_times[-30:])
                    if self._frame_times else 0.0
                )
                if self._cfg.depth.enabled:
                    log.info(
                        "Frame %4d | dets=%2d | tracks=%2d | lost=%2d | sg=%2d | "
                        "det=%.1fms | trk=%.1fms | depth=%.1fms | %.1f FPS",
                        frame_idx,
                        len(detections),
                        len(confirmed_tracks),
                        len(self._tracker.lost_tracks),
                        self._scene_graph.n_objects,
                        t_detect * 1000,
                        t_track * 1000,
                        self._depth_estimator.mean_inference_ms,
                        fps_log,
                    )
                else:
                    log.info(
                        "Frame %4d | dets=%2d | tracks=%2d | lost=%2d | sg=%2d | "
                        "det=%.1fms | trk=%.1fms | %.1f FPS",
                        frame_idx,
                        len(detections),
                        len(confirmed_tracks),
                        len(self._tracker.lost_tracks),
                        self._scene_graph.n_objects,
                        t_detect * 1000,
                        t_track * 1000,
                        fps_log,
                    )

            if self._semantic_map is not None and frame_idx % 30 == 0:
                log.info(
                    "SemanticMap | %d voxels | %d frames integrated",
                    len(self._semantic_map),
                    self._semantic_map.frames_integrated,
                )

            frame_idx += 1

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _release(self) -> None:
        self._running = False

        if self._camera is not None:
            self._camera.release()

        if self._writer is not None:
            self._writer.release()
            log.info("Output video saved.")

        if self._visualizer is not None:
            self._visualizer.close()

        self._print_summary()

    def _print_summary(self) -> None:
        if not self._frame_times:
            return
        times = np.array(self._frame_times) * 1000
        log.info("=" * 52)
        log.info("Pipeline summary")
        log.info("  Frames processed : %d",    len(times))
        log.info("  Mean latency     : %.2f ms", times.mean())
        log.info("  P95 latency      : %.2f ms", np.percentile(times, 95))
        log.info("  Max latency      : %.2f ms", times.max())
        log.info("  Mean FPS         : %.1f",    1000.0 / times.mean())
        log.info(
            "  Max track ID     : %d",
            max((t.track_id for t in self._tracker.tracks), default=0),
        )
        log.info("=" * 52)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robotics Perception Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 launch.py --source synthetic
  python3 launch.py --source video --input data/test_clip.mp4
  python3 launch.py --source webcam
  RERUN_ENABLED=false python3 launch.py --source synthetic
  python3 launch.py --source synthetic --rerun-save data/recording.rrd
        """,
    )
    parser.add_argument(
        "--source",
        choices=["webcam", "video", "synthetic"],
        default="synthetic",
        help="Camera source (default: synthetic)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input video path (required when --source video)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config YAML (default: config/default.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output video path from config",
    )
    parser.add_argument(
        "--rerun-save",
        type=str,
        default=None,
        metavar="PATH",
        help="Save .rrd recording instead of streaming "
             "(e.g. data/recording.rrd)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config)
    _setup_logging(cfg.pipeline.log_level)

    if args.output:
        cfg.pipeline.output_video_path = args.output

    if args.rerun_save:
        cfg.visualization.rerun_save_path = args.rerun_save

    if (
        args.source == "video"
        and args.input is None
        and not Path(cfg.video.input_path).exists()
    ):
        log.error(
            "--source video requires --input <path> or a valid "
            "video.input_path in %s",
            args.config,
        )
        sys.exit(1)

    pipeline = Pipeline(cfg, source=args.source, input_path=args.input)

    def _handle_signal(sig, frame):
        log.info("Signal %d received — shutting down.", sig)
        pipeline._running = False

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    pipeline.run()


if __name__ == "__main__":
    main()
