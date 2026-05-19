"""
Rerun.io logger for the robotics perception pipeline.
Written for rerun-sdk 0.31.4 — verified API surface.

API changes vs older versions
------------------------------
connect_tcp()      → connect_grpc()
Scalar(v)          → Scalars(v)
set_time_sequence  → set_time("frame_idx", sequence=n)
set_time_seconds   → set_time("wall_time", duration=t)

Entity hierarchy
----------------
world/camera/image          — RGB camera frame
world/detections/boxes      — raw YOLOv8 output (grey)
world/tracks/boxes          — confirmed KF-filtered tracks (colour-coded)
world/tracks/velocity       — KF velocity vectors (Arrows2D)
world/tracks/covariance     — 2σ position ellipses (LineStrips2D)
world/tracks/history        — past centroid trail per track (LineStrips2D)
world/world_map/visible     — WorldMap entries seen this frame (bright)
world/world_map/remembered  — WorldMap entries not currently visible (faded)
metrics/detect_ms           — detector latency time series
metrics/track_ms            — tracker latency time series
metrics/fps                 — pipeline throughput time series
metrics/n_lost              — lost track count time series
tracks/{id}/nis             — per-track NIS scalar (show_nis only)

WSL2 → Windows connection
--------------------------
Rerun 0.31.4 uses gRPC, not raw TCP.
The viewer listens on port 9876 by default.

To stream live:
  1. Open rerun.exe on Windows (it waits for connections)
  2. python3 launch.py --source video --input data/test_clip.mp4

To save a recording:
  python3 launch.py --source synthetic --rerun-save data/recording.rrd
  Then open recording.rrd with rerun.exe on Windows.
"""

from __future__ import annotations

import colorsys
import logging
import math
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np

from perception.camera_interface import CameraFrame
from perception.config_loader import PipelineConfig
from perception.detector import Detection
from tracking.track import Track

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------

def _track_colour_rgba(track_id: int) -> tuple[int, int, int, int]:
    """Deterministic RGBA colour from track ID (HSV colour wheel)."""
    hue = (track_id * 37) % 360 / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255), 220)


def _room_colour_rgba(room_id: int) -> tuple[int, int, int, int]:
    """
    Deterministic RGBA colour from room ID, offset on the HSV wheel
    away from track colours so the two don't collide visually in the
    Rerun viewer.
    """
    hue = ((room_id * 67) + 180) % 360 / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
    return (int(r * 255), int(g * 255), int(b * 255), 180)


def _ellipse_points(
    cx: float,
    cy: float,
    P2x2: np.ndarray,
    n_points: int = 36,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Compute points along a 2σ covariance ellipse.

    Uses Cholesky decomposition of the 2×2 position covariance submatrix.
    Returns (n_points+1, 2) float32 array — closed loop.
    """
    try:
        L = np.linalg.cholesky(P2x2 + np.eye(2) * 1e-6)
    except np.linalg.LinAlgError:
        L = np.eye(2)

    angles      = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    unit_circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    ellipse     = sigma * (unit_circle @ L.T)
    ellipse[:, 0] += cx
    ellipse[:, 1] += cy

    return np.vstack([ellipse, ellipse[0]]).astype(np.float32)


# ---------------------------------------------------------------------------
# WSL2 host detection
# ---------------------------------------------------------------------------

def _resolve_rerun_host(host: str) -> str:
    """
    Resolve the Rerun viewer host address.
    "auto" → parse WSL2 Windows host IP from /etc/resolv.conf.
    """
    if host != "auto":
        return host

    try:
        with open("/etc/resolv.conf") as f:
            for line in f:
                line = line.strip()
                if line.startswith("nameserver"):
                    ip = line.split()[1]
                    log.debug("WSL2 Windows host IP: %s", ip)
                    return ip
    except (FileNotFoundError, IndexError):
        pass

    log.debug("Could not detect WSL2 host — falling back to 127.0.0.1")
    return "127.0.0.1"


# ---------------------------------------------------------------------------
# RerunLogger
# ---------------------------------------------------------------------------

class RerunLogger:
    """
    Streams pipeline state to the Rerun.io viewer.
    Verified for rerun-sdk 0.31.4.

    Degrades gracefully — all log calls are silent no-ops if the
    viewer is not open or rerun is not installed.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg   = cfg
        self._vis   = cfg.visualization
        self._rr    = None
        self._ready = False
        # Tracks whether a semantic AnnotationContext (class id → name)
        # has been logged yet. Logged once, statically, when the first
        # SemanticMask arrives — Rerun's class-table semantics.
        self._semantic_annotation_logged: bool = False
        self._semantic_class_signature: Optional[tuple] = None
        # Same idea for the camera Pinhole: intrinsics don't change at
        # runtime, so log once when the first frame with a pose arrives.
        self._pinhole_logged: bool = False
        self._pinhole_signature: Optional[tuple] = None
        # Rolling buffer of past camera positions in world frame.
        # 2000 samples ≈ 67 s at 30 Hz of visual updates — long enough
        # for a showcase loop without unbounded memory growth.
        self._ego_trajectory: deque = deque(maxlen=2000)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Initialise Rerun and connect to viewer or open save file.
        Returns True if ready to log.
        """
        if not self._vis.rerun_enabled:
            log.info("Rerun.io disabled via config/env.")
            return False

        try:
            import rerun as rr
        except ImportError:
            log.warning("rerun-sdk not installed. pip install rerun-sdk")
            return False

        self._rr = rr

        try:
            rr.init(self._vis.rerun_app_id, spawn=False)

            if self._vis.rerun_save_path:
                save_path = Path(self._vis.rerun_save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                rr.save(str(save_path))
                log.info("Rerun.io → saving to: %s", save_path)
                log.info(
                    "Open on Windows: rerun.exe %s", save_path
                )
            else:
                host = _resolve_rerun_host(self._vis.rerun_host)
                addr = f"rerun+http://{host}:{self._vis.rerun_port}/proxy"
                rr.connect_grpc(addr)
                log.info("Rerun.io → streaming to %s", addr)
                log.info(
                    "Make sure rerun.exe is open on Windows."
                )

            self._ready = True
            return True

        except Exception as exc:
            log.warning(
                "Rerun.io connection failed: %s\n"
                "  → Use --rerun-save data/recording.rrd to save offline.\n"
                "  → Or set RERUN_ENABLED=false to disable.",
                exc,
            )
            self._ready = False
            return False

    # ------------------------------------------------------------------
    # Per-frame logging
    # ------------------------------------------------------------------

    def log_frame(
        self,
        frame:          CameraFrame,
        detections:     List[Detection],
        tracks:         List[Track],
        occupancy_3d=None,
        rooms=None,
        semantic_mask=None,
        camera_pose=None,
        scene_objects=None,
        world_map=None,
    ) -> None:
        """
        Log all visual data for one frame.

        The new optional kwargs (occupancy_3d, rooms, semantic_mask)
        carry data produced by the corresponding pipeline layers when
        their config flags are enabled. Pass None to skip — the call
        is then identical to the original 3-arg form.
        """
        if not self._ready or self._rr is None:
            return

        rr = self._rr
        try:
            # Timeline — 0.31.4 uses rr.set_time()
            rr.set_time("frame_idx", sequence=frame.frame_idx)
            rr.set_time("wall_time", duration=frame.timestamp)

            # Camera pose first — sets the Pinhole + Transform3D on
            # `world/camera` so the image that follows renders as a 3D
            # frustum in the right place, in the same world as the
            # voxel cloud / rooms / trajectory.
            if camera_pose is not None:
                self._log_camera_pose(rr, frame, camera_pose)

            self._log_image(rr, frame)
            self._log_detections(rr, detections)
            self._log_tracks(rr, tracks)

            # Optional layers — silently skipped when the corresponding
            # pipeline feature is disabled (callers pass None).
            if occupancy_3d is not None:
                self._log_voxels(rr, occupancy_3d)
            if rooms is not None:
                self._log_rooms(rr, rooms)
            if semantic_mask is not None:
                self._log_semantic(rr, semantic_mask)
            if scene_objects is not None:
                self._log_world_tracks(rr, scene_objects)
            if world_map is not None:
                self._log_world_map(rr, world_map, scene_objects)

        except Exception as exc:
            log.debug("Rerun log_frame error (suppressed): %s", exc)

    def _log_image(self, rr, frame: CameraFrame) -> None:
        rgb = frame.image[:, :, ::-1].copy()
        try:
            rr.log("world/camera/image", rr.Image(rgb))
        except Exception as e:
            log.debug("Image log failed: %s", e)

    def _log_detections(self, rr, detections: List[Detection]) -> None:
        if not detections:
            try:
                rr.log("world/detections/boxes", rr.Clear(recursive=False))
            except Exception:
                pass
            return

        boxes  = np.array([d.bbox_xyxy for d in detections], dtype=np.float32)
        labels = [f"{d.class_name} {d.confidence:.2f}" for d in detections]
        colors = [(160, 160, 160, 180)] * len(detections)

        try:
            rr.log(
                "world/detections/boxes",
                rr.Boxes2D(
                    array=boxes,
                    array_format=rr.Box2DFormat.XYXY,
                    labels=labels,
                    colors=colors,
                ),
            )
        except Exception as e:
            log.debug("Detections log failed: %s", e)

    def _log_tracks(self, rr, tracks: List[Track]) -> None:
        if not tracks:
            try:
                rr.log("world/tracks/boxes",      rr.Clear(recursive=False))
                rr.log("world/tracks/velocity",   rr.Clear(recursive=False))
                rr.log("world/tracks/covariance", rr.Clear(recursive=False))
                rr.log("world/tracks/history",    rr.Clear(recursive=False))
            except Exception:
                pass
            return

        boxes  = np.array([t.bbox_xyxy for t in tracks], dtype=np.float32)
        labels = [f"#{t.track_id} {t.class_name}" for t in tracks]
        colors = [_track_colour_rgba(t.track_id) for t in tracks]

        try:
            rr.log(
                "world/tracks/boxes",
                rr.Boxes2D(
                    array=boxes,
                    array_format=rr.Box2DFormat.XYXY,
                    labels=labels,
                    colors=colors,
                ),
            )
        except Exception as e:
            log.debug("Track boxes log failed: %s", e)

        if self._vis.show_velocity:
            self._log_velocity_arrows(rr, tracks)

        if self._vis.show_covariance_ellipse:
            self._log_covariance_ellipses(rr, tracks)

        if self._vis.show_nis:
            self._log_nis(rr, tracks)

        if self._vis.show_track_history:
            self._log_track_history(rr, tracks)

    def _log_velocity_arrows(self, rr, tracks: List[Track]) -> None:
        scale   = self._vis.velocity_arrow_scale
        origins = np.array([t.position     for t in tracks], dtype=np.float32)
        vectors = np.array([t.velocity[:2] * scale for t in tracks], dtype=np.float32)
        colors  = [_track_colour_rgba(t.track_id) for t in tracks]

        magnitudes = np.linalg.norm(vectors, axis=1)
        mask = magnitudes > 1.0
        if not np.any(mask):
            return

        try:
            rr.log(
                "world/tracks/velocity",
                rr.Arrows2D(
                    origins=origins[mask],
                    vectors=vectors[mask],
                    colors=[colors[i] for i, m in enumerate(mask) if m],
                ),
            )
        except Exception as e:
            log.debug("Arrows2D log failed: %s", e)

    def _log_covariance_ellipses(self, rr, tracks: List[Track]) -> None:
        """
        2σ position uncertainty ellipses as LineStrips2D.
        Key visual: ellipse shrinks as the KF converges after updates,
        and grows during predict-only periods (occlusion).
        """
        strips = []
        colors = []

        for track in tracks:
            P2     = track.covariance[:2, :2]
            cx, cy = track.position
            pts    = _ellipse_points(cx, cy, P2, n_points=36, sigma=2.0)
            strips.append(pts)
            colors.append(_track_colour_rgba(track.track_id))

        if not strips:
            return

        try:
            rr.log(
                "world/tracks/covariance",
                rr.LineStrips2D(strips, colors=colors),
            )
        except Exception as e:
            log.debug("LineStrips2D log failed: %s", e)

    def _log_nis(self, rr, tracks: List[Track]) -> None:
        for track in tracks:
            if not math.isnan(track.nis):
                try:
                    # 0.31.4: Scalars (plural) not Scalar
                    rr.log(
                        f"tracks/{track.track_id}/nis",
                        rr.Scalars(float(track.nis)),
                    )
                except Exception as e:
                    log.debug("NIS log failed track %d: %s", track.track_id, e)

    def _log_track_history(self, rr, tracks: List[Track]) -> None:
        """
        Past-position trail for each track, drawn as a LineStrips2D in
        image space. The trail is taken from `track.history` — the
        sequence of KFSnapshot objects appended on each KF update — and
        truncated to the most recent `track_history_max_points` so
        long-running tracks don't bloat the viewer.

        Tracks with fewer than 2 history entries are skipped (a
        single-point polyline is not meaningful).
        """
        max_pts = int(self._vis.track_history_max_points)
        strips: list = []
        colors: list = []
        for track in tracks:
            hist = getattr(track, "history", None)
            if not hist or len(hist) < 2:
                continue
            recent = list(hist)[-max_pts:]
            pts = np.array(
                [snap.state[:2] for snap in recent], dtype=np.float32,
            )
            strips.append(pts)
            colors.append(_track_colour_rgba(track.track_id))

        if not strips:
            try:
                rr.log("world/tracks/history", rr.Clear(recursive=False))
            except Exception:
                pass
            return

        try:
            rr.log(
                "world/tracks/history",
                rr.LineStrips2D(strips, colors=colors),
            )
        except Exception as e:
            log.debug("Track history log failed: %s", e)

    # ------------------------------------------------------------------
    # New showcase layers (voxels, rooms, semantic mask)
    # ------------------------------------------------------------------

    def _log_voxels(self, rr, occupancy_3d) -> None:
        """
        Log occupied voxel centres as a 3D point cloud. The point
        radius is set to half the grid resolution so each voxel renders
        as roughly its physical size in the Rerun viewer.
        """
        centres = occupancy_3d.voxel_centres_world()
        if centres.shape[0] == 0:
            try:
                rr.log("world/occupancy_3d", rr.Clear(recursive=False))
            except Exception:
                pass
            return
        try:
            res = float(occupancy_3d.params.resolution_m)
            rr.log(
                "world/occupancy_3d",
                rr.Points3D(
                    positions=centres.astype(np.float32),
                    radii=res * 0.5,
                    colors=[(220, 60, 60, 200)] * centres.shape[0],
                ),
            )
        except Exception as e:
            log.debug("Voxel log failed: %s", e)

    def _log_rooms(self, rr, rooms) -> None:
        """
        Log room polygons as closed line strips in world XY at z=0,
        plus a Points3D entity carrying the room labels at each
        centroid. Lifting to 3D matches the voxel-cloud frame so
        rooms render correctly relative to the 3D occupancy.
        """
        if not rooms:
            try:
                rr.log("world/rooms", rr.Clear(recursive=True))
            except Exception:
                pass
            return
        try:
            strips = []
            labels = []
            colors = []
            centroids = []
            for r in rooms:
                # Close the polygon by appending the first vertex at the end.
                poly = np.concatenate(
                    [r.polygon_world, r.polygon_world[:1]], axis=0,
                )
                strip3d = np.column_stack(
                    [poly, np.zeros(poly.shape[0])]
                ).astype(np.float32)
                strips.append(strip3d)
                labels.append(r.label)
                colors.append(_room_colour_rgba(r.room_id))
                centroids.append([
                    float(r.centroid[0]), float(r.centroid[1]), 0.1,
                ])
            rr.log(
                "world/rooms/polygons",
                rr.LineStrips3D(
                    strips, colors=colors, radii=0.02,
                ),
            )
            rr.log(
                "world/rooms/labels",
                rr.Points3D(
                    positions=np.asarray(centroids, dtype=np.float32),
                    labels=labels, colors=colors, radii=0.05,
                ),
            )
        except Exception as e:
            log.debug("Rooms log failed: %s", e)

    def _log_semantic(self, rr, semantic_mask) -> None:
        """
        Log the semantic class map as a Rerun SegmentationImage,
        co-located with the camera image. The class id→name table is
        logged once via AnnotationContext (Rerun's class-table
        primitive) and re-logged only if the segmenter swaps to a
        different vocabulary (e.g. cityscapes → ade20k mid-run).
        """
        try:
            signature = tuple(sorted(semantic_mask.class_names.items()))
            if (not self._semantic_annotation_logged
                    or signature != self._semantic_class_signature):
                # Build the annotation context. Newer rerun versions:
                # rr.AnnotationContext(class_descriptions=[(id, label, color), …]).
                # 0.31.4 accepts a list of (id, label) tuples directly.
                ctx = [(int(cid), str(name))
                       for cid, name in semantic_mask.class_names.items()]
                rr.log(
                    "world/camera/semantic",
                    rr.AnnotationContext(ctx),
                    static=True,
                )
                self._semantic_annotation_logged = True
                self._semantic_class_signature = signature

            rr.log(
                "world/camera/semantic",
                rr.SegmentationImage(semantic_mask.mask),
            )
        except Exception as e:
            log.debug("Semantic log failed: %s", e)

    def _log_camera_pose(self, rr, frame, camera_pose) -> None:
        """
        Position the camera entity in the world frame and emit its
        intrinsics. With Pinhole + Transform3D on `world/camera`, Rerun
        renders the image as a 3D frustum coinciding with the camera's
        location, so it sits naturally alongside the voxel cloud and
        room polygons.

        Also accumulates the camera's world position into a rolling
        trajectory buffer and re-emits it as a Polyline3D each frame
        — the visible "ego path" line in the showcase.
        """
        intr = frame.intrinsics
        # Pinhole is static (intrinsics don't change at runtime). Log
        # once, and only re-log if the resolution actually changes
        # (e.g. mid-run camera swap).
        signature = (float(intr.fx), float(intr.fy),
                     float(intr.cx), float(intr.cy),
                     int(intr.width), int(intr.height))
        try:
            if not self._pinhole_logged or signature != self._pinhole_signature:
                rr.log(
                    "world/camera",
                    rr.Pinhole(
                        focal_length=[float(intr.fx), float(intr.fy)],
                        principal_point=[float(intr.cx), float(intr.cy)],
                        resolution=[int(intr.width), int(intr.height)],
                    ),
                    static=True,
                )
                self._pinhole_logged = True
                self._pinhole_signature = signature
        except Exception as e:
            log.debug("Pinhole log failed: %s", e)

        # CameraPose stores R (world ← camera) and t (camera origin in
        # world). That's exactly the entity-to-parent transform Rerun
        # wants on a child entity, so the default `from_parent=False`
        # is correct — no axis flipping needed.
        try:
            rr.log(
                "world/camera",
                rr.Transform3D(
                    translation=camera_pose.t.astype(np.float32),
                    mat3x3=camera_pose.R.astype(np.float32),
                ),
            )
        except Exception as e:
            log.debug("Transform3D log failed: %s", e)

        # Accumulate trajectory + emit polyline (kept in sync regardless
        # of whether the transform log succeeded).
        self._ego_trajectory.append(camera_pose.t.copy().astype(np.float32))
        self._log_ego_trajectory(rr)

    def _log_ego_trajectory(self, rr) -> None:
        """Emit the rolling camera-position polyline at world/ego_trajectory."""
        if len(self._ego_trajectory) < 2:
            return
        try:
            pts = np.asarray(self._ego_trajectory, dtype=np.float32)
            rr.log(
                "world/ego_trajectory",
                rr.LineStrips3D(
                    [pts],
                    colors=[(60, 220, 60, 220)],
                    radii=0.015,
                ),
            )
        except Exception as e:
            log.debug("Trajectory log failed: %s", e)

    def _log_world_tracks(self, rr, objects) -> None:
        """
        Emit tracked-object world positions as 3D markers at
        `world/scene/objects`. Each marker carries a label of the
        form "#track_id class" (with "p:persistent_id" appended when
        WorldMap re-association has assigned one) and the same
        deterministic colour as its 2D bbox in image space.

        Objects without `position_world` are silently skipped —
        typically the case when no depth or ego-pose is available.
        """
        if not objects:
            try:
                rr.log("world/scene/objects", rr.Clear(recursive=False))
            except Exception:
                pass
            return
        positions: list = []
        labels:    list = []
        colors:    list = []
        for obj in objects:
            if obj.position_world is None or obj.position_world.shape[0] < 3:
                continue
            positions.append(np.asarray(obj.position_world[:3],
                                        dtype=np.float32))
            label = f"#{obj.track_id} {obj.class_name}"
            if obj.persistent_id is not None:
                label += f" p:{obj.persistent_id}"
            labels.append(label)
            colors.append(_track_colour_rgba(obj.track_id))
        if not positions:
            try:
                rr.log("world/scene/objects", rr.Clear(recursive=False))
            except Exception:
                pass
            return
        try:
            rr.log(
                "world/scene/objects",
                rr.Points3D(
                    positions=np.asarray(positions, dtype=np.float32),
                    labels=labels, colors=colors, radii=0.08,
                ),
            )
        except Exception as e:
            log.debug("World tracks log failed: %s", e)

    def _log_world_map(self, rr, world_map, scene_objects) -> None:
        """
        Render the long-term spatial memory (WorldMap) as two layers:

            world/world_map/visible    — entries whose persistent_id is
                                         also in `scene_objects` this
                                         frame. Bright, larger markers.
            world/world_map/remembered — entries NOT currently visible.
                                         Faded grey markers, smaller.

        Labels carry "p:{persistent_id} {class}  n=k" so the showcase
        viewer can read off how many times each entry has been
        re-associated. Empty layers emit Clear, matching the rest of
        the logger.
        """
        try:
            entries = world_map.all_entries()
        except Exception:
            entries = []

        visible_ids: set = set()
        if scene_objects:
            for obj in scene_objects:
                pid = getattr(obj, "persistent_id", None)
                if pid is not None:
                    visible_ids.add(int(pid))

        vis_pos:   list = []
        vis_lab:   list = []
        vis_col:   list = []
        rem_pos:   list = []
        rem_lab:   list = []
        rem_col:   list = []
        for e in entries:
            pos = np.asarray(e.position_world, dtype=np.float32)
            if pos.shape[0] < 3:
                continue
            pid = int(e.persistent_id)
            label = f"p:{pid} {e.class_name} n={e.n_observations}"
            if pid in visible_ids:
                vis_pos.append(pos[:3])
                vis_lab.append(label)
                vis_col.append(_track_colour_rgba(pid))
            else:
                rem_pos.append(pos[:3])
                rem_lab.append(label)
                # Faded grey — same hue family for all "remembered"
                # entries so the eye reads them as one class.
                rem_col.append((160, 160, 200, 110))

        def _emit(path: str, pos, lab, col, radii: float) -> None:
            if not pos:
                try:
                    rr.log(path, rr.Clear(recursive=False))
                except Exception:
                    pass
                return
            try:
                rr.log(
                    path,
                    rr.Points3D(
                        positions=np.asarray(pos, dtype=np.float32),
                        labels=lab, colors=col, radii=radii,
                    ),
                )
            except Exception as e:
                log.debug("WorldMap %s log failed: %s", path, e)

        _emit("world/world_map/visible",    vis_pos, vis_lab, vis_col, 0.11)
        _emit("world/world_map/remembered", rem_pos, rem_lab, rem_col, 0.06)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def log_metrics(
        self,
        detect_ms: float,
        track_ms:  float,
        fps:       float,
        n_lost:    int,
    ) -> None:
        """Log pipeline performance as Rerun scalar time series."""
        if not self._ready or self._rr is None:
            return

        rr = self._rr
        scalars = {
            "metrics/detect_ms": detect_ms,
            "metrics/track_ms":  track_ms,
            "metrics/fps":       fps,
            "metrics/n_lost":    float(n_lost),
        }
        for path, value in scalars.items():
            try:
                # 0.31.4: rr.Scalars (plural)
                rr.log(path, rr.Scalars(float(value)))
            except Exception as e:
                log.debug("Scalars log failed %s: %s", path, e)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._ready = False
        log.debug("RerunLogger closed.")

    @property
    def is_ready(self) -> bool:
        return self._ready
