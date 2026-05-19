"""
OpenCV frame annotator + Rerun.io dispatcher.

draw()          — annotate frame, return BGR array for VideoWriter
connect_rerun() — connect RerunLogger (call once at startup)
log_rerun()     — delegate to RerunLogger
close()         — release resources

Never calls cv2.imshow() — headless WSL compatible.
"""

from __future__ import annotations

import colorsys
import logging
import math
from typing import List

import cv2
import numpy as np

from perception.camera_interface import CameraFrame
from perception.config_loader import PipelineConfig
from perception.detector import Detection
from tracking.track import Track
from visualization.rerun_logger import RerunLogger, _ellipse_points

log = logging.getLogger(__name__)


def _bgr(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR colour from track ID (OpenCV uses BGR)."""
    hue = (track_id * 37) % 360 / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


class DebugVisualizer:
    """
    Frame annotator and Rerun.io dispatcher.

    Annotation draw order (back to front)
    --------------------------------------
    1. Raw detection boxes — light grey, 1px
    2. Track bounding boxes — colour-coded, thick
    3. Track ID / class / confidence label
    4. 2σ covariance ellipse — colour-coded, 1px
    5. Velocity arrow — colour-coded
    6. NIS label — optional, small text
    7. Stats overlay — top-left corner
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg  = cfg
        self._vis  = cfg.visualization
        self._rlog = RerunLogger(cfg)

    def connect_rerun(self) -> bool:
        """Connect to Rerun viewer. Call once after construction."""
        return self._rlog.connect()

    # ------------------------------------------------------------------
    # OpenCV annotator
    # ------------------------------------------------------------------

    def draw(
        self,
        frame:      CameraFrame,
        detections: List[Detection],
        tracks:     List[Track],
        detect_ms:  float = 0.0,
        track_ms:   float = 0.0,
    ) -> np.ndarray:
        """
        Annotate frame and return the BGR image.
        Never modifies frame.image in-place — works on a copy.
        """
        img = frame.image.copy()

        if self._vis.show_bboxes:
            self._draw_raw_detections(img, detections)
            self._draw_tracks(img, tracks)

        if self._vis.show_covariance_ellipse:
            self._draw_ellipses(img, tracks)

        if self._vis.show_velocity:
            self._draw_velocity_arrows(img, tracks)

        if self._vis.show_stats_overlay:
            self._draw_stats(img, detections, tracks, detect_ms, track_ms)

        return img

    def _draw_raw_detections(
        self, img: np.ndarray, detections: List[Detection]
    ) -> None:
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (140, 140, 140), 1)

    def _draw_tracks(
        self, img: np.ndarray, tracks: List[Track]
    ) -> None:
        t = self._vis.bbox_thickness
        for track in tracks:
            col         = _bgr(track.track_id)
            x1, y1, x2, y2 = track.bbox_xyxy.astype(int)

            cv2.rectangle(img, (x1, y1), (x2, y2), col, t)

            if self._vis.show_track_ids:
                label = f"#{track.track_id} {track.class_name} {track.score:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                )
                cv2.rectangle(
                    img,
                    (x1, max(0, y1 - th - 6)),
                    (x1 + tw + 4, y1),
                    col, -1,
                )
                cv2.putText(
                    img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

            if self._vis.show_nis and not math.isnan(track.nis):
                cv2.putText(
                    img, f"NIS:{track.nis:.1f}",
                    (x1 + 2, y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    col, 1, cv2.LINE_AA,
                )

    def _draw_ellipses(
        self, img: np.ndarray, tracks: List[Track]
    ) -> None:
        """
        2σ covariance ellipses via Cholesky — same math as RerunLogger
        so both outputs are always identical.
        """
        for track in tracks:
            col      = _bgr(track.track_id)
            P2       = track.covariance[:2, :2]
            cx, cy   = track.position
            pts      = _ellipse_points(cx, cy, P2, n_points=36, sigma=2.0)
            pts_int  = pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                img, [pts_int], isClosed=True,
                color=col, thickness=1, lineType=cv2.LINE_AA,
            )

    def _draw_velocity_arrows(
        self, img: np.ndarray, tracks: List[Track]
    ) -> None:
        scale = self._vis.velocity_arrow_scale
        for track in tracks:
            col    = _bgr(track.track_id)
            cx, cy = track.position.astype(int)
            vx, vy = track.velocity[:2]
            ex     = int(cx + vx * scale)
            ey     = int(cy + vy * scale)
            if abs(ex - cx) < 2 and abs(ey - cy) < 2:
                continue
            cv2.arrowedLine(
                img, (cx, cy), (ex, ey),
                col, 2, cv2.LINE_AA, tipLength=0.3,
            )

    def _draw_stats(
        self,
        img:        np.ndarray,
        detections: List[Detection],
        tracks:     List[Track],
        detect_ms:  float,
        track_ms:   float,
    ) -> None:
        lines = [
            f"Detections : {len(detections)}",
            f"Tracks     : {len(tracks)}",
            f"Detect     : {detect_ms:.1f} ms",
            f"Track      : {track_ms:.1f} ms",
            f"Total      : {detect_ms + track_ms:.1f} ms",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                img, line, (10, 20 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 220, 60), 1, cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # Rerun delegation
    # ------------------------------------------------------------------

    def log_rerun(
        self,
        frame:          CameraFrame,
        detections:     List[Detection],
        tracks:         List[Track],
        detect_ms:      float = 0.0,
        track_ms:       float = 0.0,
        fps:            float = 0.0,
        n_lost:         int   = 0,
        occupancy_3d=None,
        rooms=None,
        semantic_mask=None,
        camera_pose=None,
        scene_objects=None,
        world_map=None,
        occupancy_grid_2d=None,
        depth_map=None,
    ) -> None:
        self._rlog.log_frame(
            frame, detections, tracks,
            occupancy_3d=occupancy_3d, rooms=rooms,
            semantic_mask=semantic_mask, camera_pose=camera_pose,
            scene_objects=scene_objects, world_map=world_map,
            occupancy_grid_2d=occupancy_grid_2d,
            depth_map=depth_map,
        )
        self._rlog.log_metrics(detect_ms, track_ms, fps, n_lost)

    def close(self) -> None:
        self._rlog.close()
