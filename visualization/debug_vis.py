"""
Debug visualizer — OpenCV frame annotator + optional Rerun.io logger.

Draws on every confirmed track:
  - Bounding box (colour-coded by track ID)
  - Track ID label
  - Velocity vector arrow (from KF state)
  - 2σ covariance ellipse (position uncertainty)
  - Stats overlay (FPS, det count, track count)

WSL note
--------
cv2.imshow() is NOT called — this module never opens a window.
Annotated frames are returned as numpy arrays for VideoWriter.

Rerun.io
--------
The Windows viewer connects automatically when launched before the
pipeline. Download from https://rerun.io/docs/getting-started/installing-viewer
Then: rerun  (opens viewer, waits for connections)
The WSL process connects via TCP on the default port.

Upgrade path
------------
Step 8: Replace draw() with a richer Rerun.io-first logger.
Step 11: Add world model scene graph logging.
"""

from __future__ import annotations

import colorsys
import logging
from typing import List, Optional

import cv2
import numpy as np

from perception.camera_interface import CameraFrame
from perception.config_loader import PipelineConfig
from perception.detector import Detection
from tracking.track import Track

log = logging.getLogger(__name__)


def _track_colour(track_id: int) -> tuple[int, int, int]:
    """
    Deterministic BGR colour from track ID.
    Uses HSV colour wheel so adjacent IDs have distinct colours.
    """
    hue = (track_id * 37) % 360 / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


class DebugVisualizer:
    """
    Annotates frames and optionally streams to Rerun.io.

    Parameters
    ----------
    cfg : PipelineConfig — reads visualization section
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg      = cfg
        self._vis      = cfg.visualization
        self._rerun_ok = False
        self._rr       = None

        if self._vis.rerun_enabled:
            self._init_rerun()

    # ------------------------------------------------------------------
    # Rerun initialisation
    # ------------------------------------------------------------------

    def _init_rerun(self) -> None:
        try:
            import rerun as rr
            rr.init(self._vis.rerun_app_id, spawn=False)
            rr.connect_tcp()  # connects to viewer running on Windows host
            self._rr       = rr
            self._rerun_ok = True
            log.info("Rerun.io connected. Open the viewer on Windows to see the stream.")
        except Exception as exc:
            log.warning(
                "Rerun.io init failed (%s). "
                "Continuing without Rerun — set rerun_enabled: false to suppress this.",
                exc,
            )
            self._rerun_ok = False

    # ------------------------------------------------------------------
    # OpenCV annotator
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: CameraFrame,
        detections: List[Detection],
        tracks: List[Track],
        detect_ms: float = 0.0,
        track_ms:  float = 0.0,
    ) -> np.ndarray:
        """
        Annotate a frame with detections and tracks.

        Parameters
        ----------
        frame       : source CameraFrame
        detections  : raw detections from the detector (this frame)
        tracks      : confirmed tracks from the tracker (this frame)
        detect_ms   : detector latency in milliseconds (for overlay)
        track_ms    : tracker latency in milliseconds (for overlay)

        Returns
        -------
        Annotated BGR image as a numpy array — same shape as frame.image.
        """
        img = frame.image.copy()

        if self._vis.show_bboxes:
            self._draw_detections(img, detections)
            self._draw_tracks(img, tracks)

        if self._vis.show_covariance_ellipse:
            self._draw_covariance_ellipses(img, tracks)

        if self._vis.show_velocity:
            self._draw_velocity_arrows(img, tracks)

        if self._vis.show_stats_overlay:
            self._draw_stats(img, detections, tracks, detect_ms, track_ms)

        return img

    def _draw_detections(self, img: np.ndarray, detections: List[Detection]) -> None:
        """Draw raw detection boxes in light grey — shows what the detector sees."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (160, 160, 160), 1)

    def _draw_tracks(self, img: np.ndarray, tracks: List[Track]) -> None:
        """Draw confirmed track boxes with track-ID-coded colours."""
        thickness = self._vis.bbox_thickness
        for track in tracks:
            colour = _track_colour(track.track_id)
            x1, y1, x2, y2 = track.bbox_xyxy.astype(int)

            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)

            if self._vis.show_track_ids:
                label = f"ID:{track.track_id} {track.class_name} {track.score:.2f}"
                # Background for readability
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), colour, -1)
                cv2.putText(
                    img, label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

            if self._vis.show_nis and not np.isnan(track.nis):
                nis_label = f"NIS:{track.nis:.1f}"
                cv2.putText(
                    img, nis_label,
                    (x1 + 2, y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    colour, 1, cv2.LINE_AA,
                )

    def _draw_covariance_ellipses(self, img: np.ndarray, tracks: List[Track]) -> None:
        """
        Draw 2σ position uncertainty ellipse from the KF covariance.

        The ellipse is computed from the 2×2 position submatrix of P.
        Its axes are the eigenvectors of P[:2,:2], scaled by 2*sqrt(eigenvalue)
        to represent the 2-sigma confidence region.

        A tightly-converged filter → small ellipse.
        A freshly-born track or long occlusion → large ellipse.
        This is the key visual that shows the KF is doing real work.
        """
        for track in tracks:
            colour = _track_colour(track.track_id)
            P2 = track.covariance[:2, :2]  # position submatrix

            try:
                eigenvalues, eigenvectors = np.linalg.eigh(P2)
            except np.linalg.LinAlgError:
                continue

            # Eigenvalues can be very slightly negative due to float precision
            eigenvalues = np.maximum(eigenvalues, 0.0)
            axes = (
                max(int(2.0 * np.sqrt(eigenvalues[0])), 1),
                max(int(2.0 * np.sqrt(eigenvalues[1])), 1),
            )

            # Rotation angle from principal eigenvector
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

            cx, cy = track.position.astype(int)
            cv2.ellipse(
                img,
                (cx, cy),
                axes,
                angle,
                0, 360,
                colour,
                1,
                cv2.LINE_AA,
            )

    def _draw_velocity_arrows(self, img: np.ndarray, tracks: List[Track]) -> None:
        """
        Draw KF-estimated velocity as an arrow from the track centre.
        Arrow length = velocity magnitude × velocity_arrow_scale.
        A stationary track → no visible arrow.
        A fast-moving track → long arrow in direction of motion.
        """
        scale = self._vis.velocity_arrow_scale
        for track in tracks:
            colour  = _track_colour(track.track_id)
            cx, cy  = track.position.astype(int)
            vx, vy  = track.velocity[:2]  # centre velocity only

            end_x = int(cx + vx * scale)
            end_y = int(cy + vy * scale)

            if abs(end_x - cx) < 2 and abs(end_y - cy) < 2:
                continue  # don't draw sub-pixel arrows

            cv2.arrowedLine(
                img,
                (cx, cy), (end_x, end_y),
                colour, 2, cv2.LINE_AA,
                tipLength=0.3,
            )

    def _draw_stats(
        self,
        img: np.ndarray,
        detections: List[Detection],
        tracks: List[Track],
        detect_ms: float,
        track_ms: float,
    ) -> None:
        """Draw a stats overlay in the top-left corner."""
        lines = [
            f"Detections : {len(detections)}",
            f"Tracks     : {len(tracks)}",
            f"Detect     : {detect_ms:.1f} ms",
            f"Track      : {track_ms:.1f} ms",
            f"Total      : {detect_ms + track_ms:.1f} ms",
        ]
        y0, dy = 20, 18
        for i, line in enumerate(lines):
            y = y0 + i * dy
            cv2.putText(
                img, line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 0), 1, cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # Rerun.io logger
    # ------------------------------------------------------------------

    def log_rerun(
        self,
        frame: CameraFrame,
        detections: List[Detection],
        tracks: List[Track],
    ) -> None:
        """
        Stream data to Rerun.io viewer.
        No-op if Rerun is disabled or failed to connect.
        """
        if not self._rerun_ok or self._rr is None:
            return

        rr = self._rr
        try:
            rr.set_time_sequence("frame", frame.frame_idx)
            rr.set_time_seconds("time", frame.timestamp)

            # Raw camera image
            rr.log("camera/image", rr.Image(frame.image[:, :, ::-1]))  # BGR→RGB

            # Raw detections
            if detections:
                det_boxes = np.array([d.bbox_xyxy for d in detections], dtype=np.float32)
                det_labels = [f"{d.class_name} {d.confidence:.2f}" for d in detections]
                rr.log(
                    "detections/boxes",
                    rr.Boxes2D(
                        array=det_boxes,
                        array_format=rr.Box2DFormat.XYXY,
                        labels=det_labels,
                    ),
                )

            # Confirmed tracks
            if tracks:
                track_boxes  = np.array([t.bbox_xyxy for t in tracks], dtype=np.float32)
                track_labels = [f"ID:{t.track_id} {t.class_name}" for t in tracks]
                track_ids    = [t.track_id for t in tracks]
                rr.log(
                    "tracks/boxes",
                    rr.Boxes2D(
                        array=track_boxes,
                        array_format=rr.Box2DFormat.XYXY,
                        labels=track_labels,
                        class_ids=track_ids,
                    ),
                )

                # Scalar timeseries
                rr.log("stats/n_confirmed", rr.Scalar(len(tracks)))
                rr.log("stats/n_detections", rr.Scalar(len(detections)))

        except Exception as exc:
            # Never let a Rerun failure crash the pipeline
            log.debug("Rerun log error (suppressed): %s", exc)
            self._rerun_ok = False

    def close(self) -> None:
        """Clean up resources."""
        pass  # Rerun SDK manages its own connection lifecycle
