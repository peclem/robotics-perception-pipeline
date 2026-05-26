"""
Camera motion compensation for ByteTrack.

Problem
-------
When the camera moves, all tracked objects appear to move in the image
even if they are stationary in the world. Without compensation, the
Kalman filter interprets ego-motion as object motion, corrupting
velocity estimates and causing unnecessary ID switches.

Solution
--------
Estimate the 2D affine homography between consecutive frames using
sparse optical flow on background keypoints. Project each track's
predicted position through that homography to carry it into the
current frame before association.

Algorithm
---------
1. Detect FAST corners in the previous frame (fast, parameter-free)
2. Exclude corners inside tracked bounding boxes (object motion ≠ camera motion)
3. Track corners to the current frame via Lucas-Kanade optical flow
4. Estimate affine homography prev→curr from matched point pairs (RANSAC)
5. Warp each track's state into the current frame: [cx, cy] → H @ [cx, cy, 1]

Reference
---------
Aharon et al. — BoT-SORT: Robust Associations Multi-Pedestrian Tracking (2022)
arXiv:2206.14651 — Section 3.2: Camera Motion Compensation

Robotics context
----------------
Static camera: CMC does nothing (identity homography estimated).
Mobile robot:  CMC removes ego-motion from all track velocity estimates.
               Essential before integrating with a path planner.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tracking.track import Track

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera motion compensator
# ---------------------------------------------------------------------------

class CameraMotionCompensator:
    """
    Estimate and compensate for camera motion between consecutive frames.

    Wraps OpenCV Lucas-Kanade optical flow + affine RANSAC.
    Integrates into ByteTracker.update() before association.

    Parameters
    ----------
    max_corners      : maximum FAST corners to detect per frame
    quality_level    : minimum corner quality (Harris score ratio)
    min_distance     : minimum pixel distance between corners
    ransac_threshold : RANSAC inlier threshold in pixels
    min_inliers      : minimum inliers to accept homography estimate
                       If fewer inliers, identity is returned (no compensation)

    Usage
    -----
    cmc = CameraMotionCompensator()

    # In tracker update loop:
    H = cmc.estimate(prev_frame, curr_frame, tracks)
    if H is not None:
        cmc.apply(tracks, H)

    # Or combined:
    cmc.compensate(prev_frame, curr_frame, tracks)
    """

    def __init__(
        self,
        max_corners:       int   = 1000,
        quality_level:     float = 0.01,
        min_distance:      int   = 1,
        ransac_threshold:  float = 2.0,
        min_inliers:       int   = 10,
    ) -> None:
        self._max_corners      = max_corners
        self._quality_level    = quality_level
        self._min_distance     = min_distance
        self._ransac_threshold = ransac_threshold
        self._min_inliers      = min_inliers

        # LK optical flow parameters
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30, 0.01,
            ),
        )

        self._prev_gray: Optional[np.ndarray] = None
        self._n_compensations = 0
        self._n_failures      = 0

    def compensate(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        tracks:     List[Track],
    ) -> Optional[np.ndarray]:
        """
        Estimate camera homography and apply to all tracks in-place.

        Parameters
        ----------
        prev_frame : previous BGR frame
        curr_frame : current BGR frame
        tracks     : list of Track objects — states updated in-place

        Returns
        -------
        (3, 3) homography matrix, or None if estimation failed.
        """
        H = self.estimate(prev_frame, curr_frame, tracks)
        if H is not None:
            self.apply(tracks, H)
        return H

    def estimate(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        tracks:     List[Track],
    ) -> Optional[np.ndarray]:
        """
        Estimate the affine homography between two frames.

        Parameters
        ----------
        prev_frame : previous BGR frame (H, W, 3)
        curr_frame : current BGR frame (H, W, 3)
        tracks     : current tracks — used to mask object regions

        Returns
        -------
        (3, 3) float64 homography matrix, or None if estimation failed.
        Returns identity matrix when camera is static (no motion detected).
        """
        prev_gray = self._to_gray(prev_frame)
        curr_gray = self._to_gray(curr_frame)

        # Build exclusion mask for object regions
        mask = self._build_background_mask(prev_gray.shape, tracks)

        # Detect corners in background regions of previous frame
        corners = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self._max_corners,
            qualityLevel=self._quality_level,
            minDistance=self._min_distance,
            mask=mask,
        )

        if corners is None or len(corners) < self._min_inliers:
            log.debug(
                "CMC: insufficient corners detected (%d) — skipping",
                len(corners) if corners is not None else 0,
            )
            self._n_failures += 1
            return None

        # Track corners to current frame via LK optical flow
        curr_corners, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray,
            corners, None,
            **self._lk_params,
        )

        # Keep only successfully tracked points
        status  = status.ravel().astype(bool)
        prev_pts = corners[status].reshape(-1, 2)
        curr_pts = curr_corners[status].reshape(-1, 2)

        if len(prev_pts) < self._min_inliers:
            log.debug(
                "CMC: insufficient tracked points (%d) — skipping",
                len(prev_pts),
            )
            self._n_failures += 1
            return None

        # Estimate affine homography with RANSAC
        H_affine, inliers = cv2.estimateAffinePartial2D(
            prev_pts, curr_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self._ransac_threshold,
        )

        if H_affine is None or inliers is None:
            log.debug("CMC: RANSAC failed — skipping")
            self._n_failures += 1
            return None

        n_inliers = int(inliers.sum())
        if n_inliers < self._min_inliers:
            log.debug("CMC: too few inliers (%d) — skipping", n_inliers)
            self._n_failures += 1
            return None

        # Convert 2x3 affine to 3x3 homography
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = H_affine

        self._n_compensations += 1
        log.debug(
            "CMC: estimated homography from %d inliers / %d points",
            n_inliers, len(prev_pts),
        )

        return H

    def apply(self, tracks: List[Track], H: np.ndarray) -> None:
        """
        Warp all track centre positions by the camera homography.

        `H` maps the previous frame to the current one. A track's KF
        state is anchored in the previous frame (it was last associated
        there), so projecting [cx, cy] through `H` — the forward
        transform — carries it into the current frame, cancelling the
        camera's apparent motion before association. Updates the KF
        state in place.

        Applying H⁻¹ instead would displace tracks by the *negative*
        of the camera motion — doubling the apparent drift rather than
        removing it. (That was the original bug; see test_motion_
        compensation.py::test_apply_warps_in_camera_motion_direction.)

        Parameters
        ----------
        tracks : list of Track objects — KF states updated in-place
        H      : (3, 3) homography from previous frame to current frame
        """
        for track in tracks:
            cx, cy = track.position

            # Homogeneous projection through the forward (prev→curr) transform.
            pt   = np.array([cx, cy, 1.0], dtype=np.float64)
            pt_w = H @ pt
            if abs(pt_w[2]) > 1e-9:
                pt_w = pt_w / pt_w[2]   # normalise (no-op for an affine H)

            # Update KF state directly
            track.kf._x[0] = pt_w[0]   # cx
            track.kf._x[1] = pt_w[1]   # cy

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _build_background_mask(
        self,
        shape:  Tuple[int, int],
        tracks: List[Track],
    ) -> np.ndarray:
        """
        Build a binary mask excluding object bounding boxes.
        Corners detected inside object regions may move with the object
        rather than with the camera — excluding them improves CMC accuracy.
        """
        H, W = shape
        mask  = np.ones((H, W), dtype=np.uint8) * 255

        for track in tracks:
            x1, y1, x2, y2 = track.bbox_xyxy
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(W, int(x2))
            y2 = min(H, int(y2))
            mask[y1:y2, x1:x2] = 0

        return mask

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def compensation_rate(self) -> float:
        """Fraction of frames where CMC succeeded."""
        total = self._n_compensations + self._n_failures
        return self._n_compensations / total if total > 0 else 0.0

    @property
    def n_compensations(self) -> int:
        return self._n_compensations

    @property
    def n_failures(self) -> int:
        return self._n_failures

    def reset(self) -> None:
        """Reset state between sequences."""
        self._prev_gray       = None
        self._n_compensations = 0
        self._n_failures      = 0

    def __repr__(self) -> str:
        return (
            f"CameraMotionCompensator("
            f"compensations={self._n_compensations} "
            f"failures={self._n_failures} "
            f"rate={self.compensation_rate:.1%})"
        )
