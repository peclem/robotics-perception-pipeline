"""
ByteTracker — multi-object tracker using two-stage IoU association.

Algorithm (per-frame)
---------------------
1.  Split detections into D_high (conf ≥ high_thresh) and
    D_low (low_thresh ≤ conf < high_thresh).

2.  Predict all non-REMOVED tracks forward by dt seconds using
    their embedded Kalman Filters.

3.  Stage 1 — Associate D_high with ALL tracks (TENTATIVE +
    CONFIRMED + LOST) using IoU cost matrix + Hungarian algorithm.
    Matched tracks are updated. Unmatched D_high become new
    TENTATIVE tracks.

4.  Stage 2 — Associate D_low with tracks NOT matched in Stage 1.
    Low-confidence detections rescue partially-occluded tracks that
    the detector could only weakly see. They never create new tracks.

5.  Mark unmatched tracks:
    TENTATIVE  → REMOVED  (never confirmed, no second chance)
    CONFIRMED  → LOST     (seen before, give it max_age frames)
    LOST       → n_misses++, REMOVED if n_misses > max_age

6.  Promote TENTATIVE → CONFIRMED when n_hits ≥ min_hits.

7.  Purge REMOVED tracks.

8.  Return all CONFIRMED tracks for downstream consumption.

Robotics rationale
------------------
The two-stage design is critical for robotics scenarios where objects
are temporarily occluded (robot arm blocks sensor, object passes
behind furniture). A purely high-confidence tracker drops the track;
ByteTrack rescues it with low-confidence evidence. In a planning
system, a dropped track means a suddenly-unknown obstacle — dangerous.

Output guarantee
----------------
update() always returns a list of Track objects in CONFIRMED state.
Each Track carries: track_id, bbox_xyxy, covariance, velocity, nis.
The covariance matrix is the uncertainty estimate consumed by the
world model in Step 11.

Upgrade path
------------
Step 10: Replace iou_distance with appearance_distance (ReID embeddings)
         for robust re-identification after long occlusion.
Step 11: Pass confirmed tracks directly to SceneGraph.update().
Step 3 ROS2: Wrap update() in a ROS2 node subscribing to Detection[]
         and publishing Track[] — zero changes to this class.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np

from perception.camera_interface import CameraFrame
from perception.detector import Detection
from tracking.association import (
    build_iou_cost_matrix_with_gate,
    linear_assignment,
)
from tracking.track import Track, TrackState


class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Parameters (all from config['tracker'])
    ----------------------------------------
    high_thresh      : float — confidence threshold separating D_high / D_low.
                               Detections below low_thresh are discarded entirely.
    low_thresh       : float — minimum confidence to use in stage 2 rescue.
    new_track_thresh : float — minimum confidence to initialise a new track.
                               Typically equals high_thresh.
    iou_threshold    : float — minimum IoU to accept a track-detection match.
                               Cost threshold = 1 - iou_threshold.
    max_age          : int   — frames a LOST track survives without detection.
    min_hits         : int   — updates needed to promote TENTATIVE → CONFIRMED.

    Usage
    -----
    tracker = ByteTracker(config)
    for frame in camera:
        detections = detector.detect(frame)
        confirmed_tracks = tracker.update(detections, frame)
        for track in confirmed_tracks:
            print(track.track_id, track.bbox_xyxy, track.position_std)
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        cfg = config.get("tracker", {})

        self._high_thresh:      float = float(cfg.get("high_thresh",      0.50))
        self._low_thresh:       float = float(cfg.get("low_thresh",       0.10))
        self._new_track_thresh: float = float(cfg.get("new_track_thresh", 0.50))
        self._iou_threshold:    float = float(cfg.get("iou_threshold",    0.30))
        self._max_age:          int   = int(cfg.get("max_age",            30))
        self._min_hits:         int   = int(cfg.get("min_hits",           1))

        # Cost threshold for linear_assignment:
        # accept match iff cost = 1 - IoU <= 1 - iou_threshold
        self._cost_thresh: float = 1.0 - self._iou_threshold

        self._tracks: List[Track] = []
        self._last_timestamp: Optional[float] = None
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Detection],
        frame: CameraFrame,
    ) -> List[Track]:
        """
        Process one frame. Core ByteTrack update cycle.

        Parameters
        ----------
        detections : output of Detector.detect() for this frame —
                     already sorted confidence-descending (guaranteed
                     by YOLOv8Detector).
        frame      : CameraFrame carrying the monotonic timestamp
                     used to compute dt for KF prediction.

        Returns
        -------
        List of Track objects in CONFIRMED state.
        Empty list on the first frame before any track is confirmed.
        """
        self._frame_count += 1

        # ---- 0. Compute dt -----------------------------------------
        dt = self._compute_dt(frame.timestamp)
        self._last_timestamp = frame.timestamp

        # ---- 1. Split detections -----------------------------------
        dets_high, dets_low = self._split_detections(detections)

        # ---- 2. Predict all active tracks --------------------------
        active_tracks = [t for t in self._tracks if t.state != TrackState.REMOVED]
        for track in active_tracks:
            track.predict(dt)

        # ---- 3. Stage 1: D_high vs ALL tracks ----------------------
        (matched_s1,
         unmatched_tracks_s1,
         unmatched_dets_high) = self._associate(
            dets_high, active_tracks, self._cost_thresh
        )

        # Update matched tracks (stage 1)
        for t_idx, d_idx in matched_s1:
            track = active_tracks[t_idx]
            track.update(dets_high[d_idx], timestamp=frame.timestamp)
            self._maybe_confirm(track)
            # Re-confirm a LOST track that was found
            if track.state == TrackState.LOST:
                track.state = TrackState.CONFIRMED

        # ---- 4. Stage 2: D_low vs unmatched tracks -----------------
        # Only the tracks that stage 1 failed to match
        pool_s2 = [active_tracks[i] for i in unmatched_tracks_s1]

        (matched_s2,
         unmatched_tracks_s2,
         _unmatched_dets_low) = self._associate(
            dets_low, pool_s2, self._cost_thresh
        )

        # Update matched tracks (stage 2)
        for t_idx, d_idx in matched_s2:
            track = pool_s2[t_idx]
            track.update(dets_low[d_idx], timestamp=frame.timestamp)
            self._maybe_confirm(track)
            if track.state == TrackState.LOST:
                track.state = TrackState.CONFIRMED

        # ---- 5. Handle tracks missed in both stages ----------------
        for i in unmatched_tracks_s2:
            self._mark_missed(pool_s2[i])

        # ---- 6. Initialise new tracks from unmatched D_high --------
        for d_idx in unmatched_dets_high:
            det = dets_high[d_idx]
            if det.confidence >= self._new_track_thresh:
                new_track = Track(det, self._config)
                # If min_hits == 1, confirm immediately
                self._maybe_confirm(new_track)
                self._tracks.append(new_track)

        # ---- 7. Purge REMOVED tracks --------------------------------
        self._tracks = [t for t in self._tracks
                        if t.state != TrackState.REMOVED]

        # ---- 8. Return CONFIRMED tracks ----------------------------
        return [t for t in self._tracks if t.state == TrackState.CONFIRMED]

    def reset(self) -> None:
        """
        Clear all tracks and reset state.
        Call between video sequences or when reinitialising the pipeline.
        Does NOT reset the global Track ID counter — call
        Track.reset_id_counter() separately if needed.
        """
        self._tracks.clear()
        self._last_timestamp = None
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tracks(self) -> List[Track]:
        """All non-REMOVED tracks (including TENTATIVE and LOST)."""
        return [t for t in self._tracks if t.state != TrackState.REMOVED]

    @property
    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self._tracks if t.state == TrackState.CONFIRMED]

    @property
    def lost_tracks(self) -> List[Track]:
        return [t for t in self._tracks if t.state == TrackState.LOST]

    @property
    def n_confirmed(self) -> int:
        return len(self.confirmed_tracks)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_dt(self, timestamp: float) -> float:
        """
        Compute time delta since last frame.
        On the first frame, assume 30 FPS (dt = 1/30).
        Guards against negative dt from timestamp bugs.
        """
        if self._last_timestamp is None:
            return 1.0 / 30.0

        dt = timestamp - self._last_timestamp
        if dt <= 0.0:
            warnings.warn(
                f"Non-positive dt={dt:.6f}s detected. "
                "Using fallback dt=1/30. "
                "Check that frame.timestamp uses time.monotonic().",
                stacklevel=3,
            )
            return 1.0 / 30.0

        return dt

    def _split_detections(
        self,
        detections: List[Detection],
    ) -> tuple[List[Detection], List[Detection]]:
        """
        Split detections into high-confidence and low-confidence pools.

        D_high (conf >= high_thresh) → primary association + track birth
        D_low  (low_thresh <= conf < high_thresh) → rescue association only
        Detections below low_thresh are discarded entirely.
        """
        dets_high = [d for d in detections if d.confidence >= self._high_thresh]
        dets_low  = [d for d in detections
                     if self._low_thresh <= d.confidence < self._high_thresh]
        return dets_high, dets_low

    def _associate(
        self,
        detections: List[Detection],
        tracks: List[Track],
        cost_thresh: float,
    ) -> tuple[list, list, list]:
        """
        Build IoU cost matrix and run Hungarian assignment.

        Returns
        -------
        matches         : list of (track_idx, det_idx)
        unmatched_tracks: list of track indices with no match
        unmatched_dets  : list of detection indices with no match
        """
        if not tracks or not detections:
            return (
                [],
                list(range(len(tracks))),
                list(range(len(detections))),
            )

        cost_matrix = build_iou_cost_matrix_with_gate(
            tracks, detections, max_iou_distance=self._cost_thresh
        )

        return linear_assignment(cost_matrix, thresh=cost_thresh)

    def _maybe_confirm(self, track: Track) -> None:
        """
        Promote TENTATIVE → CONFIRMED if n_hits >= min_hits.
        No-op for tracks already CONFIRMED or LOST.
        """
        if track.state == TrackState.TENTATIVE and track.n_hits >= self._min_hits:
            track.state = TrackState.CONFIRMED

    def _mark_missed(self, track: Track) -> None:
        """
        Handle a track that was not matched in either association stage.

        TENTATIVE → REMOVED immediately (never confirmed, unreliable)
        CONFIRMED → LOST (keep predicting, may reappear)
        LOST      → n_misses++; REMOVED if exceeds max_age
        """
        track.n_misses += 1

        if track.state == TrackState.TENTATIVE:
            track.state = TrackState.REMOVED

        elif track.state == TrackState.CONFIRMED:
            track.state = TrackState.LOST

        elif track.state == TrackState.LOST:
            if track.n_misses > self._max_age:
                track.state = TrackState.REMOVED

    def __repr__(self) -> str:
        return (
            f"ByteTracker("
            f"confirmed={self.n_confirmed} "
            f"lost={len(self.lost_tracks)} "
            f"total={len(self.tracks)} "
            f"frame={self._frame_count})"
        )
