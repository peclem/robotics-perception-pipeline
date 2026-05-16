"""
SceneGraph — probabilistic world model for the perception pipeline.

The scene graph maintains a registry of all known objects in the scene,
updated from confirmed and lost tracks on every frame. It provides
spatial query interfaces that downstream modules (planners, occupancy
grids, visualizers) call to read the current world state.

Architecture
------------
SceneGraph
  ├── _objects: Dict[int, ObjectState]  — keyed by track_id
  └── _frame_count: int

Update flow (called once per frame from launch.py or world model node):
    scene_graph.update(confirmed_tracks, lost_tracks, timestamp)

Query flow (called by downstream modules):
    nearby = scene_graph.query_nearby(robot_position, radius=200)
    obj    = scene_graph.get_state(track_id=5)
    all    = scene_graph.all_objects()

Robotics context
----------------
In a full robotics system this module sits between the tracker and:
  - Motion planner: query_nearby() → dynamic obstacle list → costmap
  - SLAM: object positions feed into the landmark graph
  - Prediction: trajectory history → future state estimate

The coordinate frame is the camera image frame (pixels).
In Phase 3 with depth data or SLAM, positions become metric 3D.

Upgrade path
------------
Step 13: pass scene graph state to MOT17 evaluator.
Phase 3: replace pixel positions with metric 3D positions from SLAM.
         Replace query_nearby with a 3D spatial index (k-d tree).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from tracking.track import Track
from world_model.object_state import ObjectState

log = logging.getLogger(__name__)


class SceneGraph:
    """
    Probabilistic world model — registry of tracked objects.

    Parameters
    ----------
    config : full pipeline config dict (reads world_model section)

    Usage
    -----
    sg = SceneGraph(config)

    # Every frame:
    sg.update(confirmed_tracks, lost_tracks, timestamp=frame.timestamp)

    # Query:
    nearby = sg.query_nearby(np.array([320, 240]), radius=150)
    obj    = sg.get_state(track_id=3)
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        cfg = config.get("world_model", {})

        # Maximum history snapshots per object
        self._max_history: int = int(cfg.get("max_history", 30))

        # How long (seconds) to keep a LOST object before pruning it
        self._lost_timeout: float = float(cfg.get("lost_timeout_s", 1.0))

        self._objects: Dict[int, ObjectState] = {}
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Update — called once per frame
    # ------------------------------------------------------------------

    def update(
        self,
        confirmed_tracks: List[Track],
        lost_tracks:      List[Track],
        timestamp:        float,
        depth_estimates:  Optional[dict] = None,
        camera_pose=None,
    ) -> None:
        """
        Synchronise the scene graph with the current tracker state.

        Parameters
        ----------
        confirmed_tracks : CONFIRMED tracks from ByteTracker.update()
        lost_tracks      : LOST tracks from ByteTracker.lost_tracks
        timestamp        : monotonic timestamp of the current frame

        Algorithm
        ---------
        1. Update or create ObjectState for each confirmed track.
        2. Mark known objects as LOST if they are in lost_tracks.
        3. Prune objects that have been LOST for > lost_timeout_s.
        """
        self._frame_count += 1

        # Step 1: update confirmed tracks
        confirmed_ids = set()
        for track in confirmed_tracks:
            est = depth_estimates.get(track.track_id) if depth_estimates else None
            self._upsert(track, timestamp, is_lost=False, depth_estimate=est, camera_pose=camera_pose)
            confirmed_ids.add(track.track_id)

        # Step 2: mark lost tracks
        for track in lost_tracks:
            if track.track_id in self._objects:
                self._objects[track.track_id].is_lost = True
            else:
                # Track went directly to LOST without ever being confirmed
                # (e.g. min_hits > 1). Add it as a lost object.
                self._upsert(track, timestamp, is_lost=True)

        # Step 3: prune stale lost objects
        self._prune(timestamp)

    def _upsert(
        self,
        track:     Track,
        timestamp: float,
        is_lost:   bool,
        camera_pose=None,
        depth_estimate: Optional["DepthEstimate"] = None,
    ) -> None:
        """Create or update the ObjectState for a track."""
        snap = track.kf.snapshot(timestamp=timestamp, frame_idx=self._frame_count)

        if track.track_id in self._objects:
            obj = self._objects[track.track_id]
            obj.position   = track.position.copy()
            obj.covariance = track.covariance.copy()
            obj.velocity   = track.velocity.copy()
            obj.score      = track.score
            obj.last_seen  = timestamp
            obj.n_updates  = track.n_update
            obj.is_lost    = is_lost
            obj.class_id   = track.class_id
            obj.class_name = track.class_name
            obj.add_snapshot(snap)
        else:
            obj = ObjectState(
                track_id   = track.track_id,
                class_id   = track.class_id,
                class_name = track.class_name,
                position   = track.position.copy(),
                covariance = track.covariance.copy(),
                velocity   = track.velocity.copy(),
                score      = track.score,
                last_seen  = timestamp,
                n_updates  = track.n_update,
                is_lost    = is_lost,
                max_history= self._max_history,
            )
#       Update metric 3D position if available
        if depth_estimate is not None and depth_estimate.position_3d is not None:
            obj.position_3d = depth_estimate.position_3d.copy()
            obj.add_snapshot(snap)
            self._objects[track.track_id] = obj
            log.debug(
                "SceneGraph: new object id=%d class=%s",
                track.track_id, track.class_name,
            )

    def _prune(self, now: float) -> None:
        """Remove objects that have been LOST for longer than lost_timeout_s."""
        to_remove = [
            tid for tid, obj in self._objects.items()
            if obj.is_lost and (now - obj.last_seen) > self._lost_timeout
        ]
        for tid in to_remove:
            log.debug("SceneGraph: pruning stale object id=%d", tid)
            del self._objects[tid]

    # ------------------------------------------------------------------
    # Queries — downstream interface
    # ------------------------------------------------------------------

    def get_state(self, track_id: int) -> Optional[ObjectState]:
        """
        Get the ObjectState for a specific track ID.
        Returns None if the object is not in the scene graph.
        """
        return self._objects.get(track_id)

    def all_objects(self, include_lost: bool = True) -> List[ObjectState]:
        """
        Return all objects in the scene graph.

        Parameters
        ----------
        include_lost : if False, only return CONFIRMED (non-lost) objects.
        """
        if include_lost:
            return list(self._objects.values())
        return [obj for obj in self._objects.values() if not obj.is_lost]

    def query_nearby(
        self,
        position: np.ndarray,
        radius:   float,
        include_lost:    bool = False,
        use_mahalanobis: bool = False,
    ) -> List[Tuple[float, ObjectState]]:
        """
        Return all objects within a given radius of a query position,
        sorted by distance (nearest first).

        Parameters
        ----------
        position        : (2,) query point [x, y] in pixel coordinates.
        radius          : search radius in pixels.
        include_lost    : if True, include LOST objects in results.
                          Useful for planning around recently-seen obstacles.
        use_mahalanobis : if True, use Mahalanobis distance (uncertainty-aware)
                          instead of Euclidean.

        Returns
        -------
        List of (distance, ObjectState) tuples, sorted by distance ascending.
        Empty list if no objects are within radius.

        Robotics use
        ------------
        A path planner calls this every control cycle:
            obstacles = sg.query_nearby(robot_pos, radius=300)
            for dist, obj in obstacles:
                costmap.inflate(obj.position, obj.position_std)
        """
        position = np.asarray(position, dtype=np.float64)
        results: List[Tuple[float, ObjectState]] = []

        for obj in self._objects.values():
            if obj.is_lost and not include_lost:
                continue

            if use_mahalanobis:
                dist = obj.mahalanobis_to(position)
            else:
                dist = obj.distance_to(position)

            if dist <= radius:
                results.append((dist, obj))

        results.sort(key=lambda t: t[0])
        return results

    def query_by_class(
        self,
        class_name:   str,
        include_lost: bool = False,
    ) -> List[ObjectState]:
        """
        Return all objects of a given semantic class.

        Example:
            people = sg.query_by_class("person")
            vehicles = sg.query_by_class("car")
        """
        return [
            obj for obj in self._objects.values()
            if obj.class_name == class_name
            and (include_lost or not obj.is_lost)
        ]

    def most_uncertain(self, n: int = 3) -> List[ObjectState]:
        """
        Return the n objects with the largest position uncertainty area.
        Useful for active perception — point the camera at the most
        uncertain object to reduce its covariance.
        """
        objects = list(self._objects.values())
        objects.sort(key=lambda o: o.position_uncertainty_area, reverse=True)
        return objects[:n]

    def most_certain(self, n: int = 3) -> List[ObjectState]:
        """Return the n objects with the smallest position uncertainty."""
        objects = list(self._objects.values())
        objects.sort(key=lambda o: o.position_uncertainty_area)
        return objects[:n]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_objects(self) -> int:
        """Total number of objects in the scene graph (confirmed + lost)."""
        return len(self._objects)

    @property
    def n_confirmed(self) -> int:
        return sum(1 for o in self._objects.values() if not o.is_lost)

    @property
    def n_lost(self) -> int:
        return sum(1 for o in self._objects.values() if o.is_lost)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def object_ids(self) -> List[int]:
        return list(self._objects.keys())

    def reset(self) -> None:
        """Clear all objects. Call between sequences."""
        self._objects.clear()
        self._frame_count = 0

    def __repr__(self) -> str:
        return (
            f"SceneGraph("
            f"confirmed={self.n_confirmed} "
            f"lost={self.n_lost} "
            f"frame={self._frame_count})"
        )

    def summary(self) -> str:
        """Human-readable summary of current scene state."""
        lines = [f"SceneGraph — frame {self._frame_count}"]
        for obj in sorted(self._objects.values(), key=lambda o: o.track_id):
            lines.append(f"  {obj}")
        return "\n".join(lines)
