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
from world_model.stability import (
    DEFAULT_TIMEOUTS_S,
    StabilityClass,
    stability_for_class,
    timeout_for_stability,
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from perception.transform_tree import TransformTree

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

    def __init__(
        self,
        config: dict,
        transform_tree: Optional["TransformTree"] = None,
    ) -> None:
        self._config = config
        cfg = config.get("world_model", {})

        # Maximum history snapshots per object
        self._max_history: int = int(cfg.get("max_history", 30))

        # Per-stability lost-timeouts. Start from defaults
        # (DYNAMIC=1.5s, SEMI_STATIC=60s, STATIC=∞), then apply two
        # layers of overrides:
        #   1. Legacy world_model.lost_timeout_s — backwards-compat for
        #      configs that don't know about stability classes yet.
        #      Treated as the DYNAMIC timeout (the class most callers
        #      meant by "lost timeout" pre-Pass-A).
        #   2. stability.timeouts_s.<CLASS> — per-class overrides, take
        #      precedence over the legacy field.
        stab_cfg = config.get("stability", {})
        self._stability_timeouts: Dict[StabilityClass, float] = dict(DEFAULT_TIMEOUTS_S)
        if "lost_timeout_s" in cfg:
            self._stability_timeouts[StabilityClass.DYNAMIC] = float(cfg["lost_timeout_s"])
        timeout_overrides_raw = stab_cfg.get("timeouts_s", {}) or {}
        for name, secs in timeout_overrides_raw.items():
            try:
                self._stability_timeouts[StabilityClass[name.upper()]] = float(secs)
            except KeyError:
                log.warning(
                    "SceneGraph: unknown stability class %r in stability.timeouts_s",
                    name,
                )

        # Per-COCO-class stability overrides (e.g. tighten "person" to
        # DYNAMIC in a setting where the default isn't wanted).
        class_overrides_raw = stab_cfg.get("class_overrides", {}) or {}
        self._class_stability_overrides: Dict[str, StabilityClass] = {}
        for cls_name, stab_name in class_overrides_raw.items():
            try:
                self._class_stability_overrides[str(cls_name)] = \
                    StabilityClass[stab_name.upper()]
            except KeyError:
                log.warning(
                    "SceneGraph: unknown stability class %r for %s",
                    stab_name, cls_name,
                )

        # Motion-override thresholds (pixel velocity + sustained frame count).
        # Speed is in px/s — depends on camera resolution + object distance.
        # Defaults tuned for ~30 Hz pipeline at typical webcam resolution.
        self._demote_speed_px_s:  float = float(stab_cfg.get("demote_speed_px_s",  100.0))
        self._demote_frames:      int   = int(stab_cfg.get("demote_frames",        90))   # 3 s @ 30 Hz
        self._promote_speed_px_s: float = float(stab_cfg.get("promote_speed_px_s", 10.0))
        self._promote_frames:     int   = int(stab_cfg.get("promote_frames",       900))  # 30 s @ 30 Hz

        # Optional coordinate frame manager. When set, position_world is
        # computed by tree lookup (camera_frame → root). When None,
        # falls back to applying CameraPose directly (single-link case).
        self._tree: Optional["TransformTree"] = transform_tree
        self._camera_frame: str = cfg.get("camera_frame", "camera_frame")

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
            obj.n_updates  = track.n_hits
            obj.is_lost    = is_lost
            obj.class_id   = track.class_id
            obj.class_name = track.class_name
            obj.add_snapshot(snap)
        else:
            initial_stability = stability_for_class(
                track.class_name, self._class_stability_overrides,
            )
            obj = ObjectState(
                track_id   = track.track_id,
                class_id   = track.class_id,
                class_name = track.class_name,
                position   = track.position.copy(),
                covariance = track.covariance.copy(),
                velocity   = track.velocity.copy(),
                score      = track.score,
                last_seen  = timestamp,
                n_updates  = track.n_hits,
                is_lost    = is_lost,
                stability  = initial_stability,
                max_history= self._max_history,
            )
            obj.add_snapshot(snap)
            self._objects[track.track_id] = obj
            log.debug(
                "SceneGraph: new object id=%d class=%s stability=%s",
                track.track_id, track.class_name, initial_stability.name,
            )

        # Motion-based stability override on every confirmed update.
        # Skip while lost — KF predictions don't reflect observed motion.
        if not is_lost:
            self._update_stability(obj)

        if depth_estimate is not None and depth_estimate.position_3d is not None:
            obj.position_3d = depth_estimate.position_3d.copy()
            obj.position_world = self._to_world(obj.position_3d, camera_pose)

    def _to_world(
        self,
        p_camera: np.ndarray,
        camera_pose,
    ) -> Optional[np.ndarray]:
        """
        Lift a camera-frame metric point into the world (root) frame.

        Two paths:
          1. TransformTree is configured → look up camera_frame → root.
             Caller is responsible for keeping the tree updated each frame
             (typically via tree.update_from_camera_pose(camera_pose)).
          2. No tree, raw CameraPose provided → apply pose directly.
             Single-link shortcut for the common case.

        Returns None when neither source of ego-pose is available.
        """
        if self._tree is not None:
            try:
                return self._tree.transform_point(
                    p_camera,
                    target=self._tree.root,
                    source=self._camera_frame,
                )
            except (KeyError, ValueError):
                return None
        if camera_pose is not None:
            return camera_pose.transform_point(p_camera)
        return None

    def _update_stability(self, obj: ObjectState) -> None:
        """
        Motion-based stability override.

        STATIC objects observed to move > demote_speed_px_s sustained
        for demote_frames are demoted to DYNAMIC. DYNAMIC objects
        sitting still (< promote_speed_px_s) for promote_frames are
        promoted to SEMI_STATIC. SEMI_STATIC objects can demote with
        movement but never auto-promote to STATIC (STATIC requires
        an explicit class prior).

        Frame counters reset whenever the relevant condition lapses,
        which gives natural hysteresis — a single jittery frame
        doesn't trip the transition.
        """
        speed_px = obj.speed
        # Demote-on-motion: applies to anything currently above DYNAMIC.
        if obj.stability != StabilityClass.DYNAMIC:
            if speed_px > self._demote_speed_px_s:
                obj._moving_frames += 1
                if obj._moving_frames >= self._demote_frames:
                    log.debug(
                        "SceneGraph: id=%d (%s) %s → DYNAMIC (sustained motion)",
                        obj.track_id, obj.class_name, obj.stability.name,
                    )
                    obj.stability = StabilityClass.DYNAMIC
                    obj._moving_frames = 0
            else:
                obj._moving_frames = 0

        # Promote-on-stillness: DYNAMIC sitting still → SEMI_STATIC.
        if obj.stability == StabilityClass.DYNAMIC:
            if speed_px < self._promote_speed_px_s:
                obj._stationary_frames += 1
                if obj._stationary_frames >= self._promote_frames:
                    log.debug(
                        "SceneGraph: id=%d (%s) DYNAMIC → SEMI_STATIC (stationary)",
                        obj.track_id, obj.class_name,
                    )
                    obj.stability = StabilityClass.SEMI_STATIC
                    obj._stationary_frames = 0
            else:
                obj._stationary_frames = 0

    def _prune(self, now: float) -> None:
        """
        Remove LOST objects whose stability-class timeout has elapsed.

        STATIC objects with `timeout_for_stability` == inf are never
        pruned by this mechanism — they live in the SceneGraph
        indefinitely until evicted by a higher-level policy (the
        WorldMap layer in Pass B).
        """
        to_remove = []
        for tid, obj in self._objects.items():
            if not obj.is_lost:
                continue
            timeout = timeout_for_stability(
                obj.stability, self._stability_timeouts,
            )
            if timeout == float("inf"):
                continue
            if (now - obj.last_seen) > timeout:
                to_remove.append(tid)
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
        frame:           str = "camera",
    ) -> List[Tuple[float, ObjectState]]:
        """
        Return all objects within a given radius of a query position,
        sorted by distance (nearest first).

        Parameters
        ----------
        position        : query point. (2,) pixels for frame='camera',
                          (3,) metres for frame='world'.
        radius          : search radius in the same unit as `position`
                          (pixels for camera frame, metres for world frame).
        include_lost    : if True, include LOST objects in results.
                          Useful for planning around recently-seen obstacles.
        use_mahalanobis : (camera frame only) use Mahalanobis distance
                          instead of Euclidean. Ignored for world-frame queries.
        frame           : 'camera' → 2D pixel query against ObjectState.position
                          'world'  → 3D metric query against ObjectState.position_world
                          Objects without world position are skipped silently
                          in world-frame mode.

        Returns
        -------
        List of (distance, ObjectState) tuples, sorted by distance ascending.
        Empty list if no objects are within radius.

        Robotics use
        ------------
        Camera-frame (legacy 2D):
            obstacles = sg.query_nearby(robot_pos_px, radius=300)

        World-frame (planner-ready):
            obstacles = sg.query_nearby(
                robot_pos_m, radius=2.0, frame='world')
            for dist_m, obj in obstacles:
                costmap.inflate(obj.position_world, obj.position_std)
        """
        position = np.asarray(position, dtype=np.float64)
        results: List[Tuple[float, ObjectState]] = []

        if frame == "world":
            for obj in self._objects.values():
                if obj.is_lost and not include_lost:
                    continue
                if obj.position_world is None:
                    continue
                dist = float(np.linalg.norm(obj.position_world - position))
                if dist <= radius:
                    results.append((dist, obj))
        elif frame == "camera":
            for obj in self._objects.values():
                if obj.is_lost and not include_lost:
                    continue

                if use_mahalanobis:
                    dist = obj.mahalanobis_to(position)
                else:
                    dist = obj.distance_to(position)

                if dist <= radius:
                    results.append((dist, obj))
        else:
            raise ValueError(
                f"frame must be 'camera' or 'world', got '{frame}'"
            )

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
