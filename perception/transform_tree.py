"""
TransformTree — tf2-style coordinate frame manager.

Maintains a directed tree of named frames connected by SE(3) transforms.
Static transforms (camera extrinsics, sensor mounts) are set once.
Dynamic transforms (camera ego-pose, base_link motion) are updated each
frame from a PoseEstimator. Point queries walk the chain and compose.

Architecture
------------
TransformTree
    ├── _parent_of:   Dict[child_frame, parent_frame]
    ├── _static:      Dict[(parent, child), Transform]   — set once
    └── _dynamic:     Dict[(parent, child), Transform]   — overwritten

Coordinate convention follows ROS REP-103 / REP-105:
    map        — fixed world frame, Z-up
    odom       — drifting odometry frame, Z-up
    base_link  — robot body frame, X-forward
    camera_*   — sensor frames (OpenCV: X-right, Y-down, Z-forward)

A typical chain for this pipeline:
    map ← odom ← base_link ← camera_frame

The pose estimator sets odom ← base_link (the robot's ego-motion).
Camera extrinsics base_link ← camera_frame are static from config.
map ← odom is identity until a SLAM backend introduces loop closures.

Lookup semantics
----------------
lookup_transform(target, source) returns the Transform that maps a point
expressed in `source` into `target`:

    p_target = T.R @ p_source + T.t

This matches tf2's lookupTransform(target_frame, source_frame).

Notes
-----
- No time interpolation yet. Transforms are looked up at "latest". Frame
  loop is at ~30 Hz so sub-frame misalignment is negligible for now.
  Interpolation can be added behind the same lookup API when a second
  sensor with its own clock arrives (Phase 5).
- Each frame has exactly one parent. Multi-parent (DAG) is not supported.

Reference
---------
Foote, T. — tf: The transform library. Open-Source Software for Robotics
            workshop, IEEE ICRA, 2013.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Transform:
    """
    SE(3) transform between two named frames.

    p_target = R @ p_source + t

    Attributes
    ----------
    R            : (3, 3) rotation matrix  — target ← source
    t            : (3,)   translation      — source origin in target frame, metres
    target_frame : frame the transform maps INTO
    source_frame : frame the transform maps FROM
    timestamp    : monotonic seconds. 0.0 for static transforms.
    """
    R:            np.ndarray
    t:            np.ndarray
    target_frame: str
    source_frame: str
    timestamp:    float = 0.0

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=np.float64)
        self.t = np.asarray(self.t, dtype=np.float64)
        assert self.R.shape == (3, 3), f"R must be (3,3), got {self.R.shape}"
        assert self.t.shape == (3,),   f"t must be (3,),  got {self.t.shape}"

    def transform_point(self, p: np.ndarray) -> np.ndarray:
        """Transform a (3,) point from source_frame into target_frame."""
        p = np.asarray(p, dtype=np.float64)
        return self.R @ p + self.t

    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        """Transform (N, 3) points from source_frame into target_frame."""
        pts = np.asarray(pts, dtype=np.float64)
        return (self.R @ pts.T).T + self.t

    def inverse(self) -> "Transform":
        """Swap source and target. p_source = Rᵀ @ p_target − Rᵀ @ t."""
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return Transform(
            R=R_inv,
            t=t_inv,
            target_frame=self.source_frame,
            source_frame=self.target_frame,
            timestamp=self.timestamp,
        )

    def compose(self, other: "Transform") -> "Transform":
        """
        Compose self ∘ other.

        Requires other.target_frame == self.source_frame. The result
        maps points from other.source_frame into self.target_frame:

            p_self.target = self.R @ (other.R @ p_other.source + other.t) + self.t
                          = (self.R @ other.R) @ p + (self.R @ other.t + self.t)
        """
        if other.target_frame != self.source_frame:
            raise ValueError(
                f"Cannot compose: other.target='{other.target_frame}' "
                f"!= self.source='{self.source_frame}'"
            )
        return Transform(
            R=self.R @ other.R,
            t=self.R @ other.t + self.t,
            target_frame=self.target_frame,
            source_frame=other.source_frame,
            timestamp=max(self.timestamp, other.timestamp),
        )

    @classmethod
    def identity(cls, target_frame: str, source_frame: str) -> "Transform":
        return cls(
            R=np.eye(3),
            t=np.zeros(3),
            target_frame=target_frame,
            source_frame=source_frame,
            timestamp=0.0,
        )

    def __repr__(self) -> str:
        return (
            f"Transform({self.target_frame} ← {self.source_frame}, "
            f"t=[{self.t[0]:.3f},{self.t[1]:.3f},{self.t[2]:.3f}] "
            f"ts={self.timestamp:.3f})"
        )


class TransformTree:
    """
    Directed tree of named coordinate frames.

    Each frame (except the root) has exactly one parent. Lookup between
    any two connected frames composes transforms along the chain.

    Usage
    -----
    tree = TransformTree(root_frame="map")

    # Static: camera mount extrinsic (set once at startup)
    tree.set_static(parent="base_link", child="camera_frame",
                    R=R_base_cam, t=t_base_cam)

    # Dynamic: robot pose in map (updated each frame from PoseEstimator)
    tree.set_dynamic(parent="map", child="base_link",
                     R=R_map_base, t=t_map_base, timestamp=ts)

    # Query: transform a point from camera_frame to map
    p_map = tree.transform_point(p_camera,
                                 target="map", source="camera_frame")
    """

    def __init__(self, root_frame: str = "map") -> None:
        self._root = root_frame
        self._parent_of: Dict[str, str] = {}
        self._edges: Dict[Tuple[str, str], Transform] = {}
        # (parent, child) → Transform. Both static and dynamic live here;
        # dynamic ones are overwritten by set_dynamic, static by set_static.

    @property
    def root(self) -> str:
        return self._root

    @property
    def frames(self) -> List[str]:
        return [self._root] + list(self._parent_of.keys())

    def has_frame(self, frame: str) -> bool:
        return frame == self._root or frame in self._parent_of

    # ------------------------------------------------------------------
    # Edge setters
    # ------------------------------------------------------------------

    def set_static(
        self,
        parent: str,
        child:  str,
        R:      np.ndarray,
        t:      np.ndarray,
    ) -> None:
        """
        Register a time-invariant transform (parent ← child).

        Used for sensor mount extrinsics (base_link → camera_frame) and
        any other fixed geometric relationships.
        """
        self._add_edge(parent, child)
        self._edges[(parent, child)] = Transform(
            R=R, t=t,
            target_frame=parent,
            source_frame=child,
            timestamp=0.0,
        )

    def set_dynamic(
        self,
        parent:    str,
        child:     str,
        R:         np.ndarray,
        t:         np.ndarray,
        timestamp: float,
    ) -> None:
        """
        Update a time-varying transform (parent ← child).

        Used for ego-motion (map → base_link or odom → base_link). The
        latest value overwrites any previous one; no history is kept.
        """
        self._add_edge(parent, child)
        self._edges[(parent, child)] = Transform(
            R=R, t=t,
            target_frame=parent,
            source_frame=child,
            timestamp=timestamp,
        )

    def _add_edge(self, parent: str, child: str) -> None:
        """Register the parent → child relationship. Enforces single-parent."""
        if child == self._root:
            raise ValueError(f"Cannot reparent the root frame '{self._root}'")
        existing = self._parent_of.get(child)
        if existing is not None and existing != parent:
            raise ValueError(
                f"Frame '{child}' already has parent '{existing}', "
                f"cannot reparent to '{parent}'"
            )
        self._parent_of[child] = parent

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup_transform(self, target: str, source: str) -> Transform:
        """
        Return the Transform mapping points from `source` into `target`.

        Walks each frame up to the common ancestor (the lowest frame
        present on both paths to root), composing edges along the way.

        Raises
        ------
        KeyError    : if either frame is unknown
        ValueError  : if no path connects the frames (disconnected tree)
        """
        if not self.has_frame(target):
            raise KeyError(f"Unknown frame: '{target}'")
        if not self.has_frame(source):
            raise KeyError(f"Unknown frame: '{source}'")
        if target == source:
            return Transform.identity(target, source)

        target_chain = self._chain_to_root(target)   # [target, ..., root]
        source_chain = self._chain_to_root(source)
        target_set   = set(target_chain)

        # First frame on source's path that also appears on target's path.
        ancestor: Optional[str] = next(
            (f for f in source_chain if f in target_set), None
        )
        if ancestor is None:
            raise ValueError(
                f"No common ancestor between '{target}' and '{source}'"
            )

        # source → ancestor: forward edges (parent ← child) on the source chain.
        t_source_to_anc = Transform.identity(source, source)
        for child in source_chain:
            if child == ancestor:
                break
            parent = self._parent_of[child]
            edge   = self._edges[(parent, child)]   # parent ← child
            # Compose: parent ← child ∘ (current source → child) = parent ← source
            t_source_to_anc = edge.compose(t_source_to_anc)
        # t_source_to_anc maps: source → ancestor

        # target → ancestor (then invert to get ancestor → target).
        t_target_to_anc = Transform.identity(target, target)
        for child in target_chain:
            if child == ancestor:
                break
            parent = self._parent_of[child]
            edge   = self._edges[(parent, child)]
            t_target_to_anc = edge.compose(t_target_to_anc)

        t_anc_to_target = t_target_to_anc.inverse()
        return t_anc_to_target.compose(t_source_to_anc)

    def transform_point(
        self,
        p:      np.ndarray,
        target: str,
        source: str,
    ) -> np.ndarray:
        """Convenience: look up transform and apply it to a (3,) point."""
        return self.lookup_transform(target, source).transform_point(p)

    def transform_points(
        self,
        pts:    np.ndarray,
        target: str,
        source: str,
    ) -> np.ndarray:
        """Apply lookup_transform to (N, 3) points."""
        return self.lookup_transform(target, source).transform_points(pts)

    def _chain_to_root(self, frame: str) -> List[str]:
        """Frames from `frame` up to and including the root."""
        chain = [frame]
        cur = frame
        while cur in self._parent_of:
            cur = self._parent_of[cur]
            chain.append(cur)
        return chain

    # ------------------------------------------------------------------
    # PoseEstimator integration
    # ------------------------------------------------------------------

    def update_from_camera_pose(
        self,
        camera_pose,        # CameraPose, but avoid circular import
        camera_frame: str = "camera_frame",
    ) -> None:
        """
        Push a CameraPose into the tree as a dynamic edge.

        CameraPose convention is `world ← camera`, so this sets the
        root ← camera_frame edge directly. If a camera extrinsic
        (base_link ← camera_frame) is also registered, callers should
        instead push the corresponding root ← base_link edge derived
        from the camera pose. For the common single-camera-on-fixed-mount
        case this short-circuit is correct.
        """
        if camera_pose is None:
            return
        self.set_dynamic(
            parent=self._root,
            child=camera_frame,
            R=camera_pose.R,
            t=camera_pose.t,
            timestamp=camera_pose.timestamp,
        )

    def __repr__(self) -> str:
        return (
            f"TransformTree(root='{self._root}', "
            f"frames={len(self.frames)}, edges={len(self._edges)})"
        )
