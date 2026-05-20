"""
SemanticMap — persistent metric-semantic voxel map (semantic SLAM mapping layer).

This is the mapping back-end of "full semantic SLAM" (roadmap Phase 5,
item 14). The pipeline already produces, live and per-frame, every input a
metric-semantic map needs:

  - a dense metric depth map         (DepthAnything / stereo backends)
  - a per-pixel semantic label map   (Mask2Former — SemanticMask)
  - the camera's SE(3) world pose    (DPVO / DPV-SLAM / VIO PoseEstimator)
  - calibrated intrinsics            (CameraFrame.intrinsics)

What was missing was *fusion over time*: folding those per-frame observations
into a single persistent 3D model. SemanticMap is that model — a globally
consistent, voxel-hashed map where every voxel carries both an occupancy
estimate and a fused semantic-class distribution.

Why this design (and not the alternatives)
-------------------------------------------
The roadmap flagged two candidate directions and asked for a hardware fit
study on the 4070Ti (12 GB):

  * Kimera-Semantics (Rosinol 2020) — online voxel-based metric-semantic
    mapping. Its *mapping* stage is CPU-side voxel hashing; the GPU cost is
    only the 2D segmenter, which this pipeline already runs. Kimera's "needs
    IMU" caveat is about its VIO *front-end* — irrelevant here, because we
    bring our own PoseEstimator (DPVO / DPV-SLAM / VIO). We port only the
    mapping back-end.
  * Open-vocabulary mapping (OpenScene, ConceptGraphs 2024) — richer queries
    but sub-real-time, batch-only, and wants a heavy CLIP/SAM model resident
    in VRAM alongside YOLO + DepthAnything + Mask2Former + DINOv2.

SemanticMap follows the Kimera-Semantics route: it adds *zero* GPU load and
~tens of MB of host RAM, and runs inline in the 30 Hz loop. An open-vocab
layer can still slot above it later for natural-language queries.

Label fusion
------------
Each voxel keeps a sparse `class_scores: {class_id -> score}` table. On every
observation of class `c` the voxel does `class_scores[c] += weight`. This is
the recursive Bayesian MAP update for the voxel's class under a symmetric
label-confusion model: in log-space a measurement adds a large constant to
the observed class and a smaller constant to every other class; the uniform
part cancels under argmax, leaving a single positive increment on the
observed class. So vote-accumulation *is* the MAP estimate — see McCormac
et al. (SemanticFusion, 2017) and Rosinol et al. (Kimera, 2020).

`weight` defaults to 1.0 (pure counting). When a future segmenter exposes
per-pixel class probabilities, pass that probability as the weight and the
same code becomes a confidence-weighted Bayesian filter — no rule change.

Occupancy
---------
Each integrated voxel accumulates hit log-odds, clamped to ±`occupancy_clamp`.
A voxel is "occupied" once its log-odds exceed 0. This v1 integrates only the
observed *surface* voxels — it does not ray-carve free space between the
camera and the surface (Voxblox/OctoMap-style miss updates). Surface-only
integration is correct for a first pass; free-space carving is an O(range /
voxel) cost per pixel and is deferred.

Coordinate frames
-----------------
Voxels are indexed in the PoseEstimator's world frame ("map"). The voxel hash
is unbounded and origin-free: key = floor(p_world / voxel_size). Storage
scales with observed surface area, not with a fixed volume.

Known limitation — loop-closure deformation
--------------------------------------------
DPV-SLAM emits loop closures that retroactively correct earlier poses. This
v1 integrates each observation at the pose available *at that frame* and does
not re-deform already-integrated voxels when a later loop closure shifts the
trajectory. Kimera handles this with a mesh/voxel deformation graph; that is
a substantial separate effort and is deferred. In practice the drift between
closures is small at the voxel resolutions used here.

References
----------
Rosinol et al.  — Kimera: an Open-Source Library for Real-Time Metric-
                  Semantic Localization and Mapping. ICRA 2020.
McCormac et al. — SemanticFusion: Dense 3D Semantic Mapping with
                  Convolutional Neural Networks. ICRA 2017.
Hornung et al.  — OctoMap: An Efficient Probabilistic 3D Mapping Framework.
                  Autonomous Robots, 2013.
"""

from __future__ import annotations

import colorsys
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from perception.camera_interface import CameraIntrinsics
from perception.pose_estimator import CameraPose
from perception.semantic_segmenter import SemanticMask

log = logging.getLogger(__name__)


VoxelKey = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def class_colour(class_id: int) -> Tuple[int, int, int]:
    """
    Deterministic RGB colour (0-255) for a semantic class id.

    A pure id→colour function so the Rerun voxel cloud and the ROS2
    PointCloud2 publisher render the same class in the same colour
    without sharing a dataset-specific palette. Hue is spread around
    the wheel by a step coprime-ish with 360; saturation/value are
    fixed so every class is vivid and distinguishable.

    Lives here (rather than in `visualization/`) because the ROS2 node
    must reach it without importing the Rerun-heavy viz package.
    """
    hue = ((int(class_id) * 47) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.70, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class SemanticMapParams:
    """
    Geometry + integration knobs for the metric-semantic voxel map.

    voxel_size_m
        Edge length of a cubic voxel. 10 cm is a sensible default — fine
        enough for furniture/room structure, coarse enough that mono-depth
        noise doesn't smear a surface across many voxels.
    min_range_m / max_range_m
        Depth pixels outside this band are ignored. The near clip drops
        degenerate near-zero depths; the far clip drops the long tail where
        monocular depth is least trustworthy and a single pixel would
        otherwise paint a large, badly-localised voxel.
    pixel_stride
        Integrate every Nth pixel along each axis. stride=4 on a 640×480
        frame integrates ~19 k rays/frame before range filtering — plenty
        to fill voxels at 10 cm while keeping the per-frame loop cheap.
    occupancy_hit_logodds
        Log-odds added to a voxel each time it is observed.
    occupancy_clamp
        Symmetric clamp on accumulated log-odds — bounds confidence so a
        voxel can still be forgotten/overridden by later evidence.
    observation_weight
        Score added to a voxel's observed class per observation. 1.0 =
        plain vote counting; pass a per-pixel probability here once a
        segmenter exposes one.
    max_voxels
        Soft cap on stored voxels. 0 = unbounded. When exceeded after an
        integrate(), the least-recently-observed voxels are evicted.
    """
    voxel_size_m:          float = 0.10
    min_range_m:           float = 0.30
    max_range_m:           float = 8.00
    pixel_stride:          int   = 4
    occupancy_hit_logodds: float = 0.85
    occupancy_clamp:       float = 4.00
    observation_weight:    float = 1.00
    max_voxels:            int   = 0


# ---------------------------------------------------------------------------
# Voxel
# ---------------------------------------------------------------------------

@dataclass
class SemanticVoxel:
    """
    One occupied cell of the semantic map.

    class_scores
        Sparse {class_id: accumulated score}. The MAP class label is the
        argmax; normalised score is a [0, 1] confidence.
    log_odds
        Accumulated occupancy log-odds. > 0 ⇒ occupied.
    hits
        Number of observations integrated into this voxel.
    last_seen
        Monotonic timestamp of the most recent observation. Drives pruning
        and eviction.
    """
    class_scores: Dict[int, float] = field(default_factory=dict)
    log_odds:     float            = 0.0
    hits:         int              = 0
    last_seen:    float            = 0.0

    def observe(
        self,
        class_id:   int,
        weight:     float,
        hit_logodds: float,
        clamp:      float,
        timestamp:  float,
    ) -> None:
        """Fold one (class, occupancy) observation into this voxel."""
        self.class_scores[class_id] = (
            self.class_scores.get(class_id, 0.0) + weight
        )
        self.log_odds = float(
            np.clip(self.log_odds + hit_logodds, -clamp, clamp)
        )
        self.hits += 1
        self.last_seen = timestamp

    @property
    def is_occupied(self) -> bool:
        return self.log_odds > 0.0

    @property
    def dominant_class_id(self) -> int:
        """MAP class id — argmax of the accumulated scores."""
        return max(self.class_scores.items(), key=lambda kv: kv[1])[0]

    @property
    def confidence(self) -> float:
        """
        Normalised score of the MAP class in [0, 1]. 1.0 means every
        observation of this voxel agreed on its label.
        """
        total = sum(self.class_scores.values())
        if total <= 0.0:
            return 0.0
        return max(self.class_scores.values()) / total


# ---------------------------------------------------------------------------
# SemanticMap
# ---------------------------------------------------------------------------

class SemanticMap:
    """
    Persistent voxel-hashed metric-semantic map.

    Construct once with params; call integrate() each frame with a dense
    depth map, a SemanticMask, the camera world-pose and intrinsics. Query
    any time — the map is just a dict and is safe to read between frames.

    Usage
    -----
    smap = SemanticMap(SemanticMapParams(voxel_size_m=0.10))
    for frame in source:
        depth = depth_estimator.dense_depth_map(frame)
        sem   = segmenter.segment(frame)
        pose  = pose_estimator.estimate(frame)
        if depth is not None and sem is not None and pose is not None:
            smap.integrate(depth, sem, pose, frame.intrinsics)
    pts, labels, conf = smap.voxel_centres_world(), ...
    """

    def __init__(self, params: Optional[SemanticMapParams] = None) -> None:
        self._p = params or SemanticMapParams()
        if self._p.voxel_size_m <= 0.0:
            raise ValueError(
                f"voxel_size_m must be positive, got {self._p.voxel_size_m}"
            )
        if self._p.max_range_m <= self._p.min_range_m:
            raise ValueError(
                f"max_range_m ({self._p.max_range_m}) must exceed "
                f"min_range_m ({self._p.min_range_m})"
            )
        if self._p.pixel_stride < 1:
            raise ValueError(
                f"pixel_stride must be >= 1, got {self._p.pixel_stride}"
            )
        self._voxels: Dict[VoxelKey, SemanticVoxel] = {}
        # Class-id → name map and dataset, captured from the first
        # SemanticMask integrated. Lets class_at() resolve human-readable
        # names without the caller threading the SemanticMask back in.
        self._class_names: Dict[int, str] = {}
        self._dataset: str = ""
        self._frames_integrated = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def params(self) -> SemanticMapParams:
        return self._p

    @property
    def dataset(self) -> str:
        return self._dataset

    @property
    def class_names(self) -> Dict[int, str]:
        return dict(self._class_names)

    @property
    def frames_integrated(self) -> int:
        return self._frames_integrated

    def __len__(self) -> int:
        return len(self._voxels)

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def integrate(
        self,
        depth_map:     np.ndarray,
        semantic_mask: SemanticMask,
        camera_pose:   CameraPose,
        intrinsics:    CameraIntrinsics,
    ) -> int:
        """
        Fold one frame's (depth, labels, pose) observation into the map.

        Parameters
        ----------
        depth_map     : (H, W) float metric depth in the camera frame.
        semantic_mask : SemanticMask whose `mask` is (H, W) — must match
                        depth_map's shape, else the frame is skipped.
        camera_pose   : SE(3) world ← camera pose for this frame.
        intrinsics    : camera intrinsics matching the (H, W) image.

        Returns
        -------
        Number of distinct voxels touched this frame. 0 means the frame
        was skipped (shape mismatch, no valid pixels, ...).

        Notes
        -----
        depth_map and semantic_mask.mask must share the same (H, W). The
        segmenter can resample internally; when shapes disagree the frame
        is skipped rather than raising — mirrors the drivable-costmap
        projector's defensive handling in launch.py.
        """
        depth_map = np.asarray(depth_map)
        if depth_map.ndim != 2:
            return 0
        if depth_map.shape != semantic_mask.mask.shape:
            log.debug(
                "SemanticMap: depth %s vs mask %s shape mismatch — skipped.",
                depth_map.shape, semantic_mask.mask.shape,
            )
            return 0

        # Capture the label vocabulary once.
        if not self._class_names and semantic_mask.class_names:
            self._class_names = dict(semantic_mask.class_names)
            self._dataset = semantic_mask.dataset

        p = self._p
        H, W = depth_map.shape
        stride = p.pixel_stride

        # Subsampled pixel grid.
        vs = np.arange(0, H, stride)
        us = np.arange(0, W, stride)
        grid_v, grid_u = np.meshgrid(vs, us, indexing="ij")
        d = depth_map[grid_v, grid_u].astype(np.float64)

        valid = np.isfinite(d) & (d >= p.min_range_m) & (d <= p.max_range_m)
        if not valid.any():
            return 0

        u = grid_u[valid].astype(np.float64)
        v = grid_v[valid].astype(np.float64)
        z = d[valid]
        labels = semantic_mask.mask[grid_v[valid], grid_u[valid]].astype(int)

        # Pixel → camera-frame 3D (OpenCV pinhole).
        x_cam = (u - intrinsics.cx) * z / intrinsics.fx
        y_cam = (v - intrinsics.cy) * z / intrinsics.fy
        pts_cam = np.stack([x_cam, y_cam, z], axis=1)        # (N, 3)

        # Camera → world: p_world = R @ p_cam + t.
        R = np.asarray(camera_pose.R, dtype=np.float64)
        t = np.asarray(camera_pose.t, dtype=np.float64)
        pts_world = pts_cam @ R.T + t                        # (N, 3)

        # World → integer voxel keys.
        keys = np.floor(pts_world / p.voxel_size_m).astype(np.int64)

        ts = float(semantic_mask.timestamp)
        touched: set[VoxelKey] = set()
        for (kx, ky, kz), class_id in zip(keys, labels):
            key: VoxelKey = (int(kx), int(ky), int(kz))
            voxel = self._voxels.get(key)
            if voxel is None:
                voxel = SemanticVoxel()
                self._voxels[key] = voxel
            voxel.observe(
                class_id=int(class_id),
                weight=p.observation_weight,
                hit_logodds=p.occupancy_hit_logodds,
                clamp=p.occupancy_clamp,
                timestamp=ts,
            )
            touched.add(key)

        self._frames_integrated += 1
        if p.max_voxels > 0 and len(self._voxels) > p.max_voxels:
            self._evict_oldest(len(self._voxels) - p.max_voxels)
        return len(touched)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def _evict_oldest(self, n: int) -> None:
        """Drop the `n` least-recently-observed voxels."""
        if n <= 0:
            return
        oldest = sorted(
            self._voxels.items(), key=lambda kv: kv[1].last_seen,
        )[:n]
        for key, _ in oldest:
            del self._voxels[key]

    def prune(self, now: float, max_age_s: float) -> int:
        """
        Drop voxels not observed within the last `max_age_s` seconds.

        Returns the number of voxels removed. `max_age_s <= 0` is a no-op
        — the persistent map keeps every voxel indefinitely, which is the
        right default for static structure.
        """
        if max_age_s <= 0.0:
            return 0
        cutoff = now - max_age_s
        stale = [k for k, vx in self._voxels.items() if vx.last_seen < cutoff]
        for k in stale:
            del self._voxels[k]
        return len(stale)

    def clear(self) -> None:
        """Drop every voxel. Vocabulary + frame count are retained."""
        self._voxels.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def voxel_key(self, x_m: float, y_m: float, z_m: float) -> VoxelKey:
        """Integer voxel key for a world-frame point."""
        vs = self._p.voxel_size_m
        return (
            int(math.floor(x_m / vs)),
            int(math.floor(y_m / vs)),
            int(math.floor(z_m / vs)),
        )

    def voxel_at(
        self, x_m: float, y_m: float, z_m: float,
    ) -> Optional[SemanticVoxel]:
        """The SemanticVoxel covering a world point, or None if unmapped."""
        return self._voxels.get(self.voxel_key(x_m, y_m, z_m))

    def class_at(
        self, x_m: float, y_m: float, z_m: float,
    ) -> Optional[str]:
        """
        MAP semantic-class *name* at a world point, or None when the voxel
        is unmapped or the class id has no known name.
        """
        voxel = self.voxel_at(x_m, y_m, z_m)
        if voxel is None or not voxel.class_scores:
            return None
        return self._class_names.get(voxel.dominant_class_id)

    def occupied_voxels(self) -> Iterable[Tuple[VoxelKey, SemanticVoxel]]:
        """Iterate (key, voxel) over voxels whose occupancy log-odds > 0."""
        return (
            (k, vx) for k, vx in self._voxels.items() if vx.is_occupied
        )

    def voxel_centres_world(self, occupied_only: bool = True) -> np.ndarray:
        """
        (N, 3) float64 array of voxel-centre world positions.

        occupied_only restricts the result to voxels with positive
        occupancy log-odds (the default — what a consumer wants to draw).
        """
        keys = self._selected_keys(occupied_only)
        if not keys:
            return np.zeros((0, 3), dtype=np.float64)
        vs = self._p.voxel_size_m
        arr = np.array(keys, dtype=np.float64)
        return (arr + 0.5) * vs

    def voxel_labels(self, occupied_only: bool = True) -> np.ndarray:
        """
        (N,) int32 array of MAP class ids, row-aligned with
        voxel_centres_world(occupied_only=...).
        """
        keys = self._selected_keys(occupied_only)
        if not keys:
            return np.zeros((0,), dtype=np.int32)
        return np.array(
            [self._voxels[k].dominant_class_id for k in keys],
            dtype=np.int32,
        )

    def voxel_confidence(self, occupied_only: bool = True) -> np.ndarray:
        """
        (N,) float64 array of MAP-class confidences in [0, 1], row-aligned
        with voxel_centres_world(occupied_only=...).
        """
        keys = self._selected_keys(occupied_only)
        if not keys:
            return np.zeros((0,), dtype=np.float64)
        return np.array(
            [self._voxels[k].confidence for k in keys], dtype=np.float64,
        )

    def query_class(
        self, class_name: str, occupied_only: bool = True,
    ) -> np.ndarray:
        """
        (N, 3) world centres of voxels whose MAP label is `class_name`.

        The semantic-SLAM payoff query: "where is all the <road / wall /
        person / ...> the robot has ever seen?"
        """
        target_ids = {
            cid for cid, name in self._class_names.items()
            if name == class_name
        }
        if not target_ids:
            return np.zeros((0, 3), dtype=np.float64)
        keys = [
            k for k in self._selected_keys(occupied_only)
            if self._voxels[k].dominant_class_id in target_ids
        ]
        if not keys:
            return np.zeros((0, 3), dtype=np.float64)
        vs = self._p.voxel_size_m
        return (np.array(keys, dtype=np.float64) + 0.5) * vs

    def class_histogram(self, occupied_only: bool = True) -> Dict[str, int]:
        """
        {class_name: voxel_count} over the map — a quick semantic summary.
        Class ids with no known name are keyed by their stringified id.
        """
        hist: Dict[str, int] = {}
        for k in self._selected_keys(occupied_only):
            cid = self._voxels[k].dominant_class_id
            name = self._class_names.get(cid, str(cid))
            hist[name] = hist.get(name, 0) + 1
        return hist

    def _selected_keys(self, occupied_only: bool) -> List[VoxelKey]:
        """Voxel keys honouring the occupied_only filter, order-stable."""
        if not occupied_only:
            return list(self._voxels.keys())
        return [k for k, vx in self._voxels.items() if vx.is_occupied]

    def __repr__(self) -> str:
        return (
            f"SemanticMap(voxels={len(self._voxels)}, "
            f"voxel_size={self._p.voxel_size_m} m, "
            f"frames={self._frames_integrated}, "
            f"dataset={self._dataset or 'unknown'!r})"
        )
