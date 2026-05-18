"""
Detection-to-track association utilities.

Components
----------
iou_batch          : vectorised pairwise IoU between two box sets (M×N)
iou_distance       : 1 - IoU, the cost matrix for Hungarian assignment
linear_assignment  : Hungarian algorithm with cost threshold filtering

Robotics rationale
------------------
IoU is the standard association metric for bounding-box trackers.
Its failure mode is crossing trajectories — two objects swapping IDs
when their boxes overlap. The ByteTrack two-stage approach (Step 5)
mitigates this by using appearance features in Phase 2 (Step 10).

The Hungarian algorithm (scipy.optimize.linear_sum_assignment) finds
the globally optimal one-to-one assignment minimising total cost.
Building the cost matrix explicitly (rather than using a black-box
tracker library) lets you inspect, log, and debug every assignment
decision — essential for a production robotics system.

Upgrade path
------------
Step 10: Replace iou_distance with a combined IoU + ReID embedding
         distance to handle appearance-based re-identification.
         The linear_assignment function is unchanged.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# Type aliases
Matches        = List[Tuple[int, int]]   # (track_idx, det_idx) pairs
UnmatchedList  = List[int]               # indices into tracks or detections


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def iou_batch(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of bounding boxes.

    Parameters
    ----------
    boxes_a : (M, 4) float array, xyxy format
    boxes_b : (N, 4) float array, xyxy format

    Returns
    -------
    iou : (M, N) float array in [0, 1]
          iou[i, j] = IoU between boxes_a[i] and boxes_b[j]

    Implementation
    --------------
    Uses numpy broadcasting — no Python loops. O(M*N) memory, O(M*N) time.
    For a 100-track × 100-detection problem, this is a 10,000-element
    matrix computed in microseconds on CPU.
    """
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)

    # Broadcast: (M, 1, 4) vs (1, N, 4) → (M, N, 4)
    a = boxes_a[:, None, :]   # (M, 1, 4)
    b = boxes_b[None, :, :]   # (1, N, 4)

    # Intersection
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)  # (M, N)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)  # (M, N)
    inter   = inter_w * inter_h                       # (M, N)

    # Individual areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])  # (M,)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])  # (N,)

    # Union: broadcast (M,) + (N,) → (M, N)
    union = area_a[:, None] + area_b[None, :] - inter
    union = np.maximum(union, 1e-10)  # guard against degenerate boxes

    return inter / union  # (M, N)


def iou_distance(
    tracks: list,
    detections: list,
) -> np.ndarray:
    """
    Build the IoU cost matrix between tracks and detections.

    Cost = 1 - IoU, so a perfect match (IoU=1) has cost 0,
    and no overlap (IoU=0) has cost 1.

    Parameters
    ----------
    tracks     : list of Track objects
    detections : list of Detection objects

    Returns
    -------
    cost_matrix : (len(tracks), len(detections)) float array in [0, 1]
    """
    if not tracks or not detections:
        return np.empty((len(tracks), len(detections)), dtype=np.float64)

    track_boxes = np.array([t.bbox_xyxy for t in tracks],      dtype=np.float64)
    det_boxes   = np.array([d.bbox_xyxy for d in detections],  dtype=np.float64)

    return 1.0 - iou_batch(track_boxes, det_boxes)


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float,
) -> Tuple[Matches, UnmatchedList, UnmatchedList]:
    """
    Apply Hungarian algorithm and filter matches by cost threshold.

    Parameters
    ----------
    cost_matrix : (M, N) cost matrix — lower cost = better match.
                  Typically iou_distance output (values in [0, 1]).
    thresh      : maximum acceptable cost. Pairs above this are rejected.
                  thresh = 1 - iou_threshold
                  e.g. iou_threshold=0.3 → thresh=0.7
                  A match is accepted iff IoU > iou_threshold.

    Returns
    -------
    matches         : list of (row_idx, col_idx) accepted pairs
    unmatched_rows  : row indices with no accepted match (unmatched tracks)
    unmatched_cols  : col indices with no accepted match (unmatched dets)

    Algorithm
    ---------
    scipy.optimize.linear_sum_assignment implements the Jonker-Volgenant
    algorithm — O(n³) in the worst case but fast in practice with the
    LAPJV variant. It returns the globally optimal assignment minimising
    total cost. We then filter by thresh to reject poor matches.
    """
    n_rows, n_cols = cost_matrix.shape

    # Degenerate cases — no tracks or no detections
    if n_rows == 0:
        return [], [], list(range(n_cols))
    if n_cols == 0:
        return [], list(range(n_rows)), []
    if cost_matrix.size == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    matches: Matches = []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((int(r), int(c)))
            matched_rows.add(r)
            matched_cols.add(c)
        # Pairs above thresh: row and col remain unmatched

    unmatched_rows = sorted(set(range(n_rows)) - matched_rows)
    unmatched_cols = sorted(set(range(n_cols)) - matched_cols)

    return matches, unmatched_rows, unmatched_cols


def build_iou_cost_matrix_with_gate(
    tracks: list,
    detections: list,
    max_iou_distance: float = 0.9,
) -> np.ndarray:
    """
    Build cost matrix with IoU gating.

    Any track-detection pair with IoU < (1 - max_iou_distance) is set
    to a high sentinel cost, preventing the Hungarian algorithm from
    creating implausible long-range assignments.

    In practice max_iou_distance=0.9 (IoU gate = 0.1) means a track
    and detection must overlap by at least 10% to be considered.
    """
    cost = iou_distance(tracks, detections)
    # Gate: force high cost for near-zero-IoU pairs
    cost[cost > max_iou_distance] = max_iou_distance + 1.0
    return cost


# ---------------------------------------------------------------------------
# Appearance cost (cosine distance between L2-normalised embeddings)
# ---------------------------------------------------------------------------

def appearance_distance(
    tracks:               list,
    detection_embeddings: list,
) -> np.ndarray:
    """
    Build a (len(tracks), len(detections)) cosine-distance cost matrix
    between track embeddings and detection embeddings.

    Convention
    ----------
    Embeddings are assumed L2-normalised (the DINOv2 extractor enforces
    this). Cosine distance is then `1 - dot(a, b)` and lies in [0, 2].
    We clip to [0, 1] before blending with IoU cost: in practice
    foundation-model embeddings of *different* objects sit at cosine
    similarity ~0.3-0.7, never near -1, so the [0, 1] range covers all
    realistic cases and keeps the blended cost on the same scale as IoU.

    Missing-embedding handling
    --------------------------
    Either a track or a detection may have no embedding (e.g. a track
    that has never been matched yet, or a detection whose crop was
    too small to embed). Those entries get cost = 1.0 so the matcher
    ignores them via the appearance term and decides on IoU alone.

    Returns
    -------
    (T, D) float64 in [0, 1].
    """
    T = len(tracks)
    D = len(detection_embeddings)
    out = np.ones((T, D), dtype=np.float64)
    if T == 0 or D == 0:
        return out
    for i, tr in enumerate(tracks):
        t_emb = getattr(tr, "embedding", None)
        if t_emb is None:
            continue
        t_vec = np.asarray(t_emb, dtype=np.float64).reshape(-1)
        for j, d_emb in enumerate(detection_embeddings):
            if d_emb is None:
                continue
            d_vec = np.asarray(d_emb, dtype=np.float64).reshape(-1)
            if t_vec.shape != d_vec.shape:
                continue
            sim = float(np.dot(t_vec, d_vec))
            out[i, j] = float(np.clip(1.0 - sim, 0.0, 1.0))
    return out


def build_combined_cost_matrix(
    tracks:               list,
    detections:           list,
    detection_embeddings: list,
    appearance_weight:    float = 0.25,
    max_iou_distance:     float = 0.9,
) -> np.ndarray:
    """
    Blended IoU + appearance cost matrix.

        cost = (1 - w) * iou_cost + w * appearance_cost

    The IoU gate from `build_iou_cost_matrix_with_gate` is still
    applied AFTER the blend, so a track-detection pair that's
    geometrically infeasible (near-zero IoU) is rejected regardless of
    how similar the appearances are. This is intentional: motion
    consistency is more reliable per-frame than appearance, and gating
    on IoU stops a confusable lookalike across the frame from
    hijacking a track when no real geometric candidate exists.

    appearance_weight = 0.0 collapses to the IoU-only baseline.
    Typical published values: 0.25 (StrongSORT) to 0.5 (Deep OC-SORT).
    """
    iou_cost = iou_distance(tracks, detections)
    if not tracks or not detections:
        return iou_cost
    app_cost = appearance_distance(tracks, detection_embeddings)
    w = float(np.clip(appearance_weight, 0.0, 1.0))
    blended = (1.0 - w) * iou_cost + w * app_cost
    # Gate AFTER blending (see docstring above).
    blended[iou_cost > max_iou_distance] = max_iou_distance + 1.0
    return blended
