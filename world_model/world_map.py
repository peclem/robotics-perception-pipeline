"""
WorldMap — long-term spatial memory of STATIC / SEMI_STATIC objects.

Architectural role
------------------
SceneGraph holds *live* tracks across the short horizon ByteTrack +
KF can sustain. WorldMap holds *persistent* entries keyed on world
position + class + appearance embedding. When a new track is created
for a STATIC or SEMI_STATIC class, the SceneGraph queries the
WorldMap for a likely match (spatial gate + cosine similarity); if
matched, the new track adopts the WorldMap's persistent ID. This is
the "robot enters room, leaves, comes back, recognises the chair"
behaviour.

Re-association policy
---------------------
Two gates, both must pass:

    1. Spatial gate    : ‖new.position_world − entry.position_world‖
                         ≤ spatial_gate_m. Cheap O(N) scan over the
                         entries kept (N typically small — STATIC
                         objects in a building, not all pedestrians
                         ever seen).
    2. Appearance gate : cosine_similarity(new.embedding,
                                           entry.embedding)
                         ≥ similarity_threshold.
                         Skipped if either side has no embedding —
                         spatial-only re-association still works in
                         that degraded mode, just with more false
                         merges in cluttered scenes.

Class gate is implicit: re-association only considered between
entries of the same class. Cross-class matches are always wrong
(a person can't become a chair).

Eviction
--------
Two opt-in mechanisms (off by default → backwards-compatible
monotonic growth):

  - max_age_s        forget any entry whose last_seen is older than
                     now - max_age_s. Run after every insert /
                     re-associate.
  - max_entries      cap on the total number of entries. When the cap
                     is exceeded, drop the LRU entries (oldest
                     last_seen first) until back at the cap.

The "revisited and not seen" eviction (robot returns to a region and
doesn't observe an entry that should be there) is deferred — it
requires knowing which regions the robot has just observed, which is
a planner / sensor-FOV concern that doesn't belong in this layer.

Reference
---------
Kimera-Semantics — Rosinol et al. (ICRA 2020): same two-tier
                   live/persistent split for dynamic + static layers.
Lu et al.        — Layered costmaps (ICRA 2014): the foundational
                   "different memory durations per layer" idea.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from world_model.stability import StabilityClass

log = logging.getLogger(__name__)


@dataclass
class WorldMapEntry:
    """
    One persistent object in the long-term spatial memory.

    Identity (persistent_id) is the stable ID the SceneGraph uses
    across re-associations. Live tracks adopt this ID when they
    successfully match.

    Fields
    ------
    persistent_id  : stable WorldMap-assigned identifier
    class_name     : COCO class — re-association requires class match
    class_id       : COCO class integer
    position_world : (3,) [X, Y, Z] in metres, world (map) frame
    embedding      : (D,) L2-normalised appearance vector, or None
                     when no AppearanceExtractor was available at the
                     time of insertion
    stability      : promoted stability at time of insertion
    first_seen     : timestamp of insertion
    last_seen      : updated whenever the entry is re-associated
    n_observations : how many times this entry has been confirmed
    """
    persistent_id:  int
    class_name:     str
    class_id:       int
    position_world: np.ndarray
    embedding:      Optional[np.ndarray]
    stability:      StabilityClass
    first_seen:     float
    last_seen:      float
    n_observations: int = 1

    def update(
        self,
        position_world: np.ndarray,
        embedding:      Optional[np.ndarray],
        timestamp:      float,
    ) -> None:
        """
        Refine an existing entry from a confirmed re-association.

        Position is a simple running average; embedding is an
        exponential moving average so the entry adapts to gradual
        appearance changes (lighting, viewpoint) without forgetting
        the original look. Coefficients deliberately weak (α=0.1) —
        if the appearance has actually changed enough that the new
        observation barely matches, the gate will fail and a fresh
        entry will be created instead.
        """
        self.position_world = (
            (self.position_world * self.n_observations + position_world)
            / (self.n_observations + 1)
        )
        if embedding is not None and self.embedding is not None:
            alpha = 0.1
            blended = (1 - alpha) * self.embedding + alpha * embedding
            norm = float(np.linalg.norm(blended))
            if norm > 0:
                self.embedding = blended / norm
        elif self.embedding is None and embedding is not None:
            self.embedding = embedding
        self.last_seen = timestamp
        self.n_observations += 1


class WorldMap:
    """
    Spatial + appearance memory of persistent objects.

    Parameters
    ----------
    spatial_gate_m        : metres. Re-association candidates must
                            be within this Euclidean distance.
    similarity_threshold  : cosine similarity threshold ∈ [-1, 1].
                            Typical foundation-model thresholds for
                            same-instance: 0.7-0.85.
    allow_spatial_only    : if True and either side lacks an
                            embedding, fall back to spatial-only
                            re-association. If False, require both
                            embeddings.
    max_age_s             : if > 0, entries whose last_seen is older
                            than now - max_age_s are dropped on every
                            insert / re-associate. 0 disables aging.
    max_entries           : if > 0, total entries are capped at this
                            number — exceeding it drops the oldest
                            (smallest last_seen) until at the cap.
                            0 disables the cap.
    """

    def __init__(
        self,
        spatial_gate_m:       float = 1.5,
        similarity_threshold: float = 0.75,
        allow_spatial_only:   bool  = True,
        max_age_s:            float = 0.0,
        max_entries:          int   = 0,
    ) -> None:
        self._spatial_gate_m       = float(spatial_gate_m)
        self._similarity_threshold = float(similarity_threshold)
        self._allow_spatial_only   = bool(allow_spatial_only)
        self._max_age_s            = float(max_age_s)
        self._max_entries          = int(max_entries)
        self._entries: Dict[int, WorldMapEntry] = {}
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def n_entries(self) -> int:
        return len(self._entries)

    def all_entries(self) -> List[WorldMapEntry]:
        return list(self._entries.values())

    def get(self, persistent_id: int) -> Optional[WorldMapEntry]:
        return self._entries.get(persistent_id)

    # ------------------------------------------------------------------
    # Insert / re-associate
    # ------------------------------------------------------------------

    def insert(
        self,
        class_name:     str,
        class_id:       int,
        position_world: np.ndarray,
        embedding:      Optional[np.ndarray],
        stability:      StabilityClass,
        timestamp:      float,
    ) -> WorldMapEntry:
        """Add a fresh entry and return it. Assigns a new persistent_id."""
        entry = WorldMapEntry(
            persistent_id=self._next_id,
            class_name=class_name,
            class_id=class_id,
            position_world=np.asarray(position_world, dtype=np.float64),
            embedding=(np.asarray(embedding, dtype=np.float32)
                       if embedding is not None else None),
            stability=stability,
            first_seen=timestamp,
            last_seen=timestamp,
        )
        self._entries[self._next_id] = entry
        self._next_id += 1
        self._maybe_prune(timestamp)
        return entry

    def re_associate(
        self,
        class_name:     str,
        position_world: np.ndarray,
        embedding:      Optional[np.ndarray],
    ) -> Optional[WorldMapEntry]:
        """
        Search for a matching persistent entry. Returns the best match
        if both gates pass, else None.

        Both gates required when both sides have embeddings. When
        either side lacks an embedding and `allow_spatial_only` is
        True, falls back to spatial-only (nearest within the gate).
        """
        pos = np.asarray(position_world, dtype=np.float64)
        best: Optional[Tuple[float, WorldMapEntry]] = None

        for entry in self._entries.values():
            if entry.class_name != class_name:
                continue
            dist = float(np.linalg.norm(entry.position_world - pos))
            if dist > self._spatial_gate_m:
                continue

            if embedding is not None and entry.embedding is not None:
                sim = float(np.dot(embedding, entry.embedding))
                if sim < self._similarity_threshold:
                    continue
                # Combined score: stronger similarity wins ties at
                # equal distance. Prefer higher similarity overall.
                score = -sim
            else:
                if not self._allow_spatial_only:
                    continue
                score = dist   # spatial fallback: closest wins

            if best is None or score < best[0]:
                best = (score, entry)

        return best[1] if best is not None else None

    def insert_or_re_associate(
        self,
        class_name:     str,
        class_id:       int,
        position_world: np.ndarray,
        embedding:      Optional[np.ndarray],
        stability:      StabilityClass,
        timestamp:      float,
    ) -> Tuple[WorldMapEntry, bool]:
        """
        Convenience: try re-association; on success update the entry,
        on failure insert a new one.

        Returns
        -------
        (entry, was_new) where was_new=True means a fresh persistent_id
        was assigned. Callers use this to decide whether to keep the
        live track's ID or replace it with entry.persistent_id.
        """
        existing = self.re_associate(class_name, position_world, embedding)
        if existing is not None:
            existing.update(position_world, embedding, timestamp)
            self._maybe_prune(timestamp)
            return existing, False
        return self.insert(
            class_name=class_name, class_id=class_id,
            position_world=position_world, embedding=embedding,
            stability=stability, timestamp=timestamp,
        ), True

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def query_nearby(
        self,
        position: np.ndarray,
        radius:   float,
    ) -> List[Tuple[float, WorldMapEntry]]:
        """All entries within `radius` metres of `position`, sorted near→far."""
        pos = np.asarray(position, dtype=np.float64)
        results = []
        for entry in self._entries.values():
            dist = float(np.linalg.norm(entry.position_world - pos))
            if dist <= radius:
                results.append((dist, entry))
        results.sort(key=lambda t: t[0])
        return results

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _maybe_prune(self, now: float) -> int:
        """Apply age + capacity eviction. Returns total entries dropped."""
        n_dropped = 0
        if self._max_age_s > 0.0:
            n_dropped += self._prune_by_age(now)
        if self._max_entries > 0 and len(self._entries) > self._max_entries:
            n_dropped += self._prune_by_capacity()
        return n_dropped

    def _prune_by_age(self, now: float) -> int:
        cutoff = now - self._max_age_s
        stale = [pid for pid, e in self._entries.items()
                 if e.last_seen < cutoff]
        for pid in stale:
            del self._entries[pid]
        return len(stale)

    def _prune_by_capacity(self) -> int:
        # LRU by last_seen — oldest first. Cap is a hard ceiling, so
        # drop exactly the overflow count.
        overflow = len(self._entries) - self._max_entries
        if overflow <= 0:
            return 0
        ordered = sorted(self._entries.items(),
                         key=lambda kv: kv[1].last_seen)
        for pid, _entry in ordered[:overflow]:
            del self._entries[pid]
        return overflow

    def prune(self, now: float) -> int:
        """
        Public hook to apply eviction outside the insert path — useful
        if the caller wants to age stale entries periodically even when
        no insert is happening. Returns total dropped.
        """
        return self._maybe_prune(now)

    def reset(self) -> None:
        """Drop all entries. Call between deployments."""
        self._entries.clear()
        self._next_id = 1

    def __repr__(self) -> str:
        return (
            f"WorldMap(n_entries={self.n_entries}, "
            f"gate={self._spatial_gate_m} m, "
            f"sim_thr={self._similarity_threshold})"
        )
