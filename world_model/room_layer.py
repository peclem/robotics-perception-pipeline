"""
RoomLayer — top-of-graph layer above the flat SceneGraph object list.

A 3D scene graph in the Hydra / Kimera-DSG / ConceptGraphs sense is a
hierarchy: world ⊃ buildings ⊃ floors ⊃ rooms ⊃ objects. This module
adds the *room* layer to the existing flat object list. Each room is a
labelled 2D polygon in world-XY metres carrying the set of object IDs
spatially contained in it; queries like "what room is the user in?" or
"who's in the kitchen?" become O(rooms) polygon containment instead of
O(objects) brute-force.

How rooms are extracted
-----------------------
Standard indoor-mapping recipe — the same one Hydra builds on top of:

1. **Threshold the occupancy grid into walls vs free space.** Walls are
   the OCCUPIED cells (100 in the nav_msgs/OccupancyGrid value
   convention); free is everything below.
2. **Morphologically erode the free-space mask.** This breaks doorway
   connections between adjacent rooms (doors are typically 80–100 cm
   wide; eroding by ~half a doorway both sides closes them) while
   preserving room interiors. Erosion radius is a config knob.
3. **Connected-component label** the eroded free space. Each component
   is a candidate room.
4. **Filter small components by area** (< min_area_m2). Wipes out
   single-cell noise and tiny disconnected pockets behind furniture.
5. **Dilate the surviving labels back** so room polygons recover the
   full free-space extent inside their boundary.
6. **Extract polygons** via cv2.findContours on each labelled region
   independently, simplified with cv2.approxPolyDP for compactness.

This is intentionally NOT a Voronoi / watershed / Hough-line approach —
those are more principled but add code without buying much on the
indoor-occupancy-grids we have today. The morphological pipeline is
what voxblox/Hydra start with and is enough for any showcase demo on
synthetic or hand-held-glasses data.

What v1 does NOT do
-------------------
- **No persistent room IDs across frames.** Rooms are rebuilt fresh
  from each occupancy snapshot; IDs reshuffle when the geometry
  changes. A persistent-room tracker (Hungarian on Jaccard overlap)
  is the natural follow-up but requires a SLAM-grade global frame to
  be useful, so it's deferred until ego-pose drift is controlled.
- **No semantic room labels** ("kitchen" / "office"). The
  `RoomLayer` exposes the polygon + object-id set; assigning a label
  from the dominant semantic class of contained objects + walls is a
  cheap follow-up.
- **No multi-floor handling.** Rooms are flat polygons in world-XY.
  Floor separation needs Z-aware clustering — slot in when needed.

References
----------
Hughes et al. (2022) — Hydra: A real-time spatial perception system
                       for 3D scene graph construction & optimization.
                       RSS 2022.
Rosinol et al. (2020) — Kimera: an open-source library for real-time
                        metric-semantic localization and mapping.
                        ICRA 2020.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from world_model.object_state import ObjectState
from world_model.occupancy_grid import OCCUPIED, OccupancyGridParams

log = logging.getLogger(__name__)


@dataclass
class Room:
    """
    One room: a labelled 2D polygon in world-XY metres + the set of
    object track IDs spatially contained inside it.

    Attributes
    ----------
    room_id      : per-frame id (1, 2, ...), NOT stable across frames.
    label        : human-readable; defaults to "room_{id}". Reserved
                   for future semantic labelling.
    polygon_world: (N, 2) float64 ordered vertices in world-XY metres.
    centroid     : (2,) float64 world-XY centroid in metres.
    area_m2      : polygon area in square metres.
    bbox_world   : (xmin, ymin, xmax, ymax) AABB in world metres.
    object_ids   : track IDs of objects whose position_world lies inside
                   `polygon_world`.
    """
    room_id:       int
    polygon_world: np.ndarray
    centroid:      np.ndarray
    area_m2:       float
    bbox_world:    Tuple[float, float, float, float]
    object_ids:    set = field(default_factory=set)
    label:         str = ""

    def __post_init__(self) -> None:
        self.polygon_world = np.asarray(self.polygon_world, dtype=np.float64)
        self.centroid      = np.asarray(self.centroid,      dtype=np.float64)
        assert self.polygon_world.ndim == 2 and self.polygon_world.shape[1] == 2, (
            f"polygon_world must be (N, 2), got {self.polygon_world.shape}"
        )
        assert self.centroid.shape == (2,)
        if not self.label:
            self.label = f"room_{self.room_id}"

    def contains_world_xy(self, x: float, y: float) -> bool:
        """True iff (x, y) is inside the room polygon (cv2 point-polygon test)."""
        # Fast AABB reject first.
        x0, y0, x1, y1 = self.bbox_world
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return False
        # findContours expects integer pixel coords; for world-frame
        # polygon containment use cv2.pointPolygonTest, which handles
        # float64 contours.
        poly_for_cv = self.polygon_world.astype(np.float32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(poly_for_cv, (float(x), float(y)), False) >= 0


@dataclass
class RoomLayerConfig:
    """
    Tuning knobs for the room-extraction pipeline.

    erosion_m
        Half of the maximum doorway width you want to close. A typical
        residential doorway is 0.8–1.0 m wide → erode by 0.4–0.5 m to
        disconnect adjacent rooms through the doorway.
    min_area_m2
        Smaller candidate rooms are discarded. ~1 m² wipes out noise
        without losing real small rooms (closets are usually 1.5–2 m²).
    polygon_simplify_m
        Tolerance for cv2.approxPolyDP polygon simplification. ~5 cm
        keeps the polygon close to the true wall while staying compact.
    """
    erosion_m:          float = 0.45
    min_area_m2:        float = 1.0
    polygon_simplify_m: float = 0.05


class RoomLayer:
    """
    Extracts rooms from a 2D occupancy grid and assigns objects to them.

    Stateless across frames (v1). Call `update(grid, params, objects,
    frame_idx)` each frame; the previous snapshot is replaced.

    Construction
    ------------
        layer = RoomLayer(RoomLayerConfig())
        layer.update(grid_array, OccupancyGridParams(), objects, frame_idx)
        for r in layer.rooms:
            ...
        layer.room_of_object(track_id) -> Optional[Room]
        layer.room_of_world_xy(x, y)   -> Optional[Room]
    """

    def __init__(self, cfg: Optional[RoomLayerConfig] = None) -> None:
        self._cfg     = cfg or RoomLayerConfig()
        self._rooms:  List[Room] = []
        self._object_to_room: dict = {}     # track_id → room_id
        self._last_frame_idx: int = -1

    @property
    def rooms(self) -> List[Room]:
        return self._rooms

    @property
    def last_frame_idx(self) -> int:
        return self._last_frame_idx

    def update(
        self,
        grid:       np.ndarray,
        params:     OccupancyGridParams,
        objects:    Iterable[ObjectState],
        frame_idx:  int,
    ) -> None:
        """
        Rebuild the room snapshot from the current occupancy grid +
        objects.

        Parameters
        ----------
        grid       : (H, W) int8 occupancy grid (OCCUPIED = 100, else free).
        params     : geometry parameters matching `grid` (resolution,
                     origin). The same OccupancyGridParams that produced
                     the grid.
        objects    : iterable of ObjectState. Only those with
                     `position_world is not None` are assigned to rooms;
                     the rest are silently skipped.
        frame_idx  : caller's frame index — stored for telemetry.
        """
        self._last_frame_idx = int(frame_idx)
        self._rooms = self._extract_rooms(grid, params)
        self._object_to_room = self._assign_objects(objects)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def room_of_object(self, track_id: int) -> Optional[Room]:
        """Return the Room containing the given object, or None."""
        rid = self._object_to_room.get(int(track_id))
        if rid is None:
            return None
        for r in self._rooms:
            if r.room_id == rid:
                return r
        return None

    def room_of_world_xy(self, x: float, y: float) -> Optional[Room]:
        """Return the room containing (x, y) in world metres, or None."""
        for r in self._rooms:
            if r.contains_world_xy(x, y):
                return r
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_rooms(
        self,
        grid:   np.ndarray,
        params: OccupancyGridParams,
    ) -> List[Room]:
        if grid.size == 0:
            return []
        # Free-space mask: True wherever the cell is NOT marked OCCUPIED.
        # Cast to uint8 for cv2.
        free = (grid != OCCUPIED).astype(np.uint8) * 255

        # Erode to disconnect doorways. Round-up the radius to cells.
        radius_cells = max(1, int(np.round(self._cfg.erosion_m / params.resolution_m)))
        kernel_size  = 2 * radius_cells + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size),
        )
        eroded = cv2.erode(free, kernel)

        # Connected components on the eroded mask. Background label is 0.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            eroded, connectivity=8,
        )

        # Discard the background label (index 0). For each surviving
        # component, dilate its individual mask back, contour, simplify,
        # and convert to world metres.
        rooms: List[Room] = []
        min_area_cells = self._cfg.min_area_m2 / (params.resolution_m ** 2)
        simplify_cells = max(1.0, self._cfg.polygon_simplify_m / params.resolution_m)
        next_id = 1
        for lbl in range(1, n_labels):
            area_cells = int(stats[lbl, cv2.CC_STAT_AREA])
            if area_cells < min_area_cells:
                continue
            component = (labels == lbl).astype(np.uint8) * 255
            # Dilate back so polygon recovers the pre-erosion extent.
            grown = cv2.dilate(component, kernel)
            # findContours returns OUTER contour as (N, 1, 2) int32 with
            # axes (col, row) i.e. (x_cell, y_cell).
            contours, _ = cv2.findContours(
                grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue
            # Largest contour = the room (others would be holes;
            # RETR_EXTERNAL already strips holes, so contours[0] is
            # always the outer ring here).
            contour = max(contours, key=cv2.contourArea)
            simplified = cv2.approxPolyDP(
                contour, epsilon=float(simplify_cells), closed=True,
            ).reshape(-1, 2)
            if simplified.shape[0] < 3:
                continue
            # Convert cell coordinates → world metres. Cell (i, j) →
            # world (origin_x + (i+0.5)*res, origin_y + (j+0.5)*res).
            # Note simplified[:, 0] is cell-x = col; simplified[:, 1] is
            # cell-y = row.
            polygon_world = np.column_stack([
                params.origin_x_m + (simplified[:, 0] + 0.5) * params.resolution_m,
                params.origin_y_m + (simplified[:, 1] + 0.5) * params.resolution_m,
            ]).astype(np.float64)
            area_m2 = float(
                cv2.contourArea(polygon_world.astype(np.float32))
            )
            centroid = polygon_world.mean(axis=0)
            x_min, y_min = polygon_world.min(axis=0)
            x_max, y_max = polygon_world.max(axis=0)
            rooms.append(Room(
                room_id=next_id,
                polygon_world=polygon_world,
                centroid=centroid,
                area_m2=area_m2,
                bbox_world=(float(x_min), float(y_min),
                            float(x_max), float(y_max)),
            ))
            next_id += 1
        return rooms

    def _assign_objects(
        self,
        objects: Iterable[ObjectState],
    ) -> dict:
        mapping: dict = {}
        for obj in objects:
            if obj.position_world is None:
                continue
            xy = obj.position_world[:2]
            r = self.room_of_world_xy(float(xy[0]), float(xy[1]))
            if r is not None:
                mapping[int(obj.track_id)] = r.room_id
                r.object_ids.add(int(obj.track_id))
        return mapping

    def __repr__(self) -> str:
        return (
            f"RoomLayer(n_rooms={len(self._rooms)}, "
            f"frame={self._last_frame_idx})"
        )
