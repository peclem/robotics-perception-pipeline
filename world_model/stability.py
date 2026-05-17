"""
Stability classification — distinguishes static, semi-static, and
dynamic objects to drive per-class memory durations in the SceneGraph.

Why
---
The single global `lost_timeout_s` treats a person and a chair the
same: forget after 1.5 s. That throws away genuinely useful state.
A chair seen 30 s ago is almost certainly still where it was; a
person seen 30 s ago could be anywhere. Per-class persistence captures
that asymmetry.

Three stability classes
-----------------------
STATIC      — typically fixed in the world: furniture, infrastructure.
              Persist indefinitely in the SceneGraph; never pruned by
              the lost-timeout mechanism. Use STATIC as the *initial*
              class label only; motion observations can demote.
SEMI_STATIC — can move but mostly doesn't: parked vehicles, bicycles,
              displaced chairs, containers. Persist for minutes.
DYNAMIC     — moving agents: people, animals, vehicles in motion.
              Short persistence (current default 1.5 s), as before.

Classification policy
---------------------
1. **Initial class prior** — from the COCO class table below. The
   COCO classes the YOLO detector emits map cleanly to one of the
   three stability classes for the common cases.
2. **Motion-based override** — observed velocity history can promote
   or demote a track. A STATIC-classified chair that's actually being
   pushed across the room demotes to DYNAMIC; a DYNAMIC-classified
   person standing still for 30+ s promotes to SEMI_STATIC. Both
   transitions use hysteresis to avoid flipping.

Literature anchor
-----------------
The static/dynamic distinction is the foundation of multi-layered
costmaps (Lu/Smart/Pinciroli, ICRA 2014) and dynamic-aware SLAM
(DynaSLAM — Bescos et al. 2018; Kimera-Semantics — Rosinol 2020).
Hybrid "class prior + motion override" is the consensus approach.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict


class StabilityClass(IntEnum):
    """
    Per-object memory-persistence class. Numeric values are ordered
    by persistence (low → ephemeral, high → persistent) so that
    `state.stability >= SEMI_STATIC` is a meaningful query.
    """
    DYNAMIC     = 0
    SEMI_STATIC = 1
    STATIC      = 2


# Per-stability default lost-timeout, seconds. Overridable in config.
# float('inf') means "never prune by timeout" — only evicted by an
# explicit world-map eviction policy (Pass B).
DEFAULT_TIMEOUTS_S: Dict[StabilityClass, float] = {
    StabilityClass.DYNAMIC:     1.5,
    StabilityClass.SEMI_STATIC: 60.0,
    StabilityClass.STATIC:      float("inf"),
}


# COCO class → stability prior. Covers all 80 standard COCO classes
# the YOLO detector emits. Ambiguous cases (vehicles, sports items,
# kitchenware) default to SEMI_STATIC and rely on motion override.
DEFAULT_COCO_STABILITY: Dict[str, StabilityClass] = {
    # Moving agents
    "person":        StabilityClass.DYNAMIC,
    "bird":          StabilityClass.DYNAMIC,
    "cat":           StabilityClass.DYNAMIC,
    "dog":           StabilityClass.DYNAMIC,
    "horse":         StabilityClass.DYNAMIC,
    "sheep":         StabilityClass.DYNAMIC,
    "cow":           StabilityClass.DYNAMIC,
    "elephant":      StabilityClass.DYNAMIC,
    "bear":          StabilityClass.DYNAMIC,
    "zebra":         StabilityClass.DYNAMIC,
    "giraffe":       StabilityClass.DYNAMIC,

    # Vehicles — semi-static unless observed moving
    "bicycle":       StabilityClass.SEMI_STATIC,
    "car":           StabilityClass.SEMI_STATIC,
    "motorcycle":    StabilityClass.SEMI_STATIC,
    "airplane":      StabilityClass.SEMI_STATIC,
    "bus":           StabilityClass.SEMI_STATIC,
    "train":         StabilityClass.SEMI_STATIC,
    "truck":         StabilityClass.SEMI_STATIC,
    "boat":          StabilityClass.SEMI_STATIC,

    # Carryable / handheld
    "backpack":      StabilityClass.SEMI_STATIC,
    "umbrella":      StabilityClass.SEMI_STATIC,
    "handbag":       StabilityClass.SEMI_STATIC,
    "tie":           StabilityClass.SEMI_STATIC,
    "suitcase":      StabilityClass.SEMI_STATIC,
    "frisbee":       StabilityClass.SEMI_STATIC,
    "skis":          StabilityClass.SEMI_STATIC,
    "snowboard":     StabilityClass.SEMI_STATIC,
    "sports ball":   StabilityClass.SEMI_STATIC,
    "kite":          StabilityClass.SEMI_STATIC,
    "baseball bat":  StabilityClass.SEMI_STATIC,
    "baseball glove":StabilityClass.SEMI_STATIC,
    "skateboard":    StabilityClass.SEMI_STATIC,
    "surfboard":     StabilityClass.SEMI_STATIC,
    "tennis racket": StabilityClass.SEMI_STATIC,
    "bottle":        StabilityClass.SEMI_STATIC,
    "wine glass":    StabilityClass.SEMI_STATIC,
    "cup":           StabilityClass.SEMI_STATIC,
    "fork":          StabilityClass.SEMI_STATIC,
    "knife":         StabilityClass.SEMI_STATIC,
    "spoon":         StabilityClass.SEMI_STATIC,
    "bowl":          StabilityClass.SEMI_STATIC,

    # Street furniture / infrastructure — static
    "traffic light": StabilityClass.STATIC,
    "fire hydrant":  StabilityClass.STATIC,
    "stop sign":     StabilityClass.STATIC,
    "parking meter": StabilityClass.STATIC,
    "bench":         StabilityClass.STATIC,

    # Indoor furniture — static
    "chair":         StabilityClass.STATIC,
    "couch":         StabilityClass.STATIC,
    "potted plant":  StabilityClass.STATIC,
    "bed":           StabilityClass.STATIC,
    "dining table":  StabilityClass.STATIC,
    "toilet":        StabilityClass.STATIC,

    # Appliances / electronics — static
    "tv":            StabilityClass.STATIC,
    "laptop":        StabilityClass.STATIC,
    "mouse":         StabilityClass.STATIC,
    "remote":        StabilityClass.STATIC,
    "keyboard":      StabilityClass.STATIC,
    "cell phone":    StabilityClass.STATIC,
    "microwave":     StabilityClass.STATIC,
    "oven":          StabilityClass.STATIC,
    "toaster":       StabilityClass.STATIC,
    "sink":          StabilityClass.STATIC,
    "refrigerator":  StabilityClass.STATIC,

    # Small static objects
    "book":          StabilityClass.STATIC,
    "clock":         StabilityClass.STATIC,
    "vase":          StabilityClass.STATIC,
    "scissors":      StabilityClass.STATIC,
    "teddy bear":    StabilityClass.STATIC,
    "hair drier":    StabilityClass.STATIC,
    "toothbrush":    StabilityClass.STATIC,

    # Food (typically static while observed)
    "banana":        StabilityClass.STATIC,
    "apple":         StabilityClass.STATIC,
    "sandwich":      StabilityClass.STATIC,
    "orange":        StabilityClass.STATIC,
    "broccoli":      StabilityClass.STATIC,
    "carrot":        StabilityClass.STATIC,
    "hot dog":       StabilityClass.STATIC,
    "pizza":         StabilityClass.STATIC,
    "donut":         StabilityClass.STATIC,
    "cake":          StabilityClass.STATIC,
}


def stability_for_class(
    class_name: str,
    overrides:  Dict[str, StabilityClass] | None = None,
) -> StabilityClass:
    """
    Look up the stability prior for a COCO class. Unknown classes
    default to SEMI_STATIC — conservative middle ground that the
    motion override will refine once velocity is observed.
    """
    if overrides is not None and class_name in overrides:
        return overrides[class_name]
    return DEFAULT_COCO_STABILITY.get(class_name, StabilityClass.SEMI_STATIC)


def timeout_for_stability(
    stability: StabilityClass,
    overrides: Dict[StabilityClass, float] | None = None,
) -> float:
    """
    Return the lost-timeout (seconds) for a stability class. inf means
    "never prune by timeout" — the object stays in the SceneGraph until
    a different eviction policy removes it (Pass B WorldMap layer).
    """
    if overrides is not None and stability in overrides:
        return overrides[stability]
    return DEFAULT_TIMEOUTS_S[stability]
