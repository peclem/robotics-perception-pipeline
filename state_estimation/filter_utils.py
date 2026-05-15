"""
Filter consistency diagnostics.

NIS  — Normalised Innovation Squared  (external, uses measurements)
NEES — Normalised Estimation Error Squared (internal, needs ground truth)

Both are chi-squared distributed under a consistent filter.
Use NIS in production (no ground truth needed).
Use NEES in simulation to fully validate the filter.

Reference
---------
Bar-Shalom, Li, Kirubarajan — Estimation with Applications to
Tracking and Navigation (2001), Chapter 5.
"""

from __future__ import annotations

from typing import List

import numpy as np


def compute_nis_statistics(nis_values: List[float]) -> dict:
    """
    Compute NIS statistics over a sequence of filter updates.

    Parameters
    ----------
    nis_values : list of NIS scalars from kf.nis() over many frames

    Returns
    -------
    dict with keys:
        mean      : mean NIS (should be ≈ n_obs = 4 for consistent filter)
        std       : standard deviation
        pct_in_bounds : fraction in χ²(4) 95% CI [0.711, 9.488]
        n_samples : number of samples

    Interpretation
    --------------
    mean NIS ≈ 4.0  → consistent
    mean NIS >> 4.0 → overconfident (Q too small, R too large)
    mean NIS << 4.0 → underconfident (Q too large, R too small)
    """
    values = [v for v in nis_values if not np.isnan(v)]
    if not values:
        return {"mean": float("nan"), "std": float("nan"),
                "pct_in_bounds": float("nan"), "n_samples": 0}

    arr = np.array(values)
    in_bounds = np.sum((arr >= 0.711) & (arr <= 9.488))

    return {
        "mean":          float(arr.mean()),
        "std":           float(arr.std()),
        "pct_in_bounds": float(in_bounds / len(arr)),
        "n_samples":     len(arr),
    }


def compute_nees_statistics(nees_values: List[float]) -> dict:
    """
    Compute NEES statistics over a simulation run.

    Parameters
    ----------
    nees_values : list of NEES scalars from kf.nees(true_state)

    Returns
    -------
    dict with keys:
        mean         : mean NEES (should be ≈ n_state for full state,
                       ≈ n_obs for position-only check)
        std          : standard deviation
        n_samples    : number of samples
    """
    values = [v for v in nees_values if not np.isnan(v)]
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n_samples": 0}

    arr = np.array(values)
    return {
        "mean":      float(arr.mean()),
        "std":       float(arr.std()),
        "n_samples": len(arr),
    }


def chi2_bounds(dof: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Return chi-squared lower and upper bounds for consistency check.

    Parameters
    ----------
    dof        : degrees of freedom (= observation dimension for NIS,
                 = state dimension for NEES)
    confidence : confidence level (default 0.95 = 95% CI)

    Returns
    -------
    (lower, upper) tuple of chi-squared bounds

    Note: uses scipy if available, otherwise returns precomputed
    values for common DOF.
    """
    try:
        from scipy.stats import chi2
        lo = float(chi2.ppf(1.0 - confidence, dof))
        hi = float(chi2.ppf(confidence,        dof))
        return lo, hi
    except ImportError:
        pass

    # Precomputed 95% CI values for common DOF
    precomputed = {
        2: (0.051, 7.378),
        4: (0.711, 9.488),
        8: (2.733, 15.507),
        9: (3.325, 16.919),
    }
    if dof in precomputed:
        return precomputed[dof]

    # Rough approximation for other DOF
    return (max(0.0, dof - 3 * np.sqrt(2 * dof)),
            dof + 3 * np.sqrt(2 * dof))
