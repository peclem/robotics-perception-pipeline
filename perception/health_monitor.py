"""
Per-stage latency monitoring + graceful-degradation signalling.

Why
---
Silent latency drift is one of the most common production-failure
modes. The detector hits a thermal throttle, the depth model gets
swapped to a heavier checkpoint, an OS page-fault burst stalls a
frame — none of these surface unless someone is watching. Then
suddenly the planner's 5-Hz tick is consuming 10-frame-old data and
the robot does something stupid.

This module wraps each pipeline stage with a LatencyTracker, compares
observed latency against a per-stage budget, and tracks the result as
an OK/WARN/ERROR status. The aggregated HealthMonitor exposes the
state for downstream consumers (a ROS2 node, a metrics endpoint, a
log dump, or whatever).

Status model
------------
    OK    : recent median ≤ budget
    WARN  : recent median > budget for fewer than `error_after` frames
    ERROR : recent median > budget for ≥ `error_after` frames
    STALE : no observation in the last `stale_after_s` seconds
            (component is probably dead or stuck)

The WARN → ERROR escalation gives operators time to react before the
status is hard-flagged. STALE catches "component disappeared without
saying anything" — silence is the worst failure mode.

The exact thresholds are deployment-specific; defaults assume a
30 Hz pipeline targeting a ~33 ms total budget.

Reference
---------
ROS diagnostics conventions: diagnostic_msgs/DiagnosticArray + the
diagnostic_updater Python library. We don't depend on those here
(this module is pure Python); the ROS2 wrapper translates to them.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Deque, Dict, Iterator, Optional

log = logging.getLogger(__name__)


class HealthStatus(IntEnum):
    """
    Standard diagnostics ordering (matches diagnostic_msgs/DiagnosticStatus):
        OK = 0, WARN = 1, ERROR = 2, STALE = 3.
    """
    OK    = 0
    WARN  = 1
    ERROR = 2
    STALE = 3


@dataclass
class StageReport:
    """
    Snapshot of one pipeline stage's health.

    median_ms / p95_ms / max_ms are computed over the rolling window
    of recent observations. n_observations counts everything since
    creation (not just the window).
    """
    name:            str
    status:          HealthStatus
    median_ms:       float
    p95_ms:          float
    max_ms:          float
    last_ms:         float
    budget_ms:       float
    n_observations:  int
    n_breaches:      int          # consecutive breaches at report time
    last_observation_s: float     # monotonic time of most recent observe()
    message:         str          # human-readable summary


class LatencyTracker:
    """
    Rolling-window latency tracker for one pipeline stage.

    Use as a context manager around the work to be timed:

        with monitor.stage("detector"):
            detections = self._detector.detect(frame)

    Or call observe(dt_ms) directly when wrapping non-block code.

    Parameters
    ----------
    name        : stage identifier (used in reports)
    budget_ms   : per-frame latency budget. Observations exceeding
                  this contribute to the breach counter.
    window      : rolling-window size for stats. Default 60 ≈ 2 s at 30 Hz.
    warn_after  : breaches needed before status leaves OK
    error_after : breaches needed before status escalates to ERROR
    stale_after_s : if no observation in this many seconds, STALE
    """

    def __init__(
        self,
        name:           str,
        budget_ms:      float,
        window:         int   = 60,
        warn_after:     int   = 3,
        error_after:    int   = 30,
        stale_after_s:  float = 5.0,
    ) -> None:
        self.name             = name
        self.budget_ms        = float(budget_ms)
        self._window          = int(window)
        self._warn_after      = int(warn_after)
        self._error_after     = int(error_after)
        self._stale_after_s   = float(stale_after_s)

        self._samples: Deque[float] = deque(maxlen=self._window)
        self._n_observations: int   = 0
        self._consecutive_breaches: int = 0
        self._last_observation_s: float = time.monotonic()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def observe(self, latency_ms: float) -> None:
        """Record one observation (milliseconds)."""
        self._samples.append(float(latency_ms))
        self._n_observations += 1
        self._last_observation_s = time.monotonic()
        if latency_ms > self.budget_ms:
            self._consecutive_breaches += 1
        else:
            self._consecutive_breaches = 0

    def __call__(self) -> "_StageTimerCM":
        """Convenience: `with tracker(): ...` instead of `tracker.observe`."""
        return _StageTimerCM(self)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> StageReport:
        now = time.monotonic()
        samples = list(self._samples)

        if not samples:
            return StageReport(
                name=self.name,
                status=HealthStatus.STALE,
                median_ms=0.0, p95_ms=0.0, max_ms=0.0, last_ms=0.0,
                budget_ms=self.budget_ms,
                n_observations=0, n_breaches=0,
                last_observation_s=self._last_observation_s,
                message="no observations yet",
            )

        if now - self._last_observation_s > self._stale_after_s:
            return StageReport(
                name=self.name, status=HealthStatus.STALE,
                median_ms=_median(samples), p95_ms=_p95(samples),
                max_ms=max(samples), last_ms=samples[-1],
                budget_ms=self.budget_ms,
                n_observations=self._n_observations,
                n_breaches=self._consecutive_breaches,
                last_observation_s=self._last_observation_s,
                message=(
                    f"no observations in "
                    f"{now - self._last_observation_s:.1f} s "
                    f"(stale threshold {self._stale_after_s} s)"
                ),
            )

        if self._consecutive_breaches >= self._error_after:
            status, msg = HealthStatus.ERROR, (
                f"latency over budget for {self._consecutive_breaches} "
                f"consecutive observations (threshold {self._error_after})"
            )
        elif self._consecutive_breaches >= self._warn_after:
            status, msg = HealthStatus.WARN, (
                f"latency over budget for {self._consecutive_breaches} "
                f"consecutive observations (warn at {self._warn_after})"
            )
        else:
            status, msg = HealthStatus.OK, "within budget"

        return StageReport(
            name=self.name, status=status,
            median_ms=_median(samples), p95_ms=_p95(samples),
            max_ms=max(samples), last_ms=samples[-1],
            budget_ms=self.budget_ms,
            n_observations=self._n_observations,
            n_breaches=self._consecutive_breaches,
            last_observation_s=self._last_observation_s,
            message=msg,
        )

    def reset(self) -> None:
        """Drop samples + breach counter. Call between sessions."""
        self._samples.clear()
        self._n_observations = 0
        self._consecutive_breaches = 0
        self._last_observation_s = time.monotonic()


class _StageTimerCM:
    """Internal: context manager that calls observe() on exit."""
    __slots__ = ("_tracker", "_t0")

    def __init__(self, tracker: LatencyTracker):
        self._tracker = tracker
        self._t0 = 0.0

    def __enter__(self) -> "_StageTimerCM":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        self._tracker.observe(dt_ms)


class HealthMonitor:
    """
    Aggregates LatencyTracker reports across the whole pipeline.

    Usage
    -----
        monitor = HealthMonitor()
        monitor.register("detector", budget_ms=8.0)
        monitor.register("depth",    budget_ms=15.0)
        monitor.register("tracker",  budget_ms=2.0)

        # In the frame loop:
        with monitor.stage("detector"):
            ...

        # Periodically (e.g. from a ROS2 timer):
        for report in monitor.reports():
            publish_to_diagnostics(report)

    The overall pipeline status is the max (worst) of individual
    stage statuses — `monitor.overall_status()` returns that.
    """

    def __init__(self) -> None:
        self._trackers: Dict[str, LatencyTracker] = {}

    def register(
        self,
        name:       str,
        budget_ms:  float,
        **kwargs,
    ) -> LatencyTracker:
        """Add a stage to monitor. Returns the LatencyTracker."""
        tracker = LatencyTracker(name=name, budget_ms=budget_ms, **kwargs)
        self._trackers[name] = tracker
        return tracker

    def stage(self, name: str) -> _StageTimerCM:
        """Context-manager shortcut for registered tracker."""
        try:
            return self._trackers[name]()
        except KeyError:
            raise KeyError(
                f"HealthMonitor: stage '{name}' not registered. "
                f"Known: {sorted(self._trackers)}"
            )

    def get(self, name: str) -> Optional[LatencyTracker]:
        return self._trackers.get(name)

    @property
    def stages(self) -> list[str]:
        return list(self._trackers)

    def reports(self) -> Iterator[StageReport]:
        for tracker in self._trackers.values():
            yield tracker.report()

    def overall_status(self) -> HealthStatus:
        if not self._trackers:
            return HealthStatus.STALE
        # Worst (numerically highest) status wins.
        return max(r.status for r in self.reports())

    def summary_line(self) -> str:
        """One-line human-readable summary; useful for periodic logs."""
        parts = []
        for r in self.reports():
            parts.append(
                f"{r.name}={r.status.name}/{r.median_ms:.1f}ms"
                f"(b{r.budget_ms:.0f})"
            )
        return "  ".join(parts) if parts else "(no stages registered)"


# ---------------------------------------------------------------------------
# Stats helpers — small, no dependency
# ---------------------------------------------------------------------------

def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    # Nearest-rank p95 — fine for diagnostic granularity.
    idx = min(len(s) - 1, max(0, int(round(0.95 * (len(s) - 1)))))
    return s[idx]
