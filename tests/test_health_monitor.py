"""
Tests for the health monitor / latency tracker.

TestLatencyTracker  : sample recording, context manager, breach counter
TestStatusEscalation: OK → WARN → ERROR thresholds with hysteresis
TestStaleDetection  : missing observations flip to STALE
TestHealthMonitor   : aggregation + overall_status + summary
"""

from __future__ import annotations

import time

import pytest

from perception.health_monitor import (
    HealthMonitor, HealthStatus, LatencyTracker, StageReport,
)


# ---------------------------------------------------------------------------
# LatencyTracker
# ---------------------------------------------------------------------------

class TestLatencyTracker:

    def test_initial_status_is_stale_with_no_observations(self):
        r = LatencyTracker("x", budget_ms=10.0).report()
        assert r.status == HealthStatus.STALE
        assert r.n_observations == 0

    def test_under_budget_stays_ok(self):
        t = LatencyTracker("x", budget_ms=10.0)
        for _ in range(20):
            t.observe(5.0)
        r = t.report()
        assert r.status == HealthStatus.OK
        assert r.median_ms == 5.0
        assert r.n_observations == 20

    def test_context_manager_records_elapsed_time(self):
        t = LatencyTracker("x", budget_ms=10.0)
        with t():
            time.sleep(0.001)   # ~1 ms
        r = t.report()
        assert r.last_ms > 0
        assert r.n_observations == 1

    def test_rolling_window_caps_samples(self):
        t = LatencyTracker("x", budget_ms=10.0, window=5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
            t.observe(v)
        # Median over last 5 (3..7) = 5.0
        assert t.report().median_ms == 5.0
        # But n_observations counts everything
        assert t.report().n_observations == 7

    def test_breach_counter_resets_on_under_budget(self):
        t = LatencyTracker("x", budget_ms=10.0,
                           warn_after=3, error_after=10)
        for _ in range(5):
            t.observe(20.0)         # breach
        t.observe(5.0)              # reset
        for _ in range(2):
            t.observe(20.0)         # only 2 again — under warn_after
        assert t.report().status == HealthStatus.OK


# ---------------------------------------------------------------------------
# Status escalation
# ---------------------------------------------------------------------------

class TestStatusEscalation:

    def test_warn_after_consecutive_breaches(self):
        t = LatencyTracker("x", budget_ms=10.0,
                           warn_after=3, error_after=10)
        for _ in range(3):
            t.observe(20.0)
        assert t.report().status == HealthStatus.WARN

    def test_error_after_many_consecutive_breaches(self):
        t = LatencyTracker("x", budget_ms=10.0,
                           warn_after=3, error_after=5)
        for _ in range(5):
            t.observe(20.0)
        assert t.report().status == HealthStatus.ERROR

    def test_single_under_budget_clears_warn(self):
        t = LatencyTracker("x", budget_ms=10.0,
                           warn_after=2, error_after=10)
        for _ in range(2):
            t.observe(20.0)
        assert t.report().status == HealthStatus.WARN
        t.observe(5.0)
        assert t.report().status == HealthStatus.OK


# ---------------------------------------------------------------------------
# Stale detection
# ---------------------------------------------------------------------------

class TestStaleDetection:

    def test_stale_after_silence(self, monkeypatch):
        t = LatencyTracker("x", budget_ms=10.0, stale_after_s=0.5)
        t.observe(5.0)
        assert t.report().status == HealthStatus.OK

        # Fast-forward time by mocking monotonic. The report path
        # reads time.monotonic() inside health_monitor, so patch there.
        from perception import health_monitor as hm
        fake_now = time.monotonic() + 2.0
        monkeypatch.setattr(hm.time, "monotonic", lambda: fake_now)
        assert t.report().status == HealthStatus.STALE


# ---------------------------------------------------------------------------
# HealthMonitor aggregation
# ---------------------------------------------------------------------------

class TestHealthMonitor:

    def test_register_and_observe(self):
        m = HealthMonitor()
        m.register("detector", budget_ms=8.0)
        m.register("depth",    budget_ms=15.0)
        assert sorted(m.stages) == ["depth", "detector"]

        with m.stage("detector"):
            pass
        m.get("depth").observe(5.0)

        reports = list(m.reports())
        names = {r.name for r in reports}
        assert names == {"detector", "depth"}

    def test_overall_status_is_worst(self):
        m = HealthMonitor()
        d = m.register("detector", budget_ms=10.0,
                       warn_after=1, error_after=10)
        t = m.register("tracker",  budget_ms=10.0)
        # detector breaches → WARN; tracker fine → OK; overall = WARN
        d.observe(20.0)
        t.observe(5.0)
        assert m.overall_status() == HealthStatus.WARN

    def test_unknown_stage_raises(self):
        m = HealthMonitor()
        m.register("detector", budget_ms=10.0)
        with pytest.raises(KeyError, match="ghost"):
            with m.stage("ghost"):
                pass

    def test_empty_monitor_overall_is_stale(self):
        m = HealthMonitor()
        assert m.overall_status() == HealthStatus.STALE

    def test_summary_line_contains_each_stage(self):
        m = HealthMonitor()
        m.register("detector", budget_ms=10.0)
        m.register("tracker",  budget_ms=2.0)
        m.get("detector").observe(5.0)
        m.get("tracker").observe(1.0)
        s = m.summary_line()
        assert "detector" in s and "tracker" in s
