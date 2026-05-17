"""
DPVO latency benchmark on the local 4070Ti.

Phase 2 gate: decide whether DPVO fits the 30 Hz pipeline budget
alongside YOLOv8n (~5 ms) + Depth Anything V2 (~10 ms). Target
headroom for DPVO is ~17 ms median (30 Hz total budget = 33.3 ms,
minus ~16 ms for the other GPU stages).

Reports median + p95 per-frame latency after a warmup window long
enough for DPVO to bootstrap its internal state (~30 frames).
Sweeps resolution, patches-per-frame, and stride to find a workable
operating point.

Run:
    python scripts/benchmark_dpvo_latency.py
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "DPVO"))

from dpvo.config import cfg as dpvo_cfg
from dpvo.dpvo import DPVO

CHECKPOINT = ROOT / "third_party" / "DPVO" / "models" / "dpvo.pth"


def make_synthetic_frame(h: int, w: int, t: int) -> np.ndarray:
    """Deterministic moving-texture frame, gives DPVO trackable features."""
    rng = np.random.default_rng(seed=t)
    img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    band_x = (t * 8) % w
    img[:, max(0, band_x - 20):band_x + 20, :] = 255
    return img


def load_real_video_frames(path: str, h: int, w: int, n: int) -> np.ndarray:
    """Read up to n frames from a video file, resize to (w, h), return uint8 (n, h, w, 3)."""
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n:
        ok, img = cap.read()
        if not ok:
            break
        if (img.shape[1], img.shape[0]) != (w, h):
            img = cv2.resize(img, (w, h))
        frames.append(img)
    cap.release()
    if len(frames) < n:
        raise RuntimeError(f"Wanted {n} frames from {path}, got {len(frames)}")
    return np.stack(frames)


def intrinsics_for(h: int, w: int) -> np.ndarray:
    fx = fy = 0.85 * w
    cx, cy = w / 2.0, h / 2.0
    return np.array([fx, fy, cx, cy], dtype=np.float32)


def bench(
    h: int,
    w: int,
    patches: int = 96,
    stride: int = 1,
    n_frames: int = 150,
    warmup: int = 40,
    video_path: str = "",
) -> dict:
    """One configuration. `stride=2` means we feed every other frame to DPVO."""
    cfg = dpvo_cfg.clone()
    cfg.PATCHES_PER_FRAME = patches

    slam = DPVO(cfg, str(CHECKPOINT), ht=h, wd=w, viz=False)

    K_t = torch.from_numpy(intrinsics_for(h, w)).cuda()
    latencies_ms: List[float] = []

    real_frames = (
        load_real_video_frames(video_path, h, w, n_frames)
        if video_path else None
    )

    torch.cuda.reset_peak_memory_stats()

    fed_idx = 0
    for t in range(n_frames):
        if t % stride != 0:
            continue
        if real_frames is not None:
            img_np = real_frames[t]
        else:
            img_np = make_synthetic_frame(h, w, t)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).cuda()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            slam(t, img_t, K_t)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if fed_idx >= warmup // stride:
            latencies_ms.append(dt_ms)
        fed_idx += 1

    arr = np.array(latencies_ms)
    vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    median = float(np.median(arr))

    # "Effective" Hz = how often the pose updates given stride.
    # With stride=2, every other frame is skipped → pose Hz = (1/median) / stride * 1000 ? No.
    # Actually: frames arrive at 30 Hz, DPVO consumes every `stride`-th, so DPVO must finish
    # one call within `stride * (1/30)` seconds = stride * 33.3 ms.
    budget_ms = stride * 33.3

    del slam
    torch.cuda.empty_cache()
    return {
        "res":          f"{w}x{h}",
        "patches":      patches,
        "stride":       stride,
        "n":            len(arr),
        "median_ms":    median,
        "p95_ms":       float(np.percentile(arr, 95)),
        "vram_mb":      vram_mb,
        "pose_hz":      30.0 / stride,
        "per_call_budget_ms": budget_ms,
        "fits_budget":  median <= budget_ms - 16.0,  # leave 16 ms for det+depth
    }


def main() -> None:
    print(f"Device:    {torch.cuda.get_device_name(0)}")
    print(f"Torch:     {torch.__version__}")
    print(f"Checkpoint: {CHECKPOINT}\n")
    assert CHECKPOINT.exists()

    # Real-video config sweep — synthetic frames were pathological in v1.
    video = str(ROOT / "data" / "sample.mp4")
    print(f"Source video: {video}\n")

    configs = [
        # Baseline at native resolutions
        (480, 640, 96, 1),
        (720, 1280, 96, 1),
        # Patch sweep at 640x480
        (480, 640, 64, 1),
        (480, 640, 48, 1),
        # Stride=2 → pose at 15 Hz, DPVO has 66 ms budget
        (480, 640, 96, 2),
        (720, 1280, 96, 2),
    ]

    results = [bench(h, w, p, s, video_path=video) for (h, w, p, s) in configs]

    print(f"\n{'res':>9s} {'patch':>5s} {'stride':>6s}  "
          f"{'med':>6s} {'p95':>6s} {'budget':>7s} {'vram':>6s}  {'pose Hz':>7s}  verdict")
    print("-" * 80)
    for r in results:
        v = "FITS" if r["fits_budget"] else "OVER"
        print(
            f"{r['res']:>9s} {r['patches']:>5d} {r['stride']:>6d}  "
            f"{r['median_ms']:>5.1f}  {r['p95_ms']:>5.1f}  "
            f"{r['per_call_budget_ms']:>6.1f}  {r['vram_mb']:>4.0f}MB   "
            f"{r['pose_hz']:>5.1f}    {v}"
        )
    print("\nBudget = stride × 33.3 ms − 16 ms reserved for YOLOv8n + Depth Anything V2 Small.")
    print("Pose Hz = camera frame rate ÷ stride.  Most planners accept 10–15 Hz pose.")


if __name__ == "__main__":
    main()
