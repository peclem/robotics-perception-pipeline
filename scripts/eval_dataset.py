"""
Ground-truth accuracy evaluation for the perception pipeline.

Replays a dataset sequence and scores the unvalidated Phase 1 claims
against its ground truth:

  - monocular depth   : Depth Anything V2 dense map vs the depth sensor
                        -> RMSE, AbsRel, delta<1.25  (per-frame, averaged)
  - ego-pose / SLAM   : the PoseEstimator trajectory vs the GT pose
                        -> ATE (Sim(3)-aligned) + RPE (translation/rotation)

Datasets (--dataset):
  tum   indoor handheld RGB-D (TUMDatasetCamera) — depth + pose GT
  coda  outdoor/sidewalk ground robot (CODaDatasetCamera) — pose GT
        only; depth is skipped (CODa has no dense depth ground truth)

The metric functions (umeyama_alignment, ate, rpe, depth_metrics) are
pure and unit-tested in tests/test_eval_dataset.py; this module only
wires them to a live sequence.

Usage
-----
python3 scripts/eval_dataset.py --dataset tum  --sequence data/rgbd_dataset_freiburg1_room
python3 scripts/eval_dataset.py --dataset coda --sequence data/coda/seq0 --out data/eval
python3 scripts/eval_dataset.py --dataset tum  --sequence data/... --no-pose
python3 scripts/eval_dataset.py --dataset tum  --sequence data/... --depth-align median

Monocular SLAM has no metric scale, so ATE/RPE are reported after a
Sim(3) alignment (Horn / Umeyama) — the standard protocol. A pure
SE(3) alignment (no scale) is also printed for reference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from perception.config_loader import load_config
from perception.tum_dataset_camera import TUMDatasetCamera
from perception.coda_dataset_camera import CODaDatasetCamera


# ---------------------------------------------------------------------------
# Trajectory alignment
# ---------------------------------------------------------------------------

def umeyama_alignment(
    src: np.ndarray,
    dst: np.ndarray,
    with_scale: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Least-squares similarity transform mapping `src` onto `dst`.

    Solves for (s, R, t) minimising || dst - (s * R @ src + t) ||^2 via
    Umeyama (1991) — the closed-form SVD solution Horn's method also
    yields. With `with_scale=False` the scale is fixed at 1 (SE(3) /
    rigid alignment, for already-metric trajectories).

    Parameters
    ----------
    src, dst   : (N, 3) corresponding point sets (N >= 3 for a 3D fit)
    with_scale : solve for scale (Sim(3)) when True, else rigid SE(3)

    Returns
    -------
    (s, R, t) : scalar scale, (3, 3) rotation, (3,) translation
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/dst must be matching (N,3); got {src.shape}, {dst.shape}")
    n = src.shape[0]
    if n == 0:
        raise ValueError("cannot align empty trajectories")

    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d

    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)

    # Reflection guard — keep R a proper rotation (det = +1).
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_s = (sc ** 2).sum() / n
        s = float((D * np.diag(S)).sum() / var_s) if var_s > 1e-12 else 1.0
    else:
        s = 1.0

    t = mu_d - s * R @ mu_s
    return s, R, t


# ---------------------------------------------------------------------------
# Absolute Trajectory Error
# ---------------------------------------------------------------------------

def ate(
    est_pos: np.ndarray,
    gt_pos: np.ndarray,
    with_scale: bool = True,
) -> Dict[str, float]:
    """
    Absolute Trajectory Error between two corresponding position sets.

    The estimated trajectory is first aligned onto the ground truth
    with `umeyama_alignment`, then the per-frame Euclidean residuals
    are summarised.

    Parameters
    ----------
    est_pos, gt_pos : (N, 3) camera positions in their own world frames
    with_scale      : Sim(3) alignment when True (monocular), else SE(3)

    Returns
    -------
    dict with rmse, mean, median, std, min, max (metres), scale, num
    """
    est_pos = np.asarray(est_pos, dtype=np.float64)
    gt_pos = np.asarray(gt_pos, dtype=np.float64)
    s, R, t = umeyama_alignment(est_pos, gt_pos, with_scale=with_scale)

    aligned = (s * (R @ est_pos.T)).T + t
    err = np.linalg.norm(aligned - gt_pos, axis=1)
    return {
        "rmse":   float(np.sqrt((err ** 2).mean())),
        "mean":   float(err.mean()),
        "median": float(np.median(err)),
        "std":    float(err.std()),
        "min":    float(err.min()),
        "max":    float(err.max()),
        "scale":  float(s),
        "num":    int(err.size),
    }


# ---------------------------------------------------------------------------
# Relative Pose Error
# ---------------------------------------------------------------------------

def rpe(
    est_poses: List[np.ndarray],
    gt_poses: List[np.ndarray],
    delta: int = 1,
    scale: float = 1.0,
) -> Dict[str, float]:
    """
    Relative Pose Error over a fixed frame gap `delta`.

    For each pair (i, i+delta) the relative motion of the estimate and
    of the ground truth are compared; RPE is the RMSE of the residual
    transform's translation and rotation. Unlike ATE, RPE needs no
    global alignment — but a monocular estimate's relative translations
    are still off by the trajectory scale, so `scale` (e.g. the ATE
    Sim(3) scale) is applied to the estimated translations first.

    Parameters
    ----------
    est_poses, gt_poses : equal-length lists of (4, 4) world-from-camera
    delta               : frame gap for the relative comparison
    scale               : multiplier applied to estimated translations

    Returns
    -------
    dict with trans_rmse/trans_mean (m), rot_rmse/rot_mean (deg), num
    """
    if len(est_poses) != len(gt_poses):
        raise ValueError("est_poses and gt_poses must have equal length")
    if delta < 1:
        raise ValueError("delta must be >= 1")

    est = [np.array(p, dtype=np.float64) for p in est_poses]
    gt = [np.array(p, dtype=np.float64) for p in gt_poses]
    if scale != 1.0:
        est = [_scaled(p, scale) for p in est]

    trans_err: List[float] = []
    rot_err: List[float] = []
    for i in range(len(est) - delta):
        q = np.linalg.inv(gt[i]) @ gt[i + delta]
        p = np.linalg.inv(est[i]) @ est[i + delta]
        e = np.linalg.inv(q) @ p
        trans_err.append(float(np.linalg.norm(e[:3, 3])))
        cos = (np.trace(e[:3, :3]) - 1.0) / 2.0
        rot_err.append(float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))

    if not trans_err:
        raise ValueError(f"delta={delta} too large for {len(est)} poses")

    te = np.asarray(trans_err)
    re = np.asarray(rot_err)
    return {
        "trans_rmse": float(np.sqrt((te ** 2).mean())),
        "trans_mean": float(te.mean()),
        "rot_rmse":   float(np.sqrt((re ** 2).mean())),
        "rot_mean":   float(re.mean()),
        "num":        int(te.size),
    }


def _scaled(pose: np.ndarray, s: float) -> np.ndarray:
    """Copy of a 4x4 pose with its translation multiplied by `s`."""
    out = pose.copy()
    out[:3, 3] *= s
    return out


# ---------------------------------------------------------------------------
# Depth metrics
# ---------------------------------------------------------------------------

def depth_metrics(
    est: np.ndarray,
    gt: np.ndarray,
    align: str = "none",
    max_depth: float = 10.0,
) -> Optional[Dict[str, float]]:
    """
    Per-frame monocular depth accuracy against a ground-truth depth map.

    Only pixels valid in both maps are scored — `gt` carries NaN where
    the depth sensor had no return (TUMDatasetCamera convention), and
    pixels beyond `max_depth` or non-positive in either map are dropped.

    Parameters
    ----------
    est, gt   : (H, W) depth maps in metres; `est` is resized to `gt`
    align     : 'none'   — score the metric prediction as-is
                'median' — rescale est by median(gt)/median(est) first,
                           for relative-depth backends without metric scale
    max_depth : ignore ground-truth depth beyond this (metres)

    Returns
    -------
    dict with rmse, abs_rel, delta1 (metres / ratio / fraction) and the
    valid-pixel count, or None when the maps share no valid pixels
    """
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if est.shape != gt.shape:
        est = cv2.resize(est, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

    mask = (
        np.isfinite(gt) & (gt > 0) & (gt <= max_depth)
        & np.isfinite(est) & (est > 0)
    )
    if not mask.any():
        return None

    e = est[mask]
    g = gt[mask]
    if align == "median":
        e = e * (np.median(g) / max(np.median(e), 1e-9))

    ratio = np.maximum(e / g, g / e)
    return {
        "rmse":    float(np.sqrt(((e - g) ** 2).mean())),
        "abs_rel": float((np.abs(e - g) / g).mean()),
        "delta1":  float((ratio < 1.25).mean()),
        "num":     int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Pose conversion
# ---------------------------------------------------------------------------

def pose_to_matrix(pose) -> np.ndarray:
    """4x4 world-from-camera matrix from a CameraPose (R, t)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = pose.R
    T[:3, 3] = pose.t
    return T


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def run_evaluation(
    camera: TUMDatasetCamera,
    depth_estimator=None,
    pose_estimator=None,
    depth_align: str = "none",
    progress_every: int = 100,
) -> Dict[str, object]:
    """
    Replay `camera` frame by frame, collecting depth and pose results.

    Either estimator may be None — the corresponding metrics are then
    skipped. Returns the raw accumulators; scoring happens in `score`.
    """
    depth_rows: List[Dict[str, float]] = []
    est_poses: List[np.ndarray] = []
    gt_poses: List[np.ndarray] = []
    n_frames = 0
    t_start = time.monotonic()

    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        n_frames += 1

        if depth_estimator is not None:
            dmap = depth_estimator.dense_depth_map(frame)
            gt_depth = camera.depth_gt()
            if dmap is not None and gt_depth is not None:
                row = depth_metrics(dmap, gt_depth, align=depth_align)
                if row is not None:
                    depth_rows.append(row)

        if pose_estimator is not None:
            pose = pose_estimator.estimate(frame)
            gt_pose = camera.pose_gt()
            if pose is not None and gt_pose is not None:
                est_poses.append(pose_to_matrix(pose))
                gt_poses.append(np.asarray(gt_pose, dtype=np.float64))

        if progress_every and n_frames % progress_every == 0:
            print(f"  ... {n_frames}/{camera.total_frames} frames", flush=True)

    return {
        "depth_rows": depth_rows,
        "est_poses":  est_poses,
        "gt_poses":   gt_poses,
        "n_frames":   n_frames,
        "wall_s":     time.monotonic() - t_start,
    }


def score(acc: Dict[str, object], rpe_delta: int = 1) -> Dict[str, object]:
    """Turn the raw accumulators from `run_evaluation` into a metric report."""
    report: Dict[str, object] = {
        "n_frames": acc["n_frames"],
        "wall_s":   round(float(acc["wall_s"]), 2),
    }

    depth_rows: List[Dict[str, float]] = acc["depth_rows"]  # type: ignore[assignment]
    if depth_rows:
        report["depth"] = {
            "frames":  len(depth_rows),
            "rmse":    float(np.mean([r["rmse"] for r in depth_rows])),
            "abs_rel": float(np.mean([r["abs_rel"] for r in depth_rows])),
            "delta1":  float(np.mean([r["delta1"] for r in depth_rows])),
        }

    est_poses: List[np.ndarray] = acc["est_poses"]    # type: ignore[assignment]
    gt_poses: List[np.ndarray] = acc["gt_poses"]      # type: ignore[assignment]
    if len(est_poses) >= 3:
        est_pos = np.array([p[:3, 3] for p in est_poses])
        gt_pos = np.array([p[:3, 3] for p in gt_poses])
        ate_sim3 = ate(est_pos, gt_pos, with_scale=True)
        ate_se3 = ate(est_pos, gt_pos, with_scale=False)
        report["trajectory"] = {
            "frames":     len(est_poses),
            "ate_sim3":   ate_sim3,
            "ate_se3":    ate_se3,
            "rpe":        rpe(est_poses, gt_poses, delta=rpe_delta,
                              scale=ate_sim3["scale"]),
            "rpe_delta":  rpe_delta,
        }

    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(report: Dict[str, object], sequence: str,
                  dataset: str = "perception") -> str:
    """Human-readable summary block for stdout."""
    lines = [
        "=" * 64,
        f"{dataset} evaluation — {sequence}",
        "=" * 64,
        f"frames replayed : {report['n_frames']}  ({report['wall_s']}s wall)",
    ]

    depth = report.get("depth")
    if depth:
        lines += [
            "",
            f"DEPTH  ({depth['frames']} scored frames)",
            f"  RMSE      : {depth['rmse']:.4f} m",
            f"  AbsRel    : {depth['abs_rel']:.4f}",
            f"  delta<1.25: {depth['delta1']:.4f}",
        ]
    else:
        lines += ["", "DEPTH  : skipped (no estimator or no valid frames)"]

    traj = report.get("trajectory")
    if traj:
        a3 = traj["ate_sim3"]
        ar = traj["ate_se3"]
        r = traj["rpe"]
        lines += [
            "",
            f"TRAJECTORY  ({traj['frames']} associated poses)",
            f"  ATE RMSE  : {a3['rmse']:.4f} m   (Sim(3), scale {a3['scale']:.4f})",
            f"  ATE RMSE  : {ar['rmse']:.4f} m   (SE(3), no scale)",
            f"  RPE trans : {r['trans_rmse']:.4f} m   (delta={traj['rpe_delta']})",
            f"  RPE rot   : {r['rot_rmse']:.4f} deg",
        ]
    else:
        lines += ["", "TRAJECTORY : skipped (no estimator or <3 associated poses)"]

    lines.append("=" * 64)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="dataset accuracy evaluation")
    p.add_argument("--dataset", choices=["tum", "coda"], default="tum",
                   help="ground-truth dataset format")
    p.add_argument("--sequence", required=True,
                   help="extracted dataset sequence directory")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--rpe-delta", type=int, default=1,
                   help="frame gap for Relative Pose Error")
    p.add_argument("--depth-align", choices=["none", "median"], default="none",
                   help="'median' rescales a relative-depth backend before scoring")
    p.add_argument("--no-depth", action="store_true", help="skip depth evaluation")
    p.add_argument("--no-pose", action="store_true", help="skip trajectory evaluation")
    p.add_argument("--out", default=None,
                   help="directory to write eval_<dataset>_<sequence>.json")
    return p.parse_args(argv)


def _open_camera(dataset: str, raw: dict, seq_dir: Path, max_frames):
    """Open the dataset-specific CameraInterface backend."""
    if dataset == "tum":
        camera = TUMDatasetCamera(raw, seq_dir, max_frames=max_frames)
    elif dataset == "coda":
        camera = CODaDatasetCamera(raw, seq_dir, max_frames=max_frames)
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")
    camera.open()
    return camera


def _build_depth_estimator(cfg):
    """DepthAnythingEstimator from config, or None if it cannot load."""
    try:
        from perception.depth_estimator import DepthAnythingEstimator
        est = DepthAnythingEstimator(device=cfg.depth.device, model_name=cfg.depth.model)
        if not getattr(est, "is_ready", False):
            print("  depth: Depth Anything V2 failed to load — depth eval skipped.")
            return None
        est.warmup()
        return est
    except Exception as exc:  # noqa: BLE001 — graceful degradation by design
        print(f"  depth: estimator unavailable ({exc}) — depth eval skipped.")
        return None


def _build_pose_estimator(cfg, raw):
    """Configured PoseEstimator via the factory, or None if it cannot build."""
    try:
        from perception.pose_estimator_factory import build_pose_estimator
        return build_pose_estimator(cfg, raw)
    except Exception as exc:  # noqa: BLE001 — graceful degradation by design
        print(f"  pose: estimator unavailable ({exc}) — trajectory eval skipped.")
        return None


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    seq_dir = Path(args.sequence)
    if not seq_dir.is_dir():
        print(f"error: sequence directory not found: {seq_dir}", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    with open(args.config) as f:
        raw = yaml.safe_load(f) or {}

    print(f"Loading {args.dataset} sequence: {seq_dir}")
    camera = _open_camera(args.dataset, raw, seq_dir, args.max_frames)
    print(f"  {camera.total_frames} frames, intrinsics fx={camera.intrinsics.fx:.1f}")

    # CODa has no dense depth ground truth — force depth off for it.
    no_depth = args.no_depth or args.dataset == "coda"
    if args.dataset == "coda" and not args.no_depth:
        print("  depth: skipped — CODa has no dense depth ground truth.")
    depth_estimator = None if no_depth else _build_depth_estimator(cfg)
    pose_estimator = None if args.no_pose else _build_pose_estimator(cfg, raw)
    if depth_estimator is None and pose_estimator is None:
        print("error: neither depth nor pose estimator available — nothing to score.",
              file=sys.stderr)
        camera.release()
        return 1

    print("Replaying sequence ...")
    acc = run_evaluation(camera, depth_estimator, pose_estimator,
                         depth_align=args.depth_align)
    camera.release()

    report = score(acc, rpe_delta=args.rpe_delta)
    print()
    print(format_report(report, seq_dir.name, dataset=args.dataset))

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"eval_{args.dataset}_{seq_dir.name}.json"
        with open(out_path, "w") as f:
            json.dump({"dataset": args.dataset, "sequence": seq_dir.name, **report},
                      f, indent=2)
        print(f"\nwrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
