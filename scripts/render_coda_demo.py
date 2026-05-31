"""
Render the README hero demo: side-by-side camera view + 3D pose comparison
on a CODa sidewalk sequence.

Left half  — the rectified cam0 frame the Husky robot actually sees.
Right half — top-down map. Sim(3)-aligned DPV-SLAM estimate (red) vs the
             LeGO-LOAM ground-truth trajectory (green); current positions
             marked.

The two passes:

  1. Walk N frames, run DPV-SLAM per frame, collect
     (camera_image, est_position, gt_position).
  2. Umeyama-align the estimated positions onto GT (same protocol as
     scripts/eval_dataset.py), render side-by-side composites, encode
     to mp4 + gif via ffmpeg.

Reproduces media/demo.{gif,mp4}. Heavy — DPV-SLAM front-end on every
frame. Run from the repo root.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from perception.coda_dataset_camera import CODaDatasetCamera
from perception.config_loader import load_config
from perception.pose_estimator_factory import build_pose_estimator
from scripts.eval_dataset import umeyama_alignment


def _build_config(sequence_dir: str, vio_enabled: bool) -> tuple:
    with open("config/default.yaml") as f:
        raw = yaml.safe_load(f)
    raw["pose_estimator"]["type"] = "dpv_slam"
    raw["pose_estimator"]["checkpoint"] = "third_party/DPVO/models/dpvo.pth"
    raw["pose_estimator"]["stride"] = 1
    raw["coda_dataset"]["sequence_dir"] = sequence_dir
    raw["visualization"]["rerun_enabled"] = False
    raw["vio"]["enabled"] = vio_enabled
    if vio_enabled:
        raw["imu"]["type"] = "coda"
    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
    yaml.safe_dump(raw, tmp)
    tmp.close()
    cfg = load_config(tmp.name)
    return cfg, raw


def _render_frame(
    img: np.ndarray,
    est_xy: np.ndarray,
    gt_xy: np.ndarray,
    idx: int,
    seq_label: str,
    ate_m: float,
    out_w: int = 640,
    out_h: int = 720,
) -> np.ndarray:
    # Left: camera image, resized into 640×720, letterboxed to preserve aspect.
    h, w = img.shape[:2]
    aspect = w / h
    target_aspect = out_w / out_h
    if aspect > target_aspect:
        new_w = out_w
        new_h = int(out_w / aspect)
    else:
        new_h = out_h
        new_w = int(out_h * aspect)
    resized = cv2.resize(img, (new_w, new_h))
    left = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y_off = (out_h - new_h) // 2
    x_off = (out_w - new_w) // 2
    left[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    # Right: matplotlib top-down trajectory.
    fig, ax = plt.subplots(figsize=(out_w / 100, out_h / 100), dpi=100)
    ax.plot(gt_xy[: idx + 1, 0], gt_xy[: idx + 1, 1],
            color="#2ca02c", linewidth=2.0, label="Ground truth (LeGO-LOAM)")
    ax.plot(est_xy[: idx + 1, 0], est_xy[: idx + 1, 1],
            color="#d62728", linewidth=2.0, label="DPV-SLAM estimate")
    ax.scatter(gt_xy[idx, 0], gt_xy[idx, 1],
               c="#2ca02c", s=120, edgecolor="black", linewidth=1.5, zorder=5)
    ax.scatter(est_xy[idx, 0], est_xy[idx, 1],
               c="#d62728", s=120, edgecolor="black", linewidth=1.5, zorder=5)
    all_xy = np.vstack([gt_xy, est_xy])
    pad = max(2.0, 0.05 * (all_xy.max() - all_xy.min()))
    ax.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ax.set_ylim(all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{seq_label}  ·  ATE {ate_m:.2f} m  (Sim(3))")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    right = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
    right = cv2.resize(right, (out_w, out_h))

    return np.hstack([left, right])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", default="data/coda/seq5")
    p.add_argument("--seq-label", default="CODa seq5  ·  Husky sidewalk")
    p.add_argument("--warmup-frames", type=int, default=200,
                   help="frames consumed before recording — DPV-SLAM init")
    p.add_argument("--num-frames", type=int, default=300,
                   help="frames to render after warmup")
    p.add_argument("--vio", action="store_true", help="enable VIO fuser")
    p.add_argument("--out-dir", default="/tmp/coda_demo_frames")
    p.add_argument("--mp4-out", default="media/demo.mp4")
    p.add_argument("--gif-out", default="media/demo.gif")
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args(argv)

    cfg, raw = _build_config(args.sequence, vio_enabled=args.vio)
    total_needed = args.warmup_frames + args.num_frames
    camera = CODaDatasetCamera(raw, args.sequence, max_frames=total_needed)
    camera.open()
    pose = build_pose_estimator(cfg, raw)

    images: List[np.ndarray] = []
    est_pos: List[np.ndarray] = []
    gt_pos: List[np.ndarray] = []

    print(f"Walking {total_needed} frames of {args.sequence} "
          f"(warmup={args.warmup_frames}, record={args.num_frames})...")
    for i in range(total_needed):
        frame = camera.get_frame()
        if frame is None:
            break
        est = pose.estimate(frame)
        gt = camera.pose_gt()
        if i < args.warmup_frames:
            if i % 50 == 0:
                print(f"  warmup {i}/{args.warmup_frames}")
            continue
        if est is None or gt is None:
            continue
        images.append(frame.image)
        est_pos.append(np.asarray(est.t, dtype=np.float64))
        gt_pos.append(np.asarray(gt[:3, 3], dtype=np.float64))
        if (i - args.warmup_frames) % 50 == 0:
            print(f"  recording {i - args.warmup_frames}/{args.num_frames}")

    if len(images) < 10:
        print(f"Too few usable frames ({len(images)}). Aborting.")
        return 1

    est_arr = np.asarray(est_pos)
    gt_arr = np.asarray(gt_pos)
    s, R, t = umeyama_alignment(est_arr, gt_arr, with_scale=True)
    est_aligned = (s * (est_arr @ R.T)) + t
    ate_m = float(np.sqrt(np.mean(np.sum((est_aligned - gt_arr) ** 2, axis=1))))
    print(f"Sim(3) alignment: scale={s:.4f}  ATE RMSE={ate_m:.3f} m  "
          f"({len(images)} frames)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    est_2d = est_aligned[:, :2]
    gt_2d = gt_arr[:, :2]
    for i in range(len(images)):
        comp = _render_frame(images[i], est_2d, gt_2d, i,
                             args.seq_label, ate_m)
        cv2.imwrite(str(out_dir / f"{i:05d}.png"), comp)
        if i % 50 == 0:
            print(f"  rendered {i}/{len(images)}")

    print("Encoding mp4 + gif via ffmpeg ...")
    Path(args.mp4_out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(args.fps),
        "-i", str(out_dir / "%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-preset", "slow", "-crf", "28",
        args.mp4_out,
    ], check=True)
    palette = "/tmp/coda_demo_palette.png"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", args.mp4_out,
        "-vf", "fps=12,scale=720:-1:flags=lanczos,palettegen=max_colors=96",
        palette,
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", args.mp4_out, "-i", palette,
        "-filter_complex",
        "fps=12,scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
        args.gif_out,
    ], check=True)
    print(f"Wrote {args.mp4_out} and {args.gif_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
