"""
MOT17 benchmark evaluation for the robotics perception pipeline.

Usage
-----
python3 scripts/benchmark.py --dataset data/MOT17 --split train --sequences MOT17-04-FRCNN --out data/mot17_results
python3 scripts/benchmark.py --dataset data/MOT17 --split train --out data/mot17_results
"""

from __future__ import annotations

import argparse
import configparser
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from perception.camera_interface import CameraFrame, CameraIntrinsics
from perception.config_loader import load_config
from perception.detector import YOLOv8Detector
from perception.appearance_extractor import (
    AppearanceExtractor, NullAppearanceExtractor,
)
from tracking.tracker import ByteTracker


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOT17 benchmark evaluation")
    parser.add_argument("--dataset",   type=str, default="data/MOT17")
    parser.add_argument("--split",     choices=["train", "test"], default="train")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--config",    type=str, default="config/default.yaml")
    parser.add_argument("--out",       type=str, default="data/mot17_results")
    parser.add_argument("--no-eval",   action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Seqinfo
# ---------------------------------------------------------------------------

def read_seqinfo(seq_path: Path) -> dict:
    ini_path = seq_path / "seqinfo.ini"
    if not ini_path.exists():
        raise FileNotFoundError(f"seqinfo.ini not found: {ini_path}")
    cfg = configparser.ConfigParser()
    cfg.read(str(ini_path))
    info = cfg["Sequence"]
    return {
        "name":     info["name"],
        "fps":      float(info["frameRate"]),
        "width":    int(info["imWidth"]),
        "height":   int(info["imHeight"]),
        "n_frames": int(info["seqLength"]),
        "img_dir":  info["imDir"],
        "img_ext":  info["imExt"],
    }


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

def load_mot17_gt(gt_path: Path) -> Dict[int, list]:
    """
    Load MOT17 ground truth.
    Format: frame, id, x1, y1, w, h, conf, class, visibility
    Returns dict: frame_id -> list of (track_id, x1, y1, x2, y2)
    Only pedestrians (class=1) with conf=1.
    """
    gt: Dict[int, list] = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x1       = float(parts[2])
            y1       = float(parts[3])
            w        = float(parts[4])
            h        = float(parts[5])
            conf     = int(parts[6])
            cls      = int(parts[7]) if len(parts) > 7 else 1
            if conf == 0 or cls != 1:
                continue
            gt.setdefault(frame_id, []).append(
                (track_id, x1, y1, x1 + w, y1 + h)
            )
    return gt


# ---------------------------------------------------------------------------
# MOTA accumulator
# ---------------------------------------------------------------------------

def compute_iou_matrix(gt_boxes: np.ndarray, tr_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(tr_boxes) == 0:
        return np.zeros((len(gt_boxes), len(tr_boxes)))
    a = gt_boxes[:, None, :]
    b = tr_boxes[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    area_b = (tr_boxes[:, 2] - tr_boxes[:, 0]) * (tr_boxes[:, 3] - tr_boxes[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-10)


class MOTAAccumulator:
    """MOTA = 1 - (FN + FP + IDSW) / GT"""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.n_gt = self.n_fp = self.n_fn = self.n_idsw = 0
        self.n_match = 0
        self.sum_iou = 0.0
        self._gt_to_tracker: dict = {}

    def update(self, gt_ids, gt_boxes, tr_ids, tr_boxes):
        self.n_gt += len(gt_ids)
        if not gt_ids and not tr_ids:
            return
        if not gt_ids:
            self.n_fp += len(tr_ids)
            return
        if not tr_ids:
            self.n_fn += len(gt_ids)
            return

        iou_mat    = compute_iou_matrix(gt_boxes, tr_boxes)
        matched_gt = set()
        matched_tr = set()

        for idx in np.argsort(iou_mat.ravel())[::-1]:
            gi, ti = divmod(int(idx), len(tr_ids))
            if gi in matched_gt or ti in matched_tr:
                continue
            if iou_mat[gi, ti] < self.iou_threshold:
                break
            matched_gt.add(gi)
            matched_tr.add(ti)
            self.n_match += 1
            self.sum_iou += iou_mat[gi, ti]
            gt_id, tr_id = gt_ids[gi], tr_ids[ti]
            if gt_id in self._gt_to_tracker and self._gt_to_tracker[gt_id] != tr_id:
                self.n_idsw += 1
            self._gt_to_tracker[gt_id] = tr_id

        self.n_fn += len(gt_ids) - len(matched_gt)
        self.n_fp += len(tr_ids) - len(matched_tr)

    @property
    def mota(self):
        return 0.0 if self.n_gt == 0 else \
            1.0 - (self.n_fn + self.n_fp + self.n_idsw) / self.n_gt

    @property
    def motp(self):
        return 0.0 if self.n_match == 0 else self.sum_iou / self.n_match

    def summary(self) -> dict:
        return {
            "MOTA": round(self.mota * 100, 2),
            "MOTP": round(self.motp * 100, 2),
            "FP":   self.n_fp,
            "FN":   self.n_fn,
            "IDSW": self.n_idsw,
            "GT":   self.n_gt,
        }


# ---------------------------------------------------------------------------
# Frame wrapper
# ---------------------------------------------------------------------------

def make_frame_from_image(
    image: np.ndarray, frame_idx: int, timestamp: float,
    width: int, height: int,
) -> CameraFrame:
    intr = CameraIntrinsics(
        fx=width, fy=height, cx=width / 2.0, cy=height / 2.0,
        width=width, height=height,
    )
    return CameraFrame(
        image=image, timestamp=timestamp, frame_idx=frame_idx,
        intrinsics=intr, source_id="mot17",
    )


# ---------------------------------------------------------------------------
# Sequence runner
# ---------------------------------------------------------------------------

def _build_appearance_extractor(cfg: dict) -> AppearanceExtractor:
    """
    Build an AppearanceExtractor matching the benchmark config.
    Returns a NullAppearanceExtractor when `appearance.type=='null'`
    (the baseline path that produced the existing MOT17 numbers).
    """
    ap = cfg.get("appearance", {}) or {}
    t = ap.get("type", "null")
    if t == "null":
        return NullAppearanceExtractor()
    if t == "dinov2":
        from perception.appearance_extractor import DINOv2AppearanceExtractor
        return DINOv2AppearanceExtractor(
            model_name=ap.get("model", "facebook/dinov2-small"),
            device=ap.get("device", "cuda"),
        )
    raise ValueError(
        f"benchmark: unknown appearance.type={t!r}. Supported: 'null', 'dinov2'."
    )


def run_sequence(
    seq_path: Path,
    detector: YOLOv8Detector,
    config:   dict,
    out_path: Path,
) -> dict:
    info    = read_seqinfo(seq_path)
    img_dir = seq_path / info["img_dir"]
    gt_path = seq_path / "gt" / "gt.txt"
    frames  = sorted(img_dir.glob(f"*{info['img_ext']}"))

    if not frames:
        raise RuntimeError(f"No images found in {img_dir}")

    print(f"\n  Sequence : {info['name']}")
    print(f"  Frames   : {len(frames)}  FPS: {info['fps']:.1f}")
    print(f"  Size     : {info['width']}x{info['height']}")

    tracker         = ByteTracker(config)
    appearance      = _build_appearance_extractor(config)
    use_appearance  = (config.get("tracker", {}).get("use_appearance", False)
                       and not isinstance(appearance, NullAppearanceExtractor))
    if use_appearance:
        print(f"  Appearance: {type(appearance).__name__}")
    accumulator = MOTAAccumulator(iou_threshold=0.5)
    gt          = load_mot17_gt(gt_path) if gt_path.exists() else {}
    dt          = 1.0 / info["fps"]
    results     = []
    t_total     = 0.0
    timestamp   = 0.0

    for i, img_path in enumerate(frames):
        frame_id   = i + 1
        timestamp += dt

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        frame = make_frame_from_image(
            image, frame_idx=i, timestamp=timestamp,
            width=info["width"], height=info["height"],
        )

        t0               = time.monotonic()
        detections       = detector.detect(frame)
        det_embeddings = None
        if use_appearance and detections:
            bboxes = [d.bbox_xyxy for d in detections]
            det_embeddings = appearance.extract(frame.image, bboxes)
        confirmed_tracks = tracker.update(
            detections, frame, detection_embeddings=det_embeddings,
        )
        t_total         += time.monotonic() - t0

        for track in confirmed_tracks:
            x1, y1, x2, y2 = track.bbox_xyxy
            results.append((frame_id, track.track_id,
                            x1, y1, x2 - x1, y2 - y1, track.score))

        gt_frame = gt.get(frame_id, [])
        gt_ids   = [g[0] for g in gt_frame]
        gt_boxes = np.array([[g[1], g[2], g[3], g[4]] for g in gt_frame],
                             dtype=np.float32) if gt_frame else np.zeros((0, 4))
        tr_ids   = [t.track_id for t in confirmed_tracks]
        tr_boxes = np.array([t.bbox_xyxy for t in confirmed_tracks],
                             dtype=np.float32) if confirmed_tracks else np.zeros((0, 4))
        accumulator.update(gt_ids, gt_boxes, tr_ids, tr_boxes)

        if i % 100 == 0:
            hz = (i + 1) / t_total if t_total > 0 else 0.0
            m  = accumulator.summary()
            print(f"    Frame {frame_id:4d}/{len(frames)} | "
                  f"dets={len(detections):2d} | "
                  f"tracks={len(confirmed_tracks):2d} | "
                  f"MOTA={m['MOTA']:.1f}% | {hz:.1f} Hz")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for frame_id, tid, x1, y1, w, h, conf in results:
            f.write(f"{frame_id},{tid},{x1:.2f},{y1:.2f},"
                    f"{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n")

    hz      = len(frames) / t_total if t_total > 0 else 0.0
    metrics = accumulator.summary()
    metrics.update({
        "seq_name": info["name"],
        "n_frames": len(frames),
        "hz":       round(hz, 1),
        "n_tracks": max((r[1] for r in results), default=0),
    })
    print(f"  Done | MOTA={metrics['MOTA']:.1f}% "
          f"MOTP={metrics['MOTP']:.1f}% "
          f"FP={metrics['FP']} FN={metrics['FN']} "
          f"IDSW={metrics['IDSW']} | {hz:.1f} Hz")
    return metrics


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------

def write_results_markdown(
    metrics:    dict | None,
    seq_stats:  list,
    out_path:   Path,
    config_name: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Benchmark Results — MOT17",
        "",
        f"Generated: {timestamp}  ",
        f"Config: `{config_name}`",
        "",
    ]

    # "—" is the en-dash marker for missing data — used wherever the
    # caller supplied a partial metrics dict (a non-MOT17 benchmark,
    # an ablation script that doesn't compute IDSW, a test fixture).
    # Canonical MOT metrics are rendered as percentages; raw counts as
    # integers; everything else as a two-decimal float.
    _PCT_KEYS = {"MOTA", "MOTP", "IDF1", "HOTA", "DetA", "AssA", "LocA"}
    _INT_KEYS = {"FP", "FN", "IDSW"}

    def _fmt_value(key, v):
        if v is None:
            return "—"
        if key in _PCT_KEYS and isinstance(v, (int, float)):
            return f"{v:.1f}%"
        if key in _INT_KEYS:
            return str(v)
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    if metrics:
        lines += [
            "## Overall Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        # Preferred ordering for the standard MOT metrics; everything
        # else preserves insertion order (Python 3.7+ dict).
        preferred = ["MOTA", "MOTP", "IDF1", "HOTA",
                     "FP", "FN", "IDSW"]
        keys = [k for k in preferred if k in metrics] + \
               [k for k in metrics if k not in preferred]
        for key in keys:
            lines.append(f"| {key:<6} | {_fmt_value(key, metrics[key])} |")
        lines.append("")

    lines += [
        "## Per-Sequence Results",
        "",
        "| Sequence | Frames | MOTA | MOTP | FP | FN | IDSW | Hz |",
        "|----------|--------|------|------|----|----|------|----|",
    ]
    for s in sorted(seq_stats, key=lambda x: x["seq_name"]):
        lines.append(
            f"| {s['seq_name']} | {s['n_frames']} | "
            f"{_fmt_value('MOTA', s.get('MOTA'))} | "
            f"{_fmt_value('MOTP', s.get('MOTP'))} | "
            f"{_fmt_value('FP',   s.get('FP'))} | "
            f"{_fmt_value('FN',   s.get('FN'))} | "
            f"{_fmt_value('IDSW', s.get('IDSW'))} | "
            f"{s['hz']} |"
        )

    lines += [
        "",
        "## System",
        "",
        "- Detector: YOLOv8n (fp16, cuda:0)",
        "- Tracker: ByteTrack two-stage IoU association",
        "- State estimation: Kalman Filter (Joseph form, NIS-validated)",
        "- Dataset: MOT17-train",
        "- IoU threshold: 0.5",
        "",
    ]

    out_path.write_text("\n".join(lines))
    print(f"\nResults written to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    split_path   = dataset_path / args.split

    if not split_path.exists():
        print(f"Error: dataset split not found: {split_path}")
        sys.exit(1)

    all_seqs = sorted([
        d for d in split_path.iterdir()
        if d.is_dir() and (d / "img1").exists()
    ])

    if args.sequences:
        seqs = [split_path / s for s in args.sequences]
    else:
        seqs = all_seqs

    cfg     = load_config(args.config)
    raw_cfg = cfg.as_dict()

    print(f"\nMOT17 Benchmark")
    print(f"  Dataset  : {split_path}")
    print(f"  Sequences: {len(seqs)}")
    print(f"  Device   : {cfg.detector.device}")
    print(f"  Model    : {cfg.detector.model}")

    print("\nLoading detector ...")
    detector = YOLOv8Detector(raw_cfg)
    detector.warmup()
    print(f"Detector ready. Latency: {detector.mean_inference_ms:.1f} ms")

    out_dir   = Path(args.out)
    seq_stats = []

    for seq_path in seqs:
        out_file = out_dir / f"{seq_path.name}.txt"
        try:
            stats = run_sequence(seq_path, detector, raw_cfg, out_file)
            seq_stats.append(stats)
        except Exception as e:
            print(f"  Error on {seq_path.name}: {e}")
            continue

    if not seq_stats:
        print("No sequences completed.")
        sys.exit(1)

    total_frames = sum(s["n_frames"]     for s in seq_stats)
    total_time   = sum(s["n_frames"] / s["hz"] for s in seq_stats if s["hz"] > 0)
    mean_hz      = total_frames / total_time if total_time > 0 else 0.0

    mean_mota  = float(np.mean([s["MOTA"] for s in seq_stats]))
    mean_motp  = float(np.mean([s["MOTP"] for s in seq_stats]))
    total_fp   = sum(s["FP"]   for s in seq_stats)
    total_fn   = sum(s["FN"]   for s in seq_stats)
    total_idsw = sum(s["IDSW"] for s in seq_stats)

    metrics = {
        "MOTA": round(mean_mota, 2),
        "MOTP": round(mean_motp, 2),
        "FP":   total_fp,
        "FN":   total_fn,
        "IDSW": total_idsw,
    }

    print(f"\n{'='*50}")
    print(f"Final Results ({len(seq_stats)} sequences)")
    print(f"  Mean MOTA : {mean_mota:.1f}%")
    print(f"  Mean MOTP : {mean_motp:.1f}%")
    print(f"  Total FP  : {total_fp}")
    print(f"  Total FN  : {total_fn}")
    print(f"  Total IDSW: {total_idsw}")
    print(f"  Mean Hz   : {mean_hz:.1f}")
    print(f"{'='*50}")

    write_results_markdown(
        metrics=metrics,
        seq_stats=seq_stats,
        out_path=Path("docs/benchmark_results.md"),
        config_name=args.config,
    )


if __name__ == "__main__":
    main()
