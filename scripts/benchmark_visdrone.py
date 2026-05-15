"""
VisDrone2019-MOT benchmark evaluation for the robotics perception pipeline.

Dataset structure expected
--------------------------
data/VisDrone2019-MOT-train/
    sequences/{seq_name}/    — JPEG frames (0000001.jpg ...)
    annotations/{seq_name}.txt — ground truth, one line per detection

Annotation format (ground truth)
---------------------------------
frame_id, track_id, x1, y1, w, h, score, category, truncation, occlusion

    score    : 0 = ignored region, 1 = valid
    category : 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van,
               6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor

VisDrone → COCO class mapping (for detector filtering)
-------------------------------------------------------
    pedestrian/people → person  (COCO 0)
    bicycle           → bicycle (COCO 1)
    car               → car     (COCO 2)
    motor             → motorcycle (COCO 3)
    bus               → bus     (COCO 5)
    van/truck         → truck   (COCO 7)

Metrics
-------
We use a lightweight MOTA/IDF1 implementation that runs without
the MOTChallenge server. Results are written to:
    docs/benchmark_results.md

Usage
-----
# Single sequence (fast test):
python3 scripts/benchmark_visdrone.py \
    --dataset data/VisDrone2019-MOT-train \
    --sequences uav0000013_00000_v

# Full train split:
python3 scripts/benchmark_visdrone.py \
    --dataset data/VisDrone2019-MOT-train

# CPU mode:
DEVICE=cpu python3 scripts/benchmark_visdrone.py \
    --dataset data/VisDrone2019-MOT-train \
    --sequences uav0000013_00000_v
"""

from __future__ import annotations

import argparse
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
from tracking.tracker import ByteTracker
from tracking.track import Track


# ---------------------------------------------------------------------------
# VisDrone constants
# ---------------------------------------------------------------------------

# VisDrone category → COCO class ID mapping
VISDRONE_TO_COCO: Dict[int, int] = {
    1:  0,   # pedestrian → person
    2:  0,   # people     → person
    3:  1,   # bicycle    → bicycle
    4:  2,   # car        → car
    5:  7,   # van        → truck
    6:  7,   # truck      → truck
    9:  5,   # bus        → bus
    10: 3,   # motor      → motorcycle
}

# COCO class IDs we evaluate (others ignored)
EVAL_COCO_CLASSES = {0, 1, 2, 3, 5, 7}

# VisDrone frame rate (fixed at 30fps for all sequences)
VISDRONE_FPS = 30.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisDrone2019-MOT benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/VisDrone2019-MOT-train",
        help="Path to VisDrone2019-MOT-train directory",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Specific sequence names to evaluate (default: all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Pipeline config YAML",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/visdrone_results",
        help="Output directory for tracker result files",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=None,
        help="Limit number of sequences (for quick testing)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

def load_gt(ann_path: Path) -> Dict[int, List[Tuple]]:
    """
    Load VisDrone ground truth annotations.

    Parameters
    ----------
    ann_path : path to annotation .txt file

    Returns
    -------
    dict mapping frame_id → list of (track_id, x1, y1, w, h, category)
    Only returns valid detections (score==1) in tracked categories.
    """
    gt: Dict[int, List] = {}

    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue

            frame_id  = int(parts[0])
            track_id  = int(parts[1])
            x1        = float(parts[2])
            y1        = float(parts[3])
            w         = float(parts[4])
            h         = float(parts[5])
            score     = int(parts[6])     # 0=ignored, 1=valid
            category  = int(parts[7])

            # Skip ignored regions and unmapped categories
            if score == 0 or category not in VISDRONE_TO_COCO:
                continue

            if frame_id not in gt:
                gt[frame_id] = []
            gt[frame_id].append((track_id, x1, y1, w, h, category))

    return gt


# ---------------------------------------------------------------------------
# MOTA / IDF1 computation
# ---------------------------------------------------------------------------

def compute_iou_matrix(
    gt_boxes:  np.ndarray,
    tr_boxes:  np.ndarray,
) -> np.ndarray:
    """
    Pairwise IoU between GT and tracker boxes.
    gt_boxes, tr_boxes: (N, 4) in x1,y1,x2,y2 format.
    Returns (N_gt, N_tr) IoU matrix.
    """
    if len(gt_boxes) == 0 or len(tr_boxes) == 0:
        return np.zeros((len(gt_boxes), len(tr_boxes)))

    a = gt_boxes[:, None, :]    # (N_gt, 1, 4)
    b = tr_boxes[None, :, :]    # (1, N_tr, 4)

    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])

    inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
    area_a = (gt_boxes[:, 2]-gt_boxes[:, 0]) * (gt_boxes[:, 3]-gt_boxes[:, 1])
    area_b = (tr_boxes[:, 2]-tr_boxes[:, 0]) * (tr_boxes[:, 3]-tr_boxes[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    union = np.maximum(union, 1e-10)
    return inter / union


class MOTAAccumulator:
    """
    Lightweight MOTA/MOTP accumulator.

    MOTA = 1 - (FN + FP + IDSW) / GT
    MOTP = sum(IoU of matched pairs) / num_matches

    Reference: Bernardin & Stiefelhagen (2008)
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.n_gt    = 0
        self.n_fp    = 0
        self.n_fn    = 0
        self.n_idsw  = 0
        self.n_match = 0
        self.sum_iou = 0.0
        # Track last known GT→tracker ID mapping for IDSW detection
        self._gt_to_tracker: Dict[int, int] = {}

    def update(
        self,
        gt_ids:   List[int],
        gt_boxes: np.ndarray,
        tr_ids:   List[int],
        tr_boxes: np.ndarray,
    ) -> None:
        """
        Update accumulator for one frame.

        Uses greedy matching at iou_threshold — good enough for
        portfolio-level evaluation. TrackEval uses the Hungarian
        algorithm for exact matching.
        """
        self.n_gt += len(gt_ids)

        if len(gt_ids) == 0 and len(tr_ids) == 0:
            return

        if len(gt_ids) == 0:
            self.n_fp += len(tr_ids)
            return

        if len(tr_ids) == 0:
            self.n_fn += len(gt_ids)
            return

        iou_mat = compute_iou_matrix(gt_boxes, tr_boxes)

        matched_gt  = set()
        matched_tr  = set()

        # Greedy matching — highest IoU first
        flat_idx = np.argsort(iou_mat.ravel())[::-1]
        for idx in flat_idx:
            gi, ti = divmod(int(idx), len(tr_ids))
            if gi in matched_gt or ti in matched_tr:
                continue
            if iou_mat[gi, ti] < self.iou_threshold:
                break
            matched_gt.add(gi)
            matched_tr.add(ti)
            self.n_match += 1
            self.sum_iou += iou_mat[gi, ti]

            # ID switch check
            gt_id = gt_ids[gi]
            tr_id = tr_ids[ti]
            if gt_id in self._gt_to_tracker:
                if self._gt_to_tracker[gt_id] != tr_id:
                    self.n_idsw += 1
            self._gt_to_tracker[gt_id] = tr_id

        self.n_fn += len(gt_ids) - len(matched_gt)
        self.n_fp += len(tr_ids) - len(matched_tr)

    @property
    def mota(self) -> float:
        if self.n_gt == 0:
            return 0.0
        return 1.0 - (self.n_fn + self.n_fp + self.n_idsw) / self.n_gt

    @property
    def motp(self) -> float:
        if self.n_match == 0:
            return 0.0
        return self.sum_iou / self.n_match

    def summary(self) -> dict:
        return {
            "MOTA":   round(self.mota * 100, 2),
            "MOTP":   round(self.motp * 100, 2),
            "GT":     self.n_gt,
            "FP":     self.n_fp,
            "FN":     self.n_fn,
            "IDSW":   self.n_idsw,
            "Matches": self.n_match,
        }


# ---------------------------------------------------------------------------
# Sequence runner
# ---------------------------------------------------------------------------

def make_frame(
    image: np.ndarray, frame_idx: int, timestamp: float
) -> CameraFrame:
    h, w = image.shape[:2]
    intr = CameraIntrinsics(
        fx=w, fy=h, cx=w/2, cy=h/2, width=w, height=h
    )
    return CameraFrame(
        image=image, timestamp=timestamp,
        frame_idx=frame_idx, intrinsics=intr, source_id="visdrone",
    )


def run_sequence(
    seq_name:  str,
    seq_dir:   Path,
    ann_path:  Path,
    detector:  YOLOv8Detector,
    config:    dict,
    out_path:  Path,
) -> dict:
    """
    Run the full pipeline on one VisDrone sequence.
    Returns per-sequence metrics and timing stats.
    """
    # Load ground truth
    gt = load_gt(ann_path) if ann_path.exists() else {}

    # Collect frames
    frames = sorted(seq_dir.glob("*.jpg"))
    if not frames:
        frames = sorted(seq_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(f"No images in {seq_dir}")

    print(f"\n  Seq: {seq_name}  ({len(frames)} frames)")

    tracker    = ByteTracker(config)
    accumulator = MOTAAccumulator(iou_threshold=0.5)
    results    = []
    t_total    = 0.0
    dt         = 1.0 / VISDRONE_FPS
    timestamp  = 0.0

    for i, img_path in enumerate(frames):
        frame_id  = i + 1   # 1-indexed
        timestamp += dt

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        frame = make_frame(image, frame_idx=i, timestamp=timestamp)

        t0 = time.monotonic()
        detections       = detector.detect(frame)
        confirmed_tracks = tracker.update(detections, frame)
        t_total += time.monotonic() - t0

        # Write tracker output
        for track in confirmed_tracks:
            x1, y1, x2, y2 = track.bbox_xyxy
            w = x2 - x1
            h = y2 - y1
            results.append((
                frame_id, track.track_id,
                x1, y1, w, h, track.score,
            ))

        # Update MOTA accumulator
        gt_frame = gt.get(frame_id, [])
        gt_ids   = [g[0] for g in gt_frame]
        gt_boxes = np.array([
            [g[1], g[2], g[1]+g[3], g[2]+g[4]] for g in gt_frame
        ], dtype=np.float32) if gt_frame else np.zeros((0, 4))

        tr_ids   = [t.track_id for t in confirmed_tracks]
        tr_boxes = np.array([
            t.bbox_xyxy for t in confirmed_tracks
        ], dtype=np.float32) if confirmed_tracks else np.zeros((0, 4))

        accumulator.update(gt_ids, gt_boxes, tr_ids, tr_boxes)

        if i % 200 == 0:
            hz = (i+1) / t_total if t_total > 0 else 0.0
            metrics = accumulator.summary()
            print(
                f"    Frame {frame_id:4d}/{len(frames)} | "
                f"dets={len(detections):2d} | "
                f"trk={len(confirmed_tracks):2d} | "
                f"MOTA={metrics['MOTA']:.1f}% | "
                f"{hz:.1f} Hz"
            )

    # Save tracker output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for frame_id, tid, x1, y1, w, h, conf in results:
            f.write(
                f"{frame_id},{tid},"
                f"{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                f"{conf:.4f},-1,-1,-1\n"
            )

    hz     = len(frames) / t_total if t_total > 0 else 0.0
    metrics = accumulator.summary()
    metrics.update({
        "seq_name": seq_name,
        "n_frames": len(frames),
        "hz":       round(hz, 1),
    })

    print(
        f"  Done | MOTA={metrics['MOTA']:.1f}% "
        f"MOTP={metrics['MOTP']:.1f}% "
        f"FP={metrics['FP']} FN={metrics['FN']} "
        f"IDSW={metrics['IDSW']} | {hz:.1f} Hz"
    )
    return metrics


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------

def write_markdown(
    seq_metrics: List[dict],
    out_path:    Path,
    config_name: str,
) -> None:
    """Write benchmark results to docs/benchmark_results.md"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Aggregate
    total_gt = sum(s["GT"] for s in seq_metrics)
    total_fp = sum(s["FP"] for s in seq_metrics)
    total_fn = sum(s["FN"] for s in seq_metrics)
    total_sw = sum(s["IDSW"] for s in seq_metrics)
    mean_mota = float(np.mean([s["MOTA"] for s in seq_metrics]))
    mean_motp = float(np.mean([s["MOTP"] for s in seq_metrics]))
    mean_hz   = float(np.mean([s["hz"]   for s in seq_metrics]))

    lines = [
        "# Benchmark Results — VisDrone2019-MOT-train",
        "",
        f"Generated: {timestamp}  ",
        f"Config: `{config_name}`",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| MOTA   | {mean_mota:.1f}% |",
        f"| MOTP   | {mean_motp:.1f}% |",
        f"| FP     | {total_fp} |",
        f"| FN     | {total_fn} |",
        f"| IDSW   | {total_sw} |",
        f"| GT     | {total_gt} |",
        f"| Hz     | {mean_hz:.1f} |",
        "",
        "## Per-Sequence Results",
        "",
        "| Sequence | Frames | MOTA | MOTP | FP | FN | IDSW | Hz |",
        "|----------|--------|------|------|----|----|------|----|",
    ]

    for s in sorted(seq_metrics, key=lambda x: x["seq_name"]):
        lines.append(
            f"| {s['seq_name']} | {s['n_frames']} | "
            f"{s['MOTA']:.1f}% | {s['MOTP']:.1f}% | "
            f"{s['FP']} | {s['FN']} | {s['IDSW']} | {s['hz']} |"
        )

    lines += [
        "",
        "## System",
        "",
        "- Detector  : YOLOv8n (fp16, cuda:0)",
        "- Tracker   : ByteTrack two-stage IoU association",
        "- State est : Kalman Filter (Joseph form, NIS-validated)",
        "- Dataset   : VisDrone2019-MOT-train",
        "- IoU thresh: 0.5 (MOTA matching)",
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
    seq_root     = dataset_path / "sequences"
    ann_root     = dataset_path / "annotations"

    if not seq_root.exists():
        print(f"Error: sequences not found at {seq_root}")
        sys.exit(1)

    # Discover sequences
    all_seqs = sorted([d.name for d in seq_root.iterdir() if d.is_dir()])

    if args.sequences:
        seqs = args.sequences
        missing = [s for s in seqs if not (seq_root / s).exists()]
        if missing:
            print(f"Error: sequences not found: {missing}")
            sys.exit(1)
    else:
        seqs = all_seqs

    if args.max_seqs:
        seqs = seqs[:args.max_seqs]

    # Load config
    cfg     = load_config(args.config)
    raw_cfg = cfg.as_dict()

    print(f"\nVisDrone2019 Benchmark")
    print(f"  Dataset   : {dataset_path}")
    print(f"  Sequences : {len(seqs)}")
    print(f"  Device    : {cfg.detector.device}")
    print(f"  Model     : {cfg.detector.model}")

    # Build detector once — shared across all sequences
    print("\nLoading detector ...")
    detector = YOLOv8Detector(raw_cfg)
    detector.warmup()
    print(f"Detector ready. Latency: {detector.mean_inference_ms:.1f} ms")

    # Run each sequence
    out_dir     = Path(args.out)
    seq_metrics = []

    for seq_name in seqs:
        seq_dir  = seq_root / seq_name
        ann_path = ann_root / f"{seq_name}.txt"
        out_file = out_dir  / f"{seq_name}.txt"

        try:
            metrics = run_sequence(
                seq_name=seq_name,
                seq_dir=seq_dir,
                ann_path=ann_path,
                detector=detector,
                config=raw_cfg,
                out_path=out_file,
            )
            seq_metrics.append(metrics)
        except Exception as e:
            print(f"  Error on {seq_name}: {e}")
            continue

    if not seq_metrics:
        print("No sequences completed.")
        sys.exit(1)

    # Final summary
    mean_mota = float(np.mean([s["MOTA"] for s in seq_metrics]))
    mean_hz   = float(np.mean([s["hz"]   for s in seq_metrics]))
    print(f"\n{'='*50}")
    print(f"Final Results ({len(seq_metrics)} sequences)")
    print(f"  Mean MOTA : {mean_mota:.1f}%")
    print(f"  Mean Hz   : {mean_hz:.1f}")
    print(f"{'='*50}")

    write_markdown(
        seq_metrics=seq_metrics,
        out_path=Path("docs/benchmark_results.md"),
        config_name=args.config,
    )


if __name__ == "__main__":
    main()
