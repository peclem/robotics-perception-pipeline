"""
Convert MOT17 ground truth to YOLO detection format.

MOT17 GT format (gt/gt.txt):
    frame, id, x1, y1, w, h, conf, class, visibility
    conf=0 → ignore region, conf=1 → valid
    class=1 → pedestrian (only class we use)

YOLO format (per image .txt):
    class_id cx cy w h   (all normalised to [0,1])

Output structure:
    data/mot17_yolo/
        images/
            train/  ← symlinks to original JPEGs
            val/
        labels/
            train/  ← YOLO .txt files
            val/
        mot17.yaml  ← YOLO dataset config

Usage:
    python3 scripts/mot17_to_yolo.py --mot17 data/MOT17 --out data/mot17_yolo
"""

from __future__ import annotations

import argparse
import configparser
import os
import shutil
from pathlib import Path

import yaml


# Sequences used — FRCNN variant only (images identical across DPM/FRCNN/SDP)
TRAIN_SEQS = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-13-FRCNN",
]
VAL_SEQS = [
    "MOT17-09-FRCNN",
    "MOT17-11-FRCNN",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot17", default="data/MOT17",
                        help="Path to MOT17 root")
    parser.add_argument("--out",   default="data/mot17_yolo",
                        help="Output directory")
    return parser.parse_args()


def read_seqinfo(seq_path: Path) -> dict:
    ini = configparser.ConfigParser()
    ini.read(seq_path / "seqinfo.ini")
    s = ini["Sequence"]
    return {
        "width":  int(s["imWidth"]),
        "height": int(s["imHeight"]),
        "ext":    s["imExt"],
    }


def convert_sequence(
    seq_path: Path,
    split:    str,
    out_dir:  Path,
) -> int:
    """Convert one MOT17 sequence. Returns number of label files written."""
    info   = read_seqinfo(seq_path)
    W, H   = info["width"], info["height"]
    ext    = info["ext"]
    gt_path = seq_path / "gt" / "gt.txt"
    img_dir = seq_path / "img1"

    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # Load GT — group by frame
    gt: dict[int, list] = {}
    with open(gt_path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 9:
                continue
            frame    = int(p[0])
            conf     = int(p[6])
            cls      = int(p[7])
            vis      = float(p[8])
            if conf == 0 or cls != 1:      # skip ignore regions, non-pedestrian
                continue
            if vis < 0.25:                  # skip heavily occluded
                continue
            x1, y1  = float(p[2]), float(p[3])
            w,  h   = float(p[4]), float(p[5])
            # Convert to YOLO format (normalised cx, cy, w, h)
            cx = (x1 + w / 2.0) / W
            cy = (y1 + h / 2.0) / H
            nw = w / W
            nh = h / H
            # Clip to [0,1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            gt.setdefault(frame, []).append((cx, cy, nw, nh))

    seq_name = seq_path.name
    n_written = 0

    for frame_id, boxes in gt.items():
        img_name = f"{frame_id:06d}{ext}"
        src_img  = img_dir / img_name
        if not src_img.exists():
            continue

        # Symlink image (saves disk space)
        dst_img = img_out / f"{seq_name}_{img_name}"
        if not dst_img.exists():
            os.symlink(src_img.resolve(), dst_img)

        # Write label file
        lbl_file = lbl_out / f"{seq_name}_{img_name.replace(ext, '.txt')}"
        with open(lbl_file, "w") as f:
            for cx, cy, w, h in boxes:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        n_written += 1

    return n_written


def write_yaml(out_dir: Path) -> None:
    cfg = {
        "path":  str(out_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": {0: "person"},
    }
    with open(out_dir / "mot17.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def main() -> None:
    args    = parse_args()
    mot17   = Path(args.mot17)
    out_dir = Path(args.out)
    split_path = mot17 / "train"

    print(f"\nConverting MOT17 → YOLO format")
    print(f"  Source : {split_path}")
    print(f"  Output : {out_dir}")

    total = 0
    for seq_name in TRAIN_SEQS:
        seq = split_path / seq_name
        if not seq.exists():
            print(f"  Warning: {seq_name} not found, skipping")
            continue
        n = convert_sequence(seq, "train", out_dir)
        print(f"  train  {seq_name}: {n} frames")
        total += n

    for seq_name in VAL_SEQS:
        seq = split_path / seq_name
        if not seq.exists():
            print(f"  Warning: {seq_name} not found, skipping")
            continue
        n = convert_sequence(seq, "val", out_dir)
        print(f"  val    {seq_name}: {n} frames")
        total += n

    write_yaml(out_dir)
    print(f"\nTotal frames converted: {total}")
    print(f"Dataset YAML: {out_dir / 'mot17.yaml'}")
    print("\nNext step:")
    print(f"  python3 scripts/train_detector.py --data {out_dir}/mot17.yaml")


if __name__ == "__main__":
    main()
