"""
Fine-tune YOLOv8n on MOT17 for pedestrian detection.

Usage:
    python3 scripts/train_detector.py --data data/mot17_yolo/mot17.yaml

Output:
    runs/detect/mot17_finetune/weights/best.pt   ← use this
    runs/detect/mot17_finetune/weights/last.pt

After training, update config/default.yaml:
    detector:
      model: "runs/detect/mot17_finetune/weights/best.pt"

Then re-run benchmark:
    python3 scripts/benchmark.py --dataset data/MOT17 --split train --out data/mot17_results_finetuned
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n on MOT17"
    )
    parser.add_argument(
        "--data",
        default="data/mot17_yolo/mot17.yaml",
        help="Path to YOLO dataset YAML",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base model to fine-tune (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",  type=int,   default=50,
        help="Training epochs (default: 50)",
    )
    parser.add_argument(
        "--imgsz",   type=int,   default=1280,
        help="Training image size (default: 1280)",
    )
    parser.add_argument(
        "--batch",   type=int,   default=8,
        help="Batch size (default: 8 for 4070Ti 12GB)",
    )
    parser.add_argument(
        "--name",    default="mot17_finetune",
        help="Run name (default: mot17_finetune)",
    )
    parser.add_argument(
        "--device",  default="0",
        help="CUDA device (default: 0)",
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early stopping patience (default: 15)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: dataset YAML not found: {data_path}")
        print("Run first: python3 scripts/mot17_to_yolo.py")
        return

    from ultralytics import YOLO

    print(f"\nFine-tuning {args.model} on MOT17")
    print(f"  Data    : {args.data}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  ImgSz   : {args.imgsz}")
    print(f"  Batch   : {args.batch}")
    print(f"  Device  : cuda:{args.device}")
    print(f"  Name    : {args.name}\n")

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience,

        # Pedestrian-specific hyperparameters
        # Lower LR for fine-tuning — we're adapting, not training from scratch
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,

        # Augmentation — conservative for surveillance footage
        # Heavy augmentation hurts performance on fixed-camera scenes
        hsv_h=0.01,      # minimal hue shift
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=0.0,      # no rotation — pedestrians are always upright
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.5,       # reduced mosaic — objects are small, mosaic distorts them

        # Training settings
        workers=0,
        cache='disk',       # cache images in RAM for speed — needs ~8GB RAM
        close_mosaic=10,  # disable mosaic last 10 epochs for stable convergence
        amp=True,         # automatic mixed precision — fp16 on 4070Ti

        # Validation
        val=True,
        save_period=10,   # save checkpoint every 10 epochs
        plots=True,
    )

    best = Path("runs/detect") / args.name / "weights/best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights: {best}")
    print(f"\nUpdate config/default.yaml:")
    print(f'  model: "{best}"')
    print(f"\nThen benchmark:")
    print(f"  python3 scripts/benchmark.py \\")
    print(f"    --dataset data/MOT17 --split train \\")
    print(f"    --out data/mot17_results_finetuned")


if __name__ == "__main__":
    main()
