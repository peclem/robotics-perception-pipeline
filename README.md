# Robotics Perception Pipeline

![CI](https://github.com/peclem/robotics-perception-pipeline/actions/workflows/ci.yml/badge.svg)

Camera-based multi-object tracking system with Kalman filter state estimation,
designed as a modular robotics perception stack.

## Stack
- **Detection** — YOLOv8n (Ultralytics, ONNX-exportable)
- **Tracking** — ByteTrack two-stage IoU association
- **State estimation** — Kalman filter (from scratch, Joseph form)
- **Visualization** — Rerun.io + OpenCV annotator
- **Config** — Typed PipelineConfig with validation and env overrides

## Quick start
```bash
git clone https://github.com/peclem/robotics-perception-pipeline
cd robotics-perception-pipeline
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python-headless filterpy scipy numpy pyyaml pytest rerun-sdk
pip install -e .
RERUN_ENABLED=false python3 launch.py --source synthetic
```

## Run tests
```bash
python -m pytest tests/ -m "not integration" -v
```
