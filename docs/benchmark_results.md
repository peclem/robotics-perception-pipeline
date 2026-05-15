# Benchmark Results — MOT17

Generated: 2026-05-15 06:47:45  
Config: `config/default.yaml`

## Overall Metrics

| Metric | Value |
|--------|-------|
| MOTA   | 29.2% |
| MOTP   | 80.8% |
| FP     | 3592 |
| FN     | 30044 |
| IDSW   | 25 |

## Per-Sequence Results

| Sequence | Frames | MOTA | MOTP | FP | FN | IDSW | Hz |
|----------|--------|------|------|----|----|------|----|
| MOT17-04-FRCNN | 1050 | 29.2% | 80.8% | 3592 | 30044 | 25 | 133.0 |

## System

- Detector: YOLOv8n (fp16, cuda:0)
- Tracker: ByteTrack two-stage IoU association
- State estimation: Kalman Filter (Joseph form, NIS-validated)
- Dataset: MOT17-train
- IoU threshold: 0.5
