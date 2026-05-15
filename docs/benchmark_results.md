# Benchmark Results — MOT17

Generated: 2026-05-15 11:50:42  
Config: `config/default.yaml`

## Overall Metrics

| Metric | Value |
|--------|-------|
| MOTA   | 51.6% |
| MOTP   | 77.7% |
| FP     | 747 |
| FN     | 6338 |
| IDSW   | 95 |

## Per-Sequence Results

| Sequence | Frames | MOTA | MOTP | FP | FN | IDSW | Hz |
|----------|--------|------|------|----|----|------|----|
| MOT17-09-FRCNN | 525 | 52.4% | 76.9% | 392 | 2084 | 59 | 134.1 |
| MOT17-11-FRCNN | 900 | 50.8% | 78.6% | 355 | 4254 | 36 | 111.1 |

## System

- Detector: YOLOv8n (fp16, cuda:0)
- Tracker: ByteTrack two-stage IoU association
- State estimation: Kalman Filter (Joseph form, NIS-validated)
- Dataset: MOT17-train
- IoU threshold: 0.5
