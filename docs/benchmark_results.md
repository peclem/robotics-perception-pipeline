# Benchmark Results — MOT17

Generated: 2026-05-18 19:02:28  
Config: `/tmp/bench_appearance.yaml`

## Overall Metrics

| Metric | Value |
|--------|-------|
| MOTA   | 50.7% |
| MOTP   | 78.2% |
| FP     | 1724 |
| FN     | 5523 |
| IDSW   | 116 |

## Per-Sequence Results

| Sequence | Frames | MOTA | MOTP | FP | FN | IDSW | Hz |
|----------|--------|------|------|----|----|------|----|
| MOT17-09-SDP | 525 | 52.7% | 78.1% | 723 | 1742 | 54 | 31.0 |
| MOT17-11-SDP | 900 | 48.7% | 78.3% | 1001 | 3781 | 62 | 30.9 |

## System

- Detector: YOLOv8n (fp16, cuda:0)
- Tracker: ByteTrack two-stage IoU association
- State estimation: Kalman Filter (Joseph form, NIS-validated)
- Dataset: MOT17-train
- IoU threshold: 0.5
