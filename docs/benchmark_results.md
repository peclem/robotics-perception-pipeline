# Benchmark Results — MOT17

Generated: 2026-05-22 17:47:21  
Config: `/tmp/cmc_on.yaml`

## Overall Metrics

| Metric | Value |
|--------|-------|
| MOTA   | 49.2% |
| MOTP   | 76.8% |
| FP     | 5721 |
| FN     | 28249 |
| IDSW   | 357 |

## Per-Sequence Results

| Sequence | Frames | MOTA | MOTP | FP | FN | IDSW | Hz |
|----------|--------|------|------|----|----|------|----|
| MOT17-04-FRCNN | 1050 | 65.9% | 81.14 | 3973 | 12168 | 76 | 14.6 |
| MOT17-05-FRCNN | 837 | 52.7% | 77.1 | 460 | 2662 | 149 | 41.8 |
| MOT17-10-FRCNN | 654 | 45.4% | 75.76 | 671 | 6267 | 68 | 12.6 |
| MOT17-13-FRCNN | 750 | 32.7% | 73.41 | 617 | 7152 | 64 | 12.2 |

## System

- Detector: YOLOv8n (fp16, cuda:0)
- Tracker: ByteTrack two-stage IoU association
- State estimation: Kalman Filter (Joseph form, NIS-validated)
- Dataset: MOT17-train
- IoU threshold: 0.5
