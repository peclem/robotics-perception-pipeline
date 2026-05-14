"""
Robotics Perception Pipeline — main entry point.

Wires together:
    CameraInterface  (Step 2)
    YOLOv8Detector   (Step 3)
    KalmanFilter     (Step 4, owned by ByteTracker)
    ByteTracker      (Step 5)
    PipelineConfig   (Step 6)
    DebugVisualizer  (Step 7 — OpenCV annotator + optional Rerun.io)

Usage
-----
# Run on a video file (headless — writes annotated MP4):
    python launch.py --source video --input data/test_clip.mp4

# Run on webcam:
    python launch.py --source webcam

# Run on synthetic camera (no hardware needed):
    python launch.py --source synthetic

# Override config values via env:
    DEVICE=cpu python launch.py --source synthetic

# Disable Rerun.io streaming:
    RERUN_ENABLED=false python launch.py --source synthetic

WSL notes
---------
- Webcam requires usbipd-win attachment before launch.
- cv2.imshow() is NOT used — output is written to an .mp4 file.
- Rerun.io viewer runs on Windows; WSL connects via TCP automatically.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from perception.camera_interface import (
    CameraFrame,
    SyntheticCamera,
    VideoFileCamera,
    WebcamCamera,
)
from perception.config_loader import load_config, PipelineConfig
from perception.detector import YOLOv8Detector
from tracking.tracker import ByteTracker
from tracking.track import Track
from visualization.debug_vis import DebugVisualizer


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


log = logging.getLogger("launch")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Main pipeline orchestrator.

    Lifecycle
    ---------
    pipeline = Pipeline(cfg)
    pipeline.run()           # blocks until EOF or KeyboardInterrupt
    pipeline.release()       # called automatically via signal handler

    Responsibilities
    ----------------
    - Build and own all module instances
    - Drive the frame loop
    - Measure and log per-frame latency
    - Write annotated output video
    - Graceful shutdown on SIGINT / SIGTERM
    """

    def __init__(self, cfg: PipelineConfig, source: str, input_path: Optional[str]) -> None:
        self._cfg        = cfg
        self._source     = source
        self._input_path = input_path
        self._running    = False

        # Timing
        self._frame_times: list[float] = []
        self._target_dt = 1.0 / cfg.pipeline.target_hz

        # Modules — built in open()
        self._camera:     Optional[object]          = None
        self._detector:   Optional[YOLOv8Detector]  = None
        self._tracker:    Optional[ByteTracker]     = None
        self._visualizer: Optional[DebugVisualizer] = None
        self._writer:     Optional[cv2.VideoWriter] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Build modules, run the frame loop, release on exit."""
        self._open()
        self._running = True

        log.info("Pipeline running — press Ctrl+C to stop.")
        log.info(
            "Source: %s | Device: %s | Target: %.0f Hz",
            self._source,
            self._cfg.detector.device,
            self._cfg.pipeline.target_hz,
        )

        try:
            self._loop()
        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — shutting down.")
        finally:
            self._release()

    # ------------------------------------------------------------------
    # Internal: build
    # ------------------------------------------------------------------

    def _open(self) -> None:
        cfg = self._cfg
        raw = cfg.as_dict()

        # Camera
        self._camera = self._build_camera(raw)
        self._camera.open()

        # Detector
        log.info("Loading detector: %s on %s ...", cfg.detector.model, cfg.detector.device)
        self._detector = YOLOv8Detector(raw)
        self._detector.warmup()
        log.info("Detector ready. Mean warmup latency: %.1f ms", self._detector.mean_inference_ms)

        # Tracker
        self._tracker = ByteTracker(raw)

        # Visualizer
        self._visualizer = DebugVisualizer(cfg)

        # Output video writer (opened lazily on first frame)
        self._writer = None

    def _build_camera(self, raw: dict):
        cfg = self._cfg
        if self._source == "webcam":
            return WebcamCamera(
                raw,
                device_index=cfg.camera.device_index,
                intrinsics_path=cfg.camera.intrinsics_path,
            )
        elif self._source == "video":
            path = self._input_path or cfg.video.input_path
            return VideoFileCamera(
                raw,
                video_path=path,
                intrinsics_path=cfg.camera.intrinsics_path,
                loop=cfg.video.loop,
                playback_fps=cfg.video.playback_fps,
            )
        elif self._source == "synthetic":
            return SyntheticCamera(
                raw,
                width=cfg.synthetic_camera.width,
                height=cfg.synthetic_camera.height,
                num_frames=cfg.synthetic_camera.num_frames,
                fps=cfg.synthetic_camera.fps,
                num_objects=cfg.synthetic_camera.num_objects,
                seed=cfg.synthetic_camera.seed,
            )
        else:
            raise ValueError(f"Unknown source: {self._source!r}. Use webcam/video/synthetic.")

    def _open_writer(self, frame: CameraFrame) -> cv2.VideoWriter:
        """
        Open the VideoWriter on the first frame so we know the actual
        frame dimensions. Uses MP4V codec — works on all platforms.
        """
        out_path = (
            self._cfg.pipeline.output_video_path
            or self._cfg.video.output_path
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_path),
            fourcc,
            float(self._cfg.camera.fps),
            (frame.width, frame.height),
        )
        if not writer.isOpened():
            log.warning("Could not open VideoWriter at %s — no output file.", out_path)
            return None

        log.info("Writing annotated output to: %s", out_path)
        return writer

    # ------------------------------------------------------------------
    # Internal: frame loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        frame_idx = 0

        while self._running:
            t_frame_start = time.monotonic()

            # 1. Grab frame
            frame = self._camera.get_frame()
            if frame is None:
                log.info("Source exhausted after %d frames.", frame_idx)
                break

            # 2. Open writer on first frame
            if frame_idx == 0 and self._writer is None:
                self._writer = self._open_writer(frame)

            # 3. Detect
            t0 = time.monotonic()
            detections = self._detector.detect(frame)
            t_detect = time.monotonic() - t0

            # 4. Track
            t0 = time.monotonic()
            confirmed_tracks = self._tracker.update(detections, frame)
            t_track = time.monotonic() - t0

            # 5. Annotate + write
            annotated = self._visualizer.draw(
                frame=frame,
                detections=detections,
                tracks=confirmed_tracks,
                detect_ms=t_detect * 1000,
                track_ms=t_track * 1000,
            )

            if self._writer is not None:
                self._writer.write(annotated)

            # 6. Log to Rerun.io
            self._visualizer.log_rerun(
                frame=frame,
                detections=detections,
                tracks=confirmed_tracks,
            )

            # 7. Timing
            t_total = time.monotonic() - t_frame_start
            self._frame_times.append(t_total)

            if t_total > self._target_dt * 1.5:
                log.debug(
                    "Frame %d slow: %.1f ms (target %.1f ms)",
                    frame_idx, t_total * 1000, self._target_dt * 1000,
                )

            # 8. Progress log every 30 frames
            if frame_idx % 30 == 0:
                fps = 1.0 / np.mean(self._frame_times[-30:]) if self._frame_times else 0.0
                log.info(
                    "Frame %4d | dets=%2d | tracks=%2d | lost=%2d | "
                    "det=%.1fms | trk=%.1fms | %.1f FPS",
                    frame_idx,
                    len(detections),
                    len(confirmed_tracks),
                    len(self._tracker.lost_tracks),
                    t_detect * 1000,
                    t_track * 1000,
                    fps,
                )

            frame_idx += 1

    # ------------------------------------------------------------------
    # Internal: cleanup
    # ------------------------------------------------------------------

    def _release(self) -> None:
        self._running = False

        if self._camera is not None:
            self._camera.release()

        if self._writer is not None:
            self._writer.release()
            log.info("Output video saved.")

        if self._visualizer is not None:
            self._visualizer.close()

        self._print_summary()

    def _print_summary(self) -> None:
        if not self._frame_times:
            return
        times = np.array(self._frame_times) * 1000
        log.info("=" * 52)
        log.info("Pipeline summary")
        log.info("  Frames processed : %d",   len(times))
        log.info("  Mean latency     : %.2f ms", times.mean())
        log.info("  P95 latency      : %.2f ms", np.percentile(times, 95))
        log.info("  Max latency      : %.2f ms", times.max())
        log.info("  Mean FPS         : %.1f",   1000.0 / times.mean())
        log.info("  Max tracks seen  : %d",
                 max((t.track_id for t in self._tracker.tracks), default=0))
        log.info("=" * 52)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robotics Perception Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py --source synthetic
  python launch.py --source video --input data/test_clip.mp4
  python launch.py --source webcam
  DEVICE=cpu python launch.py --source synthetic
  RERUN_ENABLED=false python launch.py --source synthetic
        """,
    )
    parser.add_argument(
        "--source",
        choices=["webcam", "video", "synthetic"],
        default="synthetic",
        help="Camera source backend (default: synthetic)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input video path (required when --source video)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to config YAML (default: config/default.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output video path from config",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Load config first so logging level is respected
    cfg = load_config(args.config)
    _setup_logging(cfg.pipeline.log_level)

    # CLI --output overrides config
    if args.output:
        cfg.pipeline.output_video_path = args.output

    # Validate source-specific requirements
    if args.source == "video" and args.input is None and not Path(cfg.video.input_path).exists():
        log.error(
            "--source video requires --input <path> or a valid "
            "video.input_path in %s", args.config
        )
        sys.exit(1)

    # Register shutdown signals
    pipeline = Pipeline(cfg, source=args.source, input_path=args.input)

    def _handle_signal(sig, frame):
        log.info("Signal %d received — shutting down.", sig)
        pipeline._running = False

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    pipeline.run()


if __name__ == "__main__":
    main()
