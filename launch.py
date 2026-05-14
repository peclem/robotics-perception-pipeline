"""
Robotics Perception Pipeline — main entry point.

Usage
-----
    python3 launch.py --source synthetic
    python3 launch.py --source video --input data/test_clip.mp4
    python3 launch.py --source webcam
    RERUN_ENABLED=false python3 launch.py --source synthetic
    python3 launch.py --source synthetic --rerun-save data/recording.rrd
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

    def __init__(
        self,
        cfg:        PipelineConfig,
        source:     str,
        input_path: Optional[str],
    ) -> None:
        self._cfg        = cfg
        self._source     = source
        self._input_path = input_path
        self._running    = False

        self._frame_times: list[float] = []
        self._target_dt = 1.0 / cfg.pipeline.target_hz

        self._camera     = None
        self._detector   = None
        self._tracker    = None
        self._visualizer = None
        self._writer     = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
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
    # Build
    # ------------------------------------------------------------------

    def _open(self) -> None:
        cfg = self._cfg
        raw = cfg.as_dict()

        self._camera = self._build_camera(raw)
        self._camera.open()

        log.info(
            "Loading detector: %s on %s ...",
            cfg.detector.model,
            cfg.detector.device,
        )
        self._detector = YOLOv8Detector(raw)
        self._detector.warmup()
        log.info(
            "Detector ready. Mean warmup latency: %.1f ms",
            self._detector.mean_inference_ms,
        )

        self._tracker    = ByteTracker(raw)
        self._visualizer = DebugVisualizer(cfg)
        self._visualizer.connect_rerun()
        self._writer     = None

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
            raise ValueError(
                f"Unknown source: {self._source!r}. "
                "Use: webcam / video / synthetic"
            )

    def _open_writer(self, frame: CameraFrame) -> Optional[cv2.VideoWriter]:
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
            log.warning(
                "Could not open VideoWriter at %s — no output file.", out_path
            )
            return None

        log.info("Writing annotated output to: %s", out_path)
        return writer

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        frame_idx = 0

        while self._running:
            t_frame_start = time.monotonic()

            frame = self._camera.get_frame()
            if frame is None:
                log.info("Source exhausted after %d frames.", frame_idx)
                break

            if frame_idx == 0 and self._writer is None:
                self._writer = self._open_writer(frame)

            t0          = time.monotonic()
            detections  = self._detector.detect(frame)
            t_detect    = time.monotonic() - t0

            t0               = time.monotonic()
            confirmed_tracks = self._tracker.update(detections, frame)
            t_track          = time.monotonic() - t0

            annotated = self._visualizer.draw(
                frame=frame,
                detections=detections,
                tracks=confirmed_tracks,
                detect_ms=t_detect * 1000,
                track_ms=t_track * 1000,
            )

            if self._writer is not None:
                self._writer.write(annotated)

            fps = (
                1.0 / float(np.mean(self._frame_times[-10:]) or 1.0)
                if self._frame_times else 0.0
            )
            self._visualizer.log_rerun(
                frame=frame,
                detections=detections,
                tracks=confirmed_tracks,
                detect_ms=t_detect * 1000,
                track_ms=t_track * 1000,
                fps=fps,
                n_lost=len(self._tracker.lost_tracks),
            )

            t_total = time.monotonic() - t_frame_start
            self._frame_times.append(t_total)

            if frame_idx % 30 == 0:
                fps_log = (
                    1.0 / np.mean(self._frame_times[-30:])
                    if self._frame_times else 0.0
                )
                log.info(
                    "Frame %4d | dets=%2d | tracks=%2d | lost=%2d | "
                    "det=%.1fms | trk=%.1fms | %.1f FPS",
                    frame_idx,
                    len(detections),
                    len(confirmed_tracks),
                    len(self._tracker.lost_tracks),
                    t_detect * 1000,
                    t_track * 1000,
                    fps_log,
                )

            frame_idx += 1

    # ------------------------------------------------------------------
    # Cleanup
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
        log.info("  Frames processed : %d",    len(times))
        log.info("  Mean latency     : %.2f ms", times.mean())
        log.info("  P95 latency      : %.2f ms", np.percentile(times, 95))
        log.info("  Max latency      : %.2f ms", times.max())
        log.info("  Mean FPS         : %.1f",    1000.0 / times.mean())
        log.info(
            "  Max track ID     : %d",
            max((t.track_id for t in self._tracker.tracks), default=0),
        )
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
  python3 launch.py --source synthetic
  python3 launch.py --source video --input data/test_clip.mp4
  python3 launch.py --source webcam
  RERUN_ENABLED=false python3 launch.py --source synthetic
  python3 launch.py --source synthetic --rerun-save data/recording.rrd
        """,
    )
    parser.add_argument(
        "--source",
        choices=["webcam", "video", "synthetic"],
        default="synthetic",
        help="Camera source (default: synthetic)",
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
    parser.add_argument(
        "--rerun-save",
        type=str,
        default=None,
        metavar="PATH",
        help="Save .rrd recording instead of streaming "
             "(e.g. data/recording.rrd)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config)
    _setup_logging(cfg.pipeline.log_level)

    if args.output:
        cfg.pipeline.output_video_path = args.output

    if args.rerun_save:
        cfg.visualization.rerun_save_path = args.rerun_save

    if (
        args.source == "video"
        and args.input is None
        and not Path(cfg.video.input_path).exists()
    ):
        log.error(
            "--source video requires --input <path> or a valid "
            "video.input_path in %s",
            args.config,
        )
        sys.exit(1)

    pipeline = Pipeline(cfg, source=args.source, input_path=args.input)

    def _handle_signal(sig, frame):
        log.info("Signal %d received — shutting down.", sig)
        pipeline._running = False

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    pipeline.run()


if __name__ == "__main__":
    main()
