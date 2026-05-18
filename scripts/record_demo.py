"""
Record a Rerun .rrd + annotated .mp4 of the pipeline for a showcase.

Thin wrapper around launch.py that sets sensible recording defaults:
  - Rerun enabled and saved to disk (not streamed)
  - Annotated mp4 written next to it
  - Same configuration file as production runs

Usage:
    # From a video file
    python3 scripts/record_demo.py --input data/your_clip.mp4 --out data/demo

    # From a USB webcam
    python3 scripts/record_demo.py --webcam --out data/webcam_demo

    # From an IP-camera stream (DroidCam, Larix Broadcaster, OBS NDI, …)
    python3 scripts/record_demo.py --input rtsp://phone:1935/live \\
        --out data/iphone_demo

Produces:
    {out}.rrd      — Rerun recording, openable at https://rerun.io/viewer
    {out}.mp4      — Annotated mp4 (bbox, track IDs, velocities, …)

To override perception config (e.g. enable VIO, semantic seg, depth,
appearance ReID, …) pass --config path/to/my.yaml. Defaults to
config/default.yaml.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record a showcase Rerun .rrd + annotated .mp4 of the pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input",  type=str,
                     help="Video file or stream URL (e.g. data/x.mp4, "
                          "rtsp://phone:1935/live).")
    src.add_argument("--webcam", action="store_true",
                     help="Use the default USB webcam (device index 0).")

    p.add_argument(
        "--out", type=str, required=True,
        help="Output basename (no extension); both .rrd and .mp4 will "
             "be written next to it. Parent directory is created if "
             "needed. Example: data/demo  →  data/demo.rrd + data/demo.mp4",
    )
    p.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Pipeline config (default: config/default.yaml).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out  = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rrd_path = out.with_suffix(".rrd")
    mp4_path = out.with_suffix(".mp4")

    # Translate into launch.py's CLI + env. We splice into sys.argv so
    # launch.py's argparse picks it up unchanged.
    sys.argv = [sys.argv[0]]
    if args.webcam:
        sys.argv.extend(["--source", "webcam"])
    else:
        sys.argv.extend(["--source", "video", "--input", args.input])
    sys.argv.extend([
        "--config",     args.config,
        "--output",     str(mp4_path),
        "--rerun-save", str(rrd_path),
    ])
    # Rerun is opt-in via env in launch.py — make sure it's on.
    os.environ["RERUN_ENABLED"] = "true"

    print(f"  Recording to: {rrd_path}")
    print(f"  Annotated mp4: {mp4_path}")
    print(f"  Config:        {args.config}")
    print()

    from launch import main as _launch_main
    _launch_main()

    print()
    print("Done. Share the recording:")
    print(f"  - Drop {rrd_path} into https://rerun.io/viewer")
    print(f"  - Upload {mp4_path} to GitHub / YouTube / slides / Twitter")


if __name__ == "__main__":
    main()
