"""
Run the pipeline on a video file.

Usage:
    python scripts/run_video.py --input data/test_clip.mp4
    python scripts/run_video.py --input data/test_clip.mp4 --output data/out.mp4
    DEVICE=cpu python scripts/run_video.py --input data/test_clip.mp4
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from launch import main as _launch_main
import sys

if __name__ == "__main__":
    # Inject --source video so launch.py's arg parser sees it
    if "--source" not in sys.argv:
        sys.argv.extend(["--source", "video"])
    _launch_main()
