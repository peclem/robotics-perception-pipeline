"""
Camera calibration script — generates config/camera_intrinsics.yaml.

Usage
-----
# Live webcam (requires usbipd-win in WSL):
python3 scripts/calibrate_camera.py --device 0 --out config/camera_intrinsics.yaml

# From a video file of a checkerboard:
python3 scripts/calibrate_camera.py --input data/calib.mp4 --out config/camera_intrinsics.yaml

# Custom checkerboard size (rows x cols of inner corners):
python3 scripts/calibrate_camera.py --rows 9 --cols 6 --size 0.025

Checkerboard
------------
Print a 9x6 checkerboard with 25mm squares on A4 paper (or use the
default 9x6 pattern from OpenCV samples). Tape it flat — any warping
increases reprojection error.

'--rows' and '--cols' are the number of INNER corners, not squares.
A standard 10x7 printed checkerboard has 9x6 inner corners.

Output format
-------------
Writes config/camera_intrinsics.yaml compatible with load_intrinsics():
    camera_matrix: {fx, fy, cx, cy}
    image_size:    {width, height}
    dist_coeffs:   [k1, k2, p1, p2, k3]
    reprojection_error: float

Calibration tips
----------------
- Use at least 15 frames from different angles and distances.
- Cover the full frame area — don't cluster frames in the centre.
- Hold the board still for each capture — motion blur degrades corners.
- Reprojection error < 1.0px is usable. < 0.5px is good.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate camera and write intrinsics YAML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Video file path instead of live webcam",
    )
    parser.add_argument(
        "--rows", type=int, default=9,
        help="Inner corner rows on checkerboard (default: 9)",
    )
    parser.add_argument(
        "--cols", type=int, default=6,
        help="Inner corner columns on checkerboard (default: 6)",
    )
    parser.add_argument(
        "--size", type=float, default=0.025,
        help="Physical square size in metres (default: 0.025 = 25mm)",
    )
    parser.add_argument(
        "--min-frames", type=int, default=15,
        help="Minimum good frames before calibrating (default: 15)",
    )
    parser.add_argument(
        "--out", type=str, default="config/camera_intrinsics.yaml",
        help="Output YAML path (default: config/camera_intrinsics.yaml)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable live preview (for WSL headless mode)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def collect_frames(
    cap:        cv2.VideoCapture,
    rows:       int,
    cols:       int,
    min_frames: int,
    show:       bool,
) -> tuple[list, list, tuple[int, int]]:
    """
    Capture frames and detect checkerboard corners.

    Returns
    -------
    obj_points  : list of (N, 3) float32 arrays — 3D world coordinates
    img_points  : list of (N, 2) float32 arrays — 2D pixel coordinates
    image_size  : (width, height) of captured frames
    """
    # 3D object points for one checkerboard view
    # (0,0,0), (1,0,0), (2,0,0) ... scaled by square size
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    obj_points: list = []
    img_points: list = []
    image_size: tuple = (0, 0)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30, 0.001,
    )

    n_good    = 0
    n_checked = 0

    print(f"\nLooking for {rows}×{cols} checkerboard inner corners.")
    print(f"Need {min_frames} good frames. Move the board to different angles.")
    print("Press 'q' to quit early and calibrate with current frames.\n")

    while n_good < min_frames:
        ret, frame = cap.read()
        if not ret:
            print("Camera source exhausted.")
            break

        n_checked += 1
        if image_size == (0, 0):
            h, w = frame.shape[:2]
            image_size = (w, h)

        # Only check every 5th frame — detection is slow
        if n_checked % 5 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if found:
            # Sub-pixel refinement
            corners_sub = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            obj_points.append(objp.copy())
            img_points.append(corners_sub)
            n_good += 1
            print(f"  ✓ Frame {n_good}/{min_frames} captured")

            if show:
                display = frame.copy()
                cv2.drawChessboardCorners(display, (cols, rows), corners_sub, found)
                cv2.putText(
                    display, f"Good: {n_good}/{min_frames}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 220, 60), 2,
                )
                cv2.imshow("Calibration", display)
        else:
            if show:
                cv2.putText(
                    frame, f"No corners | Good: {n_good}/{min_frames}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 80, 220), 2,
                )
                cv2.imshow("Calibration", frame)

        if show:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Early quit — calibrating with current frames.")
                break

    if show:
        cv2.destroyAllWindows()

    return obj_points, img_points, image_size


def calibrate(
    obj_points: list,
    img_points: list,
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Run OpenCV camera calibration.

    Returns
    -------
    reprojection_error : mean reprojection error in pixels
    camera_matrix      : (3, 3) intrinsic matrix K
    dist_coeffs        : (5,) distortion coefficients [k1,k2,p1,p2,k3]
    """
    print(f"\nCalibrating with {len(obj_points)} frames...")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    # Compute per-view reprojection errors for diagnostics
    errors = []
    for i, (objp, imgp) in enumerate(zip(obj_points, img_points)):
        proj, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgp, proj, cv2.NORM_L2) / len(proj)
        errors.append(err)

    mean_err = float(np.mean(errors))
    return mean_err, K, dist.flatten()


def save_intrinsics(
    path:        str,
    K:           np.ndarray,
    dist:        np.ndarray,
    image_size:  tuple[int, int],
    reproj_err:  float,
    square_size: float,
    n_frames:    int,
) -> None:
    """Write calibration results to YAML."""
    out = {
        "camera_matrix": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        },
        "image_size": {
            "width":  int(image_size[0]),
            "height": int(image_size[1]),
        },
        "dist_coeffs": [float(d) for d in dist[:5]],
        "reprojection_error": round(reproj_err, 4),
        "calibration_info": {
            "n_frames":    n_frames,
            "square_size_m": square_size,
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Open capture
    if args.input:
        source = args.input
        if not Path(source).exists():
            print(f"Error: input file not found: {source}")
            sys.exit(1)
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(args.device)

    if not cap.isOpened():
        print("Error: could not open camera/video source.")
        print("WSL tip: attach webcam with usbipd-win before running.")
        sys.exit(1)

    show = not args.no_display

    # Collect frames
    obj_points, img_points, image_size = collect_frames(
        cap, args.rows, args.cols, args.min_frames, show
    )
    cap.release()

    n_good = len(obj_points)
    if n_good < 4:
        print(f"\nError: only {n_good} good frames — need at least 4 to calibrate.")
        print("Tips: ensure good lighting, hold the board flat, move slowly.")
        sys.exit(1)

    # Calibrate
    reproj_err, K, dist = calibrate(obj_points, img_points, image_size)

    # Save
    save_intrinsics(
        path=args.out,
        K=K,
        dist=dist,
        image_size=image_size,
        reproj_err=reproj_err,
        square_size=args.size,
        n_frames=n_good,
    )

    # Report
    print(f"\n{'='*50}")
    print(f"Calibration complete")
    print(f"  Frames used      : {n_good}")
    print(f"  Resolution       : {image_size[0]}×{image_size[1]}")
    print(f"  fx               : {K[0,0]:.2f} px")
    print(f"  fy               : {K[1,1]:.2f} px")
    print(f"  cx               : {K[0,2]:.2f} px")
    print(f"  cy               : {K[1,2]:.2f} px")
    print(f"  Distortion k1    : {dist[0]:.4f}")
    print(f"  Reprojection err : {reproj_err:.4f} px  ", end="")
    if reproj_err < 0.5:
        print("✓ excellent")
    elif reproj_err < 1.0:
        print("✓ good")
    else:
        print("⚠ high — recalibrate with more frames or flatter board")
    print(f"  Saved to         : {args.out}")
    print(f"{'='*50}\n")

    # Verify the output is loadable
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from perception.camera_interface import load_intrinsics
    intr = load_intrinsics(args.out)
    print(f"Verified: load_intrinsics() OK → fx={intr.fx:.2f} fy={intr.fy:.2f}")


if __name__ == "__main__":
    main()
