#!/usr/bin/env python3
"""Phase 1 - Court calibration from a single reference image.

Examples
--------
    # calibrate from a video (uses frame 0 by default) and write outputs
    python calibrate.py --source ~/Desktop/new_test1.mp4

    # calibrate from a still photo of the empty court
    python calibrate.py --source court.jpg

    # pick a different frame, review the automatic result in a window
    python calibrate.py --source match.mp4 --frame 120 --review

    # skip automatic detection and click the 14 keypoints yourself
    python calibrate.py --source match.mp4 --manual

    # check an existing calibration ("is the green light on?")
    python calibrate.py --check data/calibration/calibration.json

Outputs (in --output-dir, default data/calibration/):
    calibration.json      homography + keypoints + verification report
    reference_frame.jpg   the frame used for calibration
    overlay.jpg           detected keypoints and reprojected court lines
    top_down.jpg          the frame warped into the 2-D bird's-eye view
    debug_mask.png        white-line mask (auto mode only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from tennis_tracker.court import geometry as G
from tennis_tracker.court.calibration import CourtCalibration, require_calibration
from tennis_tracker.court.homography import CalibrationError, estimate_homography
from tennis_tracker.court.line_detector import DetectorConfig, detect_court
from tennis_tracker.court.visualize import (
    draw_court_overlay,
    draw_debug_lines,
    side_by_side,
    warp_top_down,
)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------- #
def load_reference_image(source: Path, frame_index: int = 0) -> tuple[np.ndarray, int | None]:
    """Read a still image, or extract one frame from a video."""
    if not source.exists():
        raise CalibrationError(f"Source not found: {source}")
    if source.suffix.lower() in IMAGE_EXT:
        img = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if img is None:
            raise CalibrationError(f"Could not decode image: {source}")
        return img, None
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise CalibrationError(f"Could not open video: {source}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index < 0:
        frame_index = max(0, n + frame_index)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        raise CalibrationError(f"Could not read frame {frame_index} of {source} ({n} frames)")
    return img, frame_index


def run_auto(image: np.ndarray, cfg: DetectorConfig, verbose: bool = True):
    res = detect_court(image, cfg)
    if verbose:
        n_meas = sum(1 for s in res.keypoint_source if s == "measured")
        print(f"  candidate lines: {len(res.lines_h)} horizontal, {len(res.lines_v)} vertical")
        print(f"  hypotheses evaluated: {res.n_hypotheses}, best score {res.best_hypothesis_score:.3f}")
        print(f"  matched model lines: {len(res.matched_lines)}/9, measured keypoints: {n_meas}/14")
        print(f"  homography RMSE {res.homography.rmse_px:.2f}px "
              f"(max {res.homography.max_err_px:.2f}px), inliers {res.homography.n_inliers}/{res.homography.n_points}")
    return res


def run_manual(image: np.ndarray, initial: np.ndarray | None = None):
    from tennis_tracker.court.manual import click_keypoints

    pts, src = click_keypoints(image, initial=initial)
    res = estimate_homography(pts, G.COURT_KEYPOINTS_M, ransac_thresh_px=6.0,
                              use_ransac=int(np.sum(np.array(src) == "manual")) >= 6)
    return pts, src, res


def save_outputs(out_dir: Path, image: np.ndarray, cal: CourtCalibration, det=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "reference_frame.jpg"), image)
    overlay = draw_court_overlay(image, cal.H, cal.keypoints_px, cal.keypoint_source)
    cv2.imwrite(str(out_dir / "overlay.jpg"), overlay)
    top = warp_top_down(image, cal.H, G.MinimapSpec(scale=30.0))
    cv2.imwrite(str(out_dir / "top_down.jpg"), top)
    cv2.imwrite(str(out_dir / "overlay_and_top_down.jpg"), side_by_side(overlay, top, target_h=720))
    if det is not None and det.white_mask is not None:
        cv2.imwrite(str(out_dir / "debug_mask.png"), det.white_mask)
        cv2.imwrite(str(out_dir / "debug_lines.jpg"),
                    draw_debug_lines(image, det.lines_h, det.lines_v, det.work_scale))


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, help="reference image or video (.mp4)")
    ap.add_argument("--frame", type=int, default=0, help="frame index when --source is a video (negative = from end)")
    ap.add_argument("--output-dir", type=Path, default=Path("data/calibration"))
    ap.add_argument("--manual", action="store_true", help="skip auto detection, click keypoints by hand")
    ap.add_argument("--review", action="store_true", help="show the automatic result and ask for confirmation")
    ap.add_argument("--no-fallback", action="store_true", help="do not open the manual window if auto detection fails")
    ap.add_argument("--no-court-mask", action="store_true", help="disable the court-colour mask in the detector")
    ap.add_argument("--work-width", type=int, default=1280, help="detector working resolution")
    ap.add_argument("--max-rmse", type=float, default=6.0, help="max allowed reprojection RMSE in px")
    ap.add_argument("--check", type=Path, help="verify an existing calibration.json and exit")
    ap.add_argument("--video-size", type=str, help="WxH of the video that will be processed (used with --check)")
    args = ap.parse_args(argv)

    # --- verify only ------------------------------------------------------
    if args.check:
        size = None
        if args.video_size:
            w, h = args.video_size.lower().split("x")
            size = (int(w), int(h))
        try:
            cal = require_calibration(args.check, image_size=size)
        except CalibrationError as e:
            print(f"RED LIGHT: {e}")
            return 2
        print(f"GREEN LIGHT: {args.check} is valid "
              f"(method={cal.method}, RMSE={cal.reproj_rmse_px:.2f}px, inliers={cal.n_inliers})")
        return 0

    if not args.source:
        ap.error("--source is required (or use --check)")

    # --- load the reference frame -----------------------------------------
    try:
        image, frame_index = load_reference_image(args.source, args.frame)
    except CalibrationError as e:
        print(f"ERROR: {e}")
        return 2
    h, w = image.shape[:2]
    print(f"Reference image: {args.source} ({w}x{h})"
          + (f", frame {frame_index}" if frame_index is not None else ""))

    cfg = DetectorConfig(work_width=args.work_width, use_court_mask=not args.no_court_mask)
    det = None
    kp = src = res = None
    method = "manual" if args.manual else "auto"

    # --- automatic detection ----------------------------------------------
    if not args.manual:
        print("Running automatic court detection ...")
        try:
            det = run_auto(image, cfg)
            kp, src, res = det.keypoints_px, det.keypoint_source, det.homography
            if args.review:
                from tennis_tracker.court.manual import review_overlay
                if not review_overlay(image, res.H, kp, src):
                    print("Automatic result rejected; switching to manual mode.")
                    method = "manual"
                    kp, src, res = run_manual(image, initial=det.keypoints_all_px)
                    det = None
        except CalibrationError as e:
            print(f"Automatic detection failed: {e}")
            if args.no_fallback or not sys.stdin.isatty():
                print("Manual fallback skipped (--no-fallback or non-interactive session). "
                      "Re-run with --manual in a terminal to click the keypoints.")
                return 2
            print("Falling back to manual calibration ...")
            method = "manual"
            try:
                kp, src, res = run_manual(image)
            except CalibrationError as e2:
                print(f"ERROR: {e2}")
                return 2
    else:
        try:
            kp, src, res = run_manual(image)
        except CalibrationError as e:
            print(f"ERROR: {e}")
            return 2

    # --- build, verify, save ----------------------------------------------
    cal = CourtCalibration(
        H=res.H, image_size=(w, h), keypoints_px=kp, keypoint_source=list(src),
        method=method, reproj_rmse_px=res.rmse_px, n_inliers=res.n_inliers,
        line_support=det.line_support if det is not None else None,
        source=str(args.source), frame_index=frame_index,
    )
    report = cal.verify(max_rmse_px=args.max_rmse)
    print("Verification:")
    print(report.format())

    out = args.output_dir
    cal_path = cal.save(out / "calibration.json")
    save_outputs(out, image, cal, det)
    print(f"Saved calibration to {cal_path}")
    print(f"Diagnostics: {out / 'overlay.jpg'}, {out / 'top_down.jpg'}")

    if not cal.is_ready:
        print("Calibration NOT accepted. Fix the reference frame or run with --manual.")
        return 1
    print("Calibration accepted. Video processing (Phase 2) may proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
