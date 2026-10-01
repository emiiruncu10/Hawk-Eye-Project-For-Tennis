#!/usr/bin/env python3
"""Phase 3 - End-to-end tennis ball tracking.

    video -> calibration gate -> 9-channel batches -> TrackNet (mps) -> argmax +
    confidence gate -> Kalman tracking / RTS smoothing -> annotated MP4 (+ CSV)

Examples
--------
    # default: yastrebksv/TrackNet checkpoint weights/model_best.pt (360x640, BGR, 256-class head)
    python track.py --video match.mp4 --threshold 0.5

    # TrackNetV2-style network with another checkpoint
    python track.py --video match.mp4 --weights weights/tracknet_v2.pt --arch v2

    # pipeline smoke test without a checkpoint (random weights => meaningless detections)
    python track.py --video ~/Desktop/new_test1.mp4 --allow-random-weights --max-frames 40
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tennis_tracker.court.homography import CalibrationError
from tennis_tracker.inference.detector import BallDetector, DetectorConfig
from tennis_tracker.inference.pipeline import PipelineConfig, TrackNetInputPipeline
from tennis_tracker.inference.preprocess import TrackNetInputSpec
from tennis_tracker.inference.tensorize import check_environment
from tennis_tracker.inference.video import VideoError
from tennis_tracker.model import ARCHS, build_model, load_weights
from tennis_tracker.tracking.annotate import AnnotatorConfig, export_csv, render_annotated_video
from tennis_tracker.tracking.tracker import BallTracker, TrackerConfig, track_summary

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "model_best.pt"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, default=Path("data/calibration/calibration.json"))
    ap.add_argument("--weights", type=Path, default=None,
                    help="TrackNet checkpoint (.pt/.pth state dict); default weights/model_best.pt if it exists")
    ap.add_argument("--arch", default="v1", choices=sorted(ARCHS), help="v1 = yastrebksv/TrackNet (default), v2 = TrackNetV2")
    ap.add_argument("--allow-truncate", action="store_true", help="accept oversized 1-D checkpoint tensors (truncated)")
    ap.add_argument("--preset", default=None, help="input preset override (default: chosen by --arch)")
    ap.add_argument("--allow-random-weights", action="store_true", help="run without a checkpoint (pipeline test only)")
    ap.add_argument("--unsafe-pickle", action="store_true", help="allow full unpickling of a trusted legacy checkpoint")
    ap.add_argument("--device", default=None, help="mps | cuda | cpu (default auto)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5, help="heat-map confidence threshold")
    ap.add_argument("--no-subpixel", action="store_true")
    ap.add_argument("--max-peaks", type=int, default=3, help="candidate heat-map peaks per frame")
    ap.add_argument("--no-roi", action="store_true", help="do not restrict candidates to the calibrated court region")
    ap.add_argument("--max-missing", type=int, default=8, help="frames to interpolate before 'lost'")
    ap.add_argument("--no-smooth", action="store_true", help="disable RTS smoothing")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--allow-rescale", action="store_true")
    ap.add_argument("--output", type=Path, default=None, help="annotated MP4 (default data/output/<video>_tracked.mp4)")
    ap.add_argument("--csv", type=Path, default=None, help="trajectory CSV (default next to --output)")
    ap.add_argument("--tail", type=int, default=12)
    ap.add_argument("--no-minimap", action="store_true")
    ap.add_argument("--draw-court", action="store_true", help="overlay the reprojected court lines")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else print

    env = check_environment(args.device)
    log("Environment: " + ", ".join(f"{k}={v}" for k, v in env.items()))
    if not env.get("device_ok"):
        print("ERROR: selected device is not usable"); return 2

    # --- model ------------------------------------------------------------
    arch = ARCHS[args.arch]
    model = build_model(args.arch)
    if args.weights is None and not args.allow_random_weights and DEFAULT_WEIGHTS.exists():
        args.weights = DEFAULT_WEIGHTS
    if args.weights:
        try:
            rep = load_weights(model, args.weights, strict=False, unsafe_pickle=args.unsafe_pickle,
                               allow_truncate=args.allow_truncate)
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            print(f"ERROR loading weights: {e}"); return 2
        log(f"Weights: {rep.summary()}")
        if not rep.complete:
            print(f"WARNING: {len(rep.missing_keys)} model tensors were not found in the checkpoint")
    elif args.allow_random_weights:
        print("WARNING: no --weights given; running with RANDOM weights. Detections are meaningless "
              "(pipeline test only).")
    else:
        print("ERROR: --weights is required (or --allow-random-weights for a pipeline test)"); return 2

    spec = TrackNetInputSpec.from_name(args.preset or arch["preset"])
    if args.end is None and args.max_frames is not None:
        args.end = args.start + args.max_frames
    pcfg = PipelineConfig(spec=spec, batch_size=args.batch_size, device=args.device,
                          start_frame=args.start, end_frame=args.end, allow_rescale=args.allow_rescale)

    # --- calibration gate + video -----------------------------------------
    try:
        pipe = TrackNetInputPipeline(args.video, args.calibration, pcfg)
    except CalibrationError as e:
        print(f"RED LIGHT - calibration gate failed, video not processed:\n{e}"); return 1
    except VideoError as e:
        print(f"ERROR: {e}"); return 2
    vi = pipe.video_info
    log(f"Calibration OK ({pipe.calibration.method}, RMSE {pipe.calibration.reproj_rmse_px:.2f}px)")
    log(f"Video: {vi.path} {vi.width}x{vi.height} @ {vi.fps:.2f} fps, {vi.frame_count} frames")
    log(f"Model: TrackNet {args.arch} ({model.n_parameters / 1e6:.1f} M params) on {pipe.device}; "
        f"input {spec.height}x{spec.width} {spec.channel_order.upper()} {spec.frame_order}; "
        f"threshold {args.threshold}")

    # --- inference + tracking ---------------------------------------------
    detector = BallDetector(model, pipe.device, DetectorConfig(
        threshold=args.threshold, subpixel=not args.no_subpixel,
        output_frame_index=arch["output_frame_index"], max_peaks=args.max_peaks))
    roi = None if args.no_roi else pipe.calibration.plausible_region_mask()
    tracker = BallTracker(TrackerConfig(max_missing=args.max_missing, smooth=not args.no_smooth, roi_mask=roi))
    t0 = time.perf_counter()
    n_det = 0
    expected = pipe.expected_sequences
    for bi, batch in enumerate(pipe):
        for det in detector.detect(batch):
            n_det += int(det.visible)
            tracker.step(det)
        if not args.quiet and (bi + 1) % 10 == 0:
            done = pipe.stats.sequences
            print(f"  inference {done}/{expected} frames, {done / (time.perf_counter() - t0):.1f} fps")
    points = tracker.finish()
    t_inf = time.perf_counter() - t0
    summ = track_summary(points)
    log(f"Inference done: {summ['frames']} frames in {t_inf:.1f}s ({summ['frames'] / max(t_inf, 1e-9):.1f} fps); "
        f"raw detections {n_det}; tracked: {summ['detected']} detected, {summ['predicted']} predicted, "
        f"{summ['lost']} lost; segments {tracker.n_segments}; gate rejections {tracker.n_rejected}; "
        f"ROI-suppressed candidates {tracker.n_roi_suppressed}")

    # --- outputs ----------------------------------------------------------
    out = args.output or Path("data/output") / f"{args.video.stem}_tracked.mp4"
    csv_path = args.csv or out.with_suffix(".csv")
    acfg = AnnotatorConfig(tail_length=args.tail, draw_minimap=not args.no_minimap, draw_court_lines=args.draw_court)
    t1 = time.perf_counter()
    n_frames = render_annotated_video(args.video, points, out, pipe.calibration, acfg, args.start, args.end)
    export_csv(points, csv_path, pipe.calibration)
    log(f"Wrote {out} ({n_frames} frames, {time.perf_counter() - t1:.1f}s) and {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
