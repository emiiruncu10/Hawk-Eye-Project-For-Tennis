#!/usr/bin/env python3
"""Confidence-threshold sweep for the ball detector on a real video.

Runs TrackNet once with threshold 0 (or loads a raw-detections CSV), then
replays the detections through the tracker for every threshold and reports,
per threshold:

  visible     detections above the threshold (candidate positives)
  accepted    of those, how many the Kalman gate accepted (track-consistent)
  rejected    gate failures = likely false positives at that threshold
  predicted   frames interpolated by the filter (short gaps)
  lost        frames with no position (long gaps / no track)
  segments    number of separate track segments (fragmentation)
  jump_px     median displacement between consecutive accepted detections

Score = accepted - 2 * rejected - 0.5 * lost  (higher is better); ties are
broken by preferring the middle of the best plateau, which is the most
robust choice when the confidence distribution is bimodal.

Examples
--------
    python sweep_threshold.py --video ~/Desktop/new_test1.mp4
    python sweep_threshold.py --raw-csv data/output/raw_detections_v1.csv --min 0.3 --max 0.8 --step 0.05
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from tennis_tracker.inference.detector import BallDetection
from tennis_tracker.tracking.tracker import BallTracker, TrackerConfig, track_summary


def run_inference(video: Path, calibration: Path, weights: Path, arch: str, device: str | None,
                  batch_size: int, end: int | None) -> list[tuple[int, float, float, float]]:
    from tennis_tracker.inference.detector import BallDetector, DetectorConfig
    from tennis_tracker.inference.pipeline import PipelineConfig, TrackNetInputPipeline
    from tennis_tracker.inference.preprocess import TrackNetInputSpec
    from tennis_tracker.model import ARCHS, build_model, load_weights

    model = build_model(arch)
    rep = load_weights(model, weights)
    print("Weights:", rep.summary())
    spec = TrackNetInputSpec.from_name(ARCHS[arch]["preset"])
    pipe = TrackNetInputPipeline(video, calibration, PipelineConfig(spec=spec, batch_size=batch_size,
                                                                     device=device, end_frame=end))
    det = BallDetector(model, pipe.device, DetectorConfig(threshold=0.0,
                                                          output_frame_index=ARCHS[arch]["output_frame_index"]))
    rows = []
    for batch in pipe:
        for d in det.detect(batch):
            rows.append((d.frame_index, d.confidence, d.x, d.y))
    return rows


def load_raw(path: Path) -> list[tuple[int, float, float, float]]:
    with path.open() as f:
        return [(int(r["frame"]), float(r["confidence"]), float(r["x_px"]), float(r["y_px"]))
                for r in csv.DictReader(f)]


def evaluate(rows, threshold: float, tracker_cfg: TrackerConfig) -> dict:
    tr = BallTracker(tracker_cfg)
    visible = 0
    for frame, conf, x, y in rows:
        vis = conf >= threshold
        visible += int(vis)
        tr.step(BallDetection(frame, x if vis else None, y if vis else None, conf, vis))
    pts = tr.finish()
    s = track_summary(pts)
    acc = [p for p in pts if p.status == "detected"]
    jumps = [np.hypot(b.raw_x - a.raw_x, b.raw_y - a.raw_y) for a, b in zip(acc, acc[1:])
             if b.frame_index == a.frame_index + 1]
    return {"threshold": threshold, "visible": visible, "accepted": s["detected"], "rejected": tr.n_rejected,
            "predicted": s["predicted"], "lost": s["lost"], "segments": tr.n_segments,
            "jump_px": float(np.median(jumps)) if jumps else float("nan"),
            "score": s["detected"] - 2 * tr.n_rejected - 0.5 * s["lost"]}


def choose(results: list[dict]) -> dict:
    best = max(r["score"] for r in results)
    plateau = [r for r in results if abs(r["score"] - best) < 1e-9]
    return plateau[len(plateau) // 2]           # middle of the best plateau


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--calibration", type=Path, default=Path("data/calibration/calibration.json"))
    ap.add_argument("--weights", type=Path, default=Path("weights/model_best.pt"))
    ap.add_argument("--arch", default="v1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--raw-csv", type=Path, help="reuse raw detections instead of running the model")
    ap.add_argument("--save-raw", type=Path, default=Path("data/output/raw_detections.csv"))
    ap.add_argument("--min", type=float, default=0.3)
    ap.add_argument("--max", type=float, default=0.8)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--max-missing", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("data/output/threshold_sweep.csv"))
    args = ap.parse_args(argv)

    if args.raw_csv:
        rows = load_raw(args.raw_csv)
    elif args.video:
        rows = run_inference(args.video, args.calibration, args.weights, args.arch, args.device,
                             args.batch_size, args.end)
        args.save_raw.parent.mkdir(parents=True, exist_ok=True)
        with args.save_raw.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["frame", "confidence", "x_px", "y_px"]); w.writerows(rows)
        print(f"Raw detections saved to {args.save_raw}")
    else:
        ap.error("give --video or --raw-csv")

    conf = np.array([r[1] for r in rows])
    print(f"{len(rows)} frames; confidence percentiles 5/25/50/75/95 = "
          f"{np.round(np.percentile(conf, [5, 25, 50, 75, 95]), 3).tolist()}")
    print("histogram (0.1 bins):", np.histogram(conf, bins=np.arange(0, 1.01, 0.1))[0].tolist())

    thresholds = np.round(np.arange(args.min, args.max + 1e-9, args.step), 3)
    cfg = TrackerConfig(max_missing=args.max_missing)
    results = [evaluate(rows, float(t), cfg) for t in thresholds]
    cols = ["threshold", "visible", "accepted", "rejected", "predicted", "lost", "segments", "jump_px", "score"]
    print(" ".join(f"{c:>9}" for c in cols))
    for r in results:
        print(" ".join(f"{r[c]:>9.2f}" if isinstance(r[c], float) else f"{r[c]:>9}" for c in cols))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(results)
    best = choose(results)
    print(f"\nRecommended threshold: {best['threshold']:.2f}  "
          f"(accepted {best['accepted']}, rejected {best['rejected']}, lost {best['lost']}, "
          f"segments {best['segments']}); sweep table saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
