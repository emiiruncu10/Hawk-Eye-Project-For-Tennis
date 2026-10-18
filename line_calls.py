#!/usr/bin/env python3
"""Phase 4 - Bounce detection and line calls (IN / OUT / CLOSE).

Works on the outputs of ``track.py`` (no re-inference): reads the tracked
CSV, finds the bounces, judges each against the court lines and renders an
annotated video plus an events CSV.

Examples
--------
    python line_calls.py --video match.mp4 --calibration data/calibration/match/calibration.json
    python line_calls.py --video match.mp4 --calibration ... --game doubles
    python line_calls.py --video match.mp4 --calibration ... --no-video      # CSV only

Outputs (default ``data/output/``):
    <video>_calls.mp4   ball trail, bounce markers (green IN, red OUT, amber CLOSE),
                        call banner, minimap with all bounces
    <video>_calls.csv   one row per bounce: frame, court position, call, clearance,
                        nearest line, uncertainty
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tennis_tracker.court.calibration import require_calibration
from tennis_tracker.court.homography import CalibrationError
from tennis_tracker.inference.video import VideoError, probe_video
from tennis_tracker.tracking.bounce import BounceConfig, detect_bounces, find_duplicate_frames, read_track_csv
from tennis_tracker.tracking.call_annotate import CallAnnotatorConfig, render_line_calls_video
from tennis_tracker.tracking.line_call import CourtRules, call_bounces, export_calls_csv, summarize


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, default=Path("data/calibration/calibration.json"))
    ap.add_argument("--track-csv", type=Path, default=None, help="default data/output/<video>_tracked.csv")
    ap.add_argument("--game", default="singles", choices=["singles", "doubles"])
    ap.add_argument("--output", type=Path, default=None, help="default data/output/<video>_calls.mp4")
    ap.add_argument("--events-csv", type=Path, default=None, help="default next to --output")
    ap.add_argument("--no-video", action="store_true", help="write the events CSV only")
    ap.add_argument("--skip-duplicate-check", action="store_true",
                    help="do not scan the video for repeated frames (faster; only safe for native-rate footage)")
    ap.add_argument("--hold", type=float, default=1.2, help="seconds a call stays on screen")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    log = (lambda *a, **k: None) if args.quiet else print

    try:
        info = probe_video(args.video)
        cal = require_calibration(args.calibration, image_size=info.size)
    except CalibrationError as e:
        print(f"RED LIGHT - calibration gate failed:\n{e}"); return 1
    except VideoError as e:
        print(f"ERROR: {e}"); return 2
    track_csv = args.track_csv or Path("data/output") / f"{args.video.stem}_tracked.csv"
    if not track_csv.exists():
        print(f"ERROR: tracking CSV not found: {track_csv} (run track.py first)"); return 2

    t0 = time.perf_counter()
    points = read_track_csv(track_csv)
    dups = [] if args.skip_duplicate_check else find_duplicate_frames(args.video)
    bounces = detect_bounces(points, cal, info.fps, BounceConfig(), duplicate_frames=dups,
                             n_frames=max(info.frame_count, max(p.frame_index for p in points) + 1))
    rules = CourtRules(game=args.game)
    calls = call_bounces(bounces, rules)
    log(f"Video {info.width}x{info.height} @ {info.fps:.2f} fps, {info.frame_count} frames; "
        f"{len(dups)} repeated frames; {len(points)} tracked frames from {track_csv}")
    log(f"Detected {len(bounces)} bounces in {time.perf_counter() - t0:.1f}s; rules: {args.game}")
    log(f"{'#':>3} {'frame':>8} {'half':>5} {'court x,y (m)':>15} {'call':>6} {'clearance':>10} "
        f"{'+-band':>7}  nearest line")
    for i, c in enumerate(calls, 1):
        b = c.bounce
        log(f"{i:>3} {b.frame:8.1f} {b.half:>5} {b.court_x_m:7.2f},{b.court_y_m:6.2f} {c.call:>6} "
            f"{100 * c.clearance_m:8.1f}cm {100 * c.band_m:5.0f}cm  {c.nearest_line}")
    s = summarize(calls)
    log(f"Summary: IN {s['IN']}, OUT {s['OUT']}, CLOSE {s['CLOSE']}")

    out = args.output or Path("data/output") / f"{args.video.stem}_calls.mp4"
    events = args.events_csv or out.with_suffix(".csv")
    export_calls_csv(calls, events)
    log(f"Wrote {events}")
    if not args.no_video:
        t1 = time.perf_counter()
        n = render_line_calls_video(args.video, points, calls, out, cal, CallAnnotatorConfig(hold_s=args.hold))
        log(f"Wrote {out} ({n} frames, {time.perf_counter() - t1:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
