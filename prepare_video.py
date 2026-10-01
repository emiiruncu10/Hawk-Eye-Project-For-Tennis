#!/usr/bin/env python3
"""Phase 2 - Video -> 9-channel TrackNet input tensors (dry run / inspection).

The calibration gate runs first: if ``--calibration`` is missing or not
green, the script aborts *before* opening the video.

Examples
--------
    python prepare_video.py --video ~/Desktop/new_test1.mp4
    python prepare_video.py --video match.mp4 --preset yastrebksv --batch-size 4 --device mps
    python prepare_video.py --video match.mp4 --max-sequences 20 --save-sample data/output/sample_batch.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from tennis_tracker.court.homography import CalibrationError
from tennis_tracker.inference.pipeline import PipelineConfig, TrackNetInputPipeline
from tennis_tracker.inference.preprocess import TrackNetInputSpec
from tennis_tracker.inference.tensorize import check_environment
from tennis_tracker.inference.video import VideoError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True, help="input .mp4")
    ap.add_argument("--calibration", type=Path, default=Path("data/calibration/calibration.json"))
    ap.add_argument("--preset", default="tracknet_v2", help="tracknet_v2 (288x512 RGB) | yastrebksv (360x640 BGR)")
    ap.add_argument("--height", type=int, help="override model input height")
    ap.add_argument("--width", type=int, help="override model input width")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None, help="mps | cuda | cpu (default: auto)")
    ap.add_argument("--warmup", default="skip", choices=["skip", "repeat"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--max-sequences", type=int, default=None)
    ap.add_argument("--allow-rescale", action="store_true", help="adapt calibration to a resized video (same aspect)")
    ap.add_argument("--save-sample", type=Path, help="save the first batch as .npz (tensor + frame indices)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    env = check_environment(args.device)
    print("Environment: " + ", ".join(f"{k}={v}" for k, v in env.items()))
    if not env.get("device_ok"):
        print("ERROR: selected device is not usable"); return 2

    spec = TrackNetInputSpec.from_name(args.preset)
    if args.height or args.width:
        spec = TrackNetInputSpec(height=args.height or spec.height, width=args.width or spec.width,
                                 channel_order=spec.channel_order, frame_order=spec.frame_order)
    cfg = PipelineConfig(spec=spec, batch_size=args.batch_size, device=args.device, warmup=args.warmup,
                         start_frame=args.start, end_frame=args.end, max_sequences=args.max_sequences,
                         allow_rescale=args.allow_rescale)

    try:
        pipe = TrackNetInputPipeline(args.video, args.calibration, cfg)
    except CalibrationError as e:
        print(f"RED LIGHT - calibration gate failed, video not processed:\n{e}"); return 1
    except VideoError as e:
        print(f"ERROR: {e}"); return 2

    vi = pipe.video_info
    print(f"Calibration OK: {args.calibration} ({pipe.calibration.method}, RMSE {pipe.calibration.reproj_rmse_px:.2f}px)")
    print(f"Video: {vi.path} {vi.width}x{vi.height} @ {vi.fps:.2f} fps, {vi.frame_count} frames ({vi.fourcc})")
    print(f"Input spec: {spec.height}x{spec.width}, {spec.channel_order.upper()}, {spec.frame_order}, "
          f"{spec.n_channels} channels; batch {args.batch_size}; device {pipe.device}")
    print(f"Expected sequences: {pipe.expected_sequences}")

    first = None
    vmin, vmax, vsum, n_el = float("inf"), float("-inf"), 0.0, 0
    for i, batch in enumerate(pipe):
        t = batch.tensor
        vmin = min(vmin, float(t.min())); vmax = max(vmax, float(t.max()))
        vsum += float(t.sum()); n_el += t.numel()
        if first is None:
            first = batch
            print(f"First batch: shape {tuple(t.shape)}, dtype {t.dtype}, device {t.device}, "
                  f"frames t = {batch.frame_indices.tolist()}")
        if not args.quiet and (i + 1) % 10 == 0:
            print(f"  batch {i + 1}: sequences {pipe.stats.sequences}, {pipe.stats.fps:.1f} frames/s")

    st = pipe.stats
    print(f"Done: {st.frames_read} frames -> {st.sequences} sequences -> {st.batches} batches "
          f"in {st.seconds:.2f}s ({st.fps:.1f} frames/s)")
    if n_el:
        print(f"Tensor value range [{vmin:.3f}, {vmax:.3f}], mean {vsum / n_el:.3f}  (expected within [0, 1])")

    if args.save_sample and first is not None:
        args.save_sample.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_sample, tensor=first.tensor.cpu().numpy(),
                            frame_indices=first.frame_indices,
                            sequence_indices=np.array(first.sequence_indices),
                            original_size=np.array(first.original_size),
                            spec=np.array([spec.height, spec.width]), channel_order=spec.channel_order,
                            frame_order=spec.frame_order)
        print(f"Saved sample batch to {args.save_sample}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
