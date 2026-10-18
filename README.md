# TennisTracker

Single-camera tennis ball tracking for Apple Silicon: classical court calibration,
a 9-channel TrackNet input pipeline, TrackNet inference on the Metal (MPS) backend,
Kalman-filter trajectory smoothing, and annotated video / CSV export.

Everything runs locally on an M4 Pro (24 GB) with PyTorch `mps`. Broadcast footage
from four surfaces (Paris hard court, Australian Open, Roland Garros clay,
Wimbledon grass) and one amateur wide-angle clip have been used for validation.

---

## 1. Architecture

The design follows `TrackNet_Installation_and_Architecture_Flow.md`: calibration is
done **once per camera position**, then every frame of the video flows through the
tensor pipeline and the network.

```
                 Phase 1 (once)                    Phase 2 (per frame)                 Phase 3 (per frame)
 reference frame ──► white-line mask ──► Hough ──► ┐
                     court-colour + extent filters │
                     hypothesis search (vectorised)│         video ──► resize 360x640 ──► 3-frame buffer
                     line matching + RANSAC        │            (cv2.VideoCapture)        (t-2, t-1, t)
                     verification ("green light")  │                                          │
                                                   ▼                                          ▼
                              calibration.json (H: px -> m)  ──── gate ────►  9-channel float32 NCHW tensor on mps
                                                                                              │
                                                                                     TrackNet heat-map
                                                                                              │
                                                                          top-K peaks + confidence threshold
                                                                                              │
                                                                          Kalman tracker (gating, gaps, RTS)
                                                                                              │
                                                                         annotated MP4 (tail + minimap) + CSV
```

| Phase | Module(s) | Entry point |
|---|---|---|
| 1. Court detection & homography | `tennis_tracker/court/` (`geometry`, `line_detector`, `homography`, `calibration`, `manual`, `visualize`) | `calibrate.py` |
| 2. Video -> 9-channel tensors | `tennis_tracker/inference/` (`video`, `preprocess`, `sequence`, `tensorize`, `pipeline`) | `prepare_video.py` |
| 3. Inference, tracking, annotation | `tennis_tracker/model/` (`tracknet`, `weights`), `inference/detector`, `tennis_tracker/tracking/` (`kalman`, `tracker`, `annotate`) | `track.py`, `sweep_threshold.py` |
| 4. Bounce detection, line calls | `tennis_tracker/tracking/` (`bounce`, `line_call`, `call_annotate`) | `line_calls.py` |

### Phase 1 – Court detection and homography
1. Downscale the reference frame to 1280 px; build a white-line mask (luminance floor 120 plus
   Farin's line-structure constraint: a line pixel must be brighter than its neighbours `tau` px
   away on both sides, contrast >= 24).
2. Estimate the playing-surface colour (dominant hue of the frame centre); restrict the mask to
   the surface region and keep only lines that have surface colour on at least one side.
3. Probabilistic Hough -> segments -> merged infinite lines -> horizontal / vertical sets, then
   *mutual extent pruning* (horizontals must cross the span of the strongest verticals and vice
   versa) to drop crowd and grandstand edges.
4. Every pair of image lines is matched against every pair of court-model lines; each 4-point
   homography is scored on precision (model lines on white pixels) **and** recall (white pixels
   explained by the model) – tens of thousands of hypotheses, vectorised NumPy, a few seconds.
5. All 9 painted lines are matched, the 14 keypoints are recomputed as exact intersections and
   `cv2.findHomography(RANSAC)` gives the final pixel -> metre matrix.
6. Verification: matrix validity and conditioning, reprojection RMSE (<= 6 px), inliers, at least
   6 measured keypoints, orientation (far baseline above, not mirrored), convexity, court area,
   visibility, near baseline >= 90 % of the far one, line support >= 60 %. Only a fully green
   report is written with `status: ok`, and `require_calibration()` re-runs the checks before any
   video is opened.

Court model: ITF dimensions (23.77 m x 10.97 m, singles 8.23 m, service lines 6.40 m from the
net). World frame in metres, origin at the far-left doubles corner, `x` across the court, `y`
towards the camera. The 14-keypoint order matches `yastrebksv/TennisCourtDetector`, so a trained
keypoint network can replace the classical detector without touching the homography code.

### Phase 2 – Tensor pipeline
`TrackNetInputPipeline` calls `require_calibration()` **before** `cv2.VideoCapture`, checks the
video resolution against the calibration (`--allow-rescale` adapts a same-aspect resize), then
streams frames: `cv2.resize` to the model size, optional BGR->RGB, HWC->CHW, `/255` -> float32,
a rolling `(t-2, t-1, t)` buffer, channel concatenation to `(N, 9, H, W)` and transfer to `mps`.

| preset | resolution | colour | channel layout | used by |
|---|---|---|---|---|
| `yastrebksv` (default for `--arch v1`) | 360 x 640 | BGR | `[t, t-1, t-2]` | `weights/model_best.pt` |
| `tracknet_v2` | 288 x 512 | RGB | `[t-2, t-1, t]` | TrackNetV2-style checkpoints |

### Phase 3 – Inference, tracking, annotation
* `BallDetector`: heat-maps on the accelerator, up to `--max-peaks` local maxima per frame
  (non-maximum suppression, peaks within 6 model px of the frame edge ignored), log-parabolic
  sub-pixel refinement, confidence threshold, scale back to video pixels.
* `BallTracker`: constant-acceleration Kalman filter (state x, y, vx, vy, ax, ay) with
  chi-square gating; among the candidate peaks it associates the one closest to its prediction,
  coasts through gaps up to `--max-missing` frames (`predicted`), marks longer gaps `lost`,
  rejects single outliers, restarts after persistent jumps, and applies Rauch-Tung-Striebel
  smoothing per segment. Optional plausible-region mask from the calibration
  (court + 5 m sideways + 8 m behind the baselines + lob band above the far court).
* `TrajectoryAnnotator`: `cv2.circle` for the ball, fading `cv2.line` tail, calibrated top-down
  minimap with the ball's ground projection, HUD with court coordinates in metres,
  H.264 (`avc1`) writer with `mp4v` fallback.

---

### Phase 4 – Bounce detection and line calls
In/out is decided at the bounce, the only moment the ball is on the ground and its image
position maps exactly onto the court through the ground-plane homography (in flight the
mapping is wrong, which is why per-frame court coordinates are only meaningful at bounces).
`line_calls.py` works on `track.py`'s CSV, so no re-inference is needed (a few seconds per clip).

1. **Repeated frames** (25 fps material in 30 fps files) are found from the video pixels and
   dropped; the remaining frames are re-timed to their real capture times.
2. **Bounce detection** – around every measurement the raw track is fitted with a continuous
   piecewise-quadratic model whose velocity may jump at a knot (sub-frame search). A bounce
   is an upward jump of the vertical image velocity (>= 3.5 m/s at the local image scale,
   statistically significant) with an almost unchanged horizontal velocity, located on or
   around the court, and consistent with the half it lands in: far half – the ball moves up
   right after it and is not falling fast before it (serve contact); near half – it falls
   right before it and its upward speed afterwards stays below the ball's restitution (~0.8)
   times the downward speed before it (a racket hit adds energy). Each event is judged by
   its strongest knot, and a second event on the same half within 1 s (a racket hit or a
   second bounce) is dropped.
3. **Localisation** – the knot and position are refined with independent curvature on each
   side when enough points exist; the ball centre is shifted down by its radius to the
   contact point, mapped to metres, and its covariance (fit + tracking noise + calibration
   error) is propagated through the homography Jacobian.
4. **Line call** – a ball touching a line is in. The calibration aligns the model with the
   *centres* of the painted lines, so the boundary is half a line width outside them
   (baselines 10 cm, other lines 5 cm), plus a 2 cm contact-patch half-width. The signed
   clearance to the nearest boundary gives **IN** / **OUT**, or **CLOSE** when it is within
   `max(2 sigma, 3 cm)` of the line, sigma being the position uncertainty perpendicular to
   that line. Default court: singles (`--game doubles` available). Service boxes are not
   applied to serves yet.

Validation:

| set | bounces | found | false positives |
|---|---|---|---|
| simulation (3-D flights, pinhole camera, 0.7 px noise, 120 runs) | 120 | 120 | 0 |
| 8 real clips, every detection checked frame by frame | 69 | 69 | 0 accepted |

Simulated accuracy (1 sigma): near half 0.8 cm lateral / 4.7 cm depth, far half 1.3 cm /
13 cm; the reported uncertainty matches these within ~20 %. Known limitations: recall is
lower on small far courts in wide-angle amateur footage; a slow volley (no bounce before it)
can pass as a bounce; a net-cord ball that drops on the hitter's side is judged by geometry
only; on very low cameras far-half positions are extrapolated (see test7) and stay CLOSE.

## 2. Environment and hardware

* Apple Silicon M4 Pro, 24 GB unified memory, macOS. Inference uses PyTorch's Metal backend
  (`torch.device("mps")`, selected automatically by `tennis_tracker/device.py`; CUDA / CPU
  fallback).
* Tensors are always float32 (MPS has no float64). `numpy_to_tensor()` falls back to
  `torch.frombuffer` if `torch.from_numpy` is unavailable (torch built against NumPy 1.x under
  NumPy 2.x); `check_environment()` reports the active path.
* Run **one** inference process at a time: two concurrent MPS jobs stall each other badly.

```bash
cd ~/Desktop/TennisTracker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # torch >= 2.4, numpy >= 1.26 (<3), opencv-python, scipy, pytest ...
python -c "import torch; print('mps:', torch.backends.mps.is_available())"
```

Validated locally with Python 3.11, PyTorch 2.3.0, NumPy 2.4.3, OpenCV 4.13.

---

## 3. Model weights and provenance

| file | source | architecture | status |
|---|---|---|---|
| `weights/model_best.pt` (42.9 MB) | yastrebksv/TrackNet README -> Google Drive id `1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl` | `--arch v1`: `tracknet_v1()`, 18 conv blocks, 10.7 M params, 256-way softmax per pixel, same layer names as `BallTrackerNet` | **verified**: loads `strict=True`, 126/126 tensors; default for `track.py` |
| `weights/tracknetv2_track.pt` (45.4 MB) | ChgygLin/TrackNetV2-pytorch `tf2torch/track.pt` | `--arch v2` layout (keys renamed on load) | **not usable**: BatchNorm statistics are per image column (Keras model normalised over the width axis) and it is a badminton model; kept for reference |

SHA-256 checksums are in `weights/SHA256SUMS`; `*.pt` files are git-ignored, re-download per
`weights/README.md`. `load_weights()` uses `torch.load(weights_only=True)`, unwraps
`state_dict` containers, strips `module.` prefixes, renames known third-party layouts and falls
back to shape-ordered matching, reporting exactly what was loaded.

**Input spec for `model_best.pt`**: 360 x 640, BGR, frames ordered t, t-1, t-2, values in [0, 1],
tensor `(N, 9, 360, 640)`; the heat-map for frame t is output channel 0 (expected grey level of
the 256-way classification). **Recommended `--threshold 0.5`**: on the validation clips the raw
confidences are bimodal (most frames > 0.8, occlusions < 0.1) and the sweep is flat from 0.15 to
0.70; 0.5 is the middle of that plateau. Throughput on the M4 Pro: ~5.5 frames/s at batch 8.

---

## 4. Usage

### 4.1 Calibrate a new camera position (`calibrate.py`)

```bash
# from a video (frame 0) – automatic detection, writes data/calibration/<name>/
python calibrate.py --source match.mp4 --output-dir data/calibration/match

# choose another frame, show the result and confirm it (y / n)
python calibrate.py --source match.mp4 --frame 120 --review --output-dir data/calibration/match

# click the 14 keypoints yourself (s = skip, u = undo, Enter = finish)
python calibrate.py --source match.mp4 --manual --output-dir data/calibration/match

# non-interactive (CI / scripts): never open a window
python calibrate.py --source match.mp4 --no-fallback

# is the green light on, and does it match the video resolution?
python calibrate.py --check data/calibration/match/calibration.json --video-size 1920x1080
```

Options: `--source` image or video, `--frame N` (negative = from the end), `--output-dir`,
`--manual`, `--review`, `--no-fallback`, `--no-court-mask`, `--work-width` (default 1280),
`--max-rmse` (default 6 px), `--check`, `--video-size`. Exit codes: `0` green, `1` rejected,
`2` input error / abort.

Pick a reference frame in which the whole court is visible and unoccluded. Wide-angle amateur
cameras can tilt the sidelines below 45 degrees; the line-angle threshold is a
`DetectorConfig` field (`horizontal_max_angle_deg`) reachable from Python.

### 4.2 End-to-end tracking (`track.py`)

```bash
# defaults: --arch v1, weights/model_best.pt, threshold 0.5, ROI on, smoothing on
python track.py --video match.mp4 --calibration data/calibration/match/calibration.json

# explicit weights / threshold, court overlay, custom outputs
python track.py --video match.mp4 --calibration data/calibration/match/calibration.json \
                --weights weights/model_best.pt --threshold 0.5 --draw-court \
                --output data/output/match_tracked.mp4 --csv data/output/match_tracked.csv

# wide-angle / unusual camera: disable the plausible-region mask
python track.py --video amateur.mp4 --calibration data/calibration/amateur/calibration.json --no-roi

# pipeline smoke test without a checkpoint (detections are random)
python track.py --video match.mp4 --allow-random-weights --max-frames 40
```

| option | default | meaning |
|---|---|---|
| `--video`, `--calibration` | – / `data/calibration/calibration.json` | inputs; the calibration gate runs before the video is opened |
| `--weights`, `--arch` | `weights/model_best.pt`, `v1` | checkpoint and network layout (`v1` yastrebksv, `v2` TrackNetV2) |
| `--threshold` | 0.5 | heat-map confidence below which the ball is occluded |
| `--max-peaks` | 3 | candidate peaks per frame for the tracker to choose from |
| `--no-roi` | off | do not drop candidates outside the calibrated plausible region |
| `--max-missing` | 8 | frames interpolated by the Kalman filter before `lost` |
| `--no-smooth`, `--no-subpixel` | off | disable RTS smoothing / sub-pixel refinement |
| `--start`, `--end`, `--max-frames` | whole video | frame range |
| `--allow-rescale` | off | adapt the calibration to a same-aspect resized video |
| `--output`, `--csv` | `data/output/<video>_tracked.*` | output paths |
| `--tail`, `--no-minimap`, `--draw-court` | 12, off, off | annotation |
| `--device`, `--batch-size`, `--preset` | auto, 8, by arch | execution |
| `--allow-random-weights`, `--allow-truncate`, `--unsafe-pickle` | off | testing / legacy checkpoints |

The run prints the weight-loading report, video and model info, then a summary such as
`tracked: 519 detected, 71 predicted, 164 lost; segments 19; gate rejections 41;
ROI-suppressed candidates 0`. Exit codes: `0` ok, `1` calibration gate failed (video never
opened), `2` bad input / device / weights.

### 4.3 Threshold sweep (`sweep_threshold.py`)

```bash
python sweep_threshold.py --video match.mp4 --calibration data/calibration/match/calibration.json
python sweep_threshold.py --raw-csv data/output/raw_detections.csv --min 0.3 --max 0.8 --step 0.05
```

Runs the model once at threshold 0 (or reuses `--raw-csv`), replays the detections through the
tracker for every threshold and prints visible / accepted / rejected / predicted / lost /
segments / median jump per threshold, with a recommendation (middle of the best plateau).
Writes `data/output/threshold_sweep.csv` and the raw detections.

### 4.4 Tensor pipeline only (`prepare_video.py`)

```bash
python prepare_video.py --video match.mp4 --calibration data/calibration/match/calibration.json \
                        --preset yastrebksv --batch-size 8 --save-sample data/output/sample_batch.npz
```

Reports tensor shape, dtype, device, value range and throughput without running the network.

### 4.5 Line calls (`line_calls.py`)

```bash
# after track.py: bounces, IN / OUT / CLOSE, annotated video + events CSV
python line_calls.py --video match.mp4 --calibration data/calibration/match/calibration.json

python line_calls.py --video match.mp4 --calibration ... --game doubles
python line_calls.py --video match.mp4 --calibration ... --no-video        # CSV only
```

Options: `--track-csv` (default `data/output/<video>_tracked.csv`), `--game singles|doubles`,
`--output`, `--events-csv`, `--no-video`, `--hold` (seconds a call stays on screen),
`--skip-duplicate-check` (native-rate footage only). The calibration gate runs first.
Exit codes: `0` ok, `1` calibration gate failed, `2` missing input.

### 4.6 From Python

```python
from tennis_tracker.court import require_calibration
from tennis_tracker.inference import iter_tracknet_batches, TrackNetInputSpec
from tennis_tracker.inference.detector import BallDetector, DetectorConfig
from tennis_tracker.model import build_model, load_weights
from tennis_tracker.tracking import BallTracker, TrackerConfig

cal = require_calibration("data/calibration/match/calibration.json", image_size=(1920, 1080))
model = build_model("v1"); load_weights(model, "weights/model_best.pt")
det = BallDetector(model, "mps", DetectorConfig(threshold=0.5, output_frame_index=0))
tracker = BallTracker(TrackerConfig(roi_mask=cal.plausible_region_mask()))
for batch in iter_tracknet_batches("match.mp4", "data/calibration/match/calibration.json",
                                   spec=TrackNetInputSpec.tracknet_yastrebksv(), batch_size=8, device="mps"):
    for d in det.detect(batch):
        tracker.step(d)
points = tracker.finish()                       # TrackPoint(frame_index, x, y, status, ...)
court_xy_m = cal.pixel_to_court([[p.x, p.y] for p in points if p.has_position])
```

---

## 5. Outputs

`data/calibration/<name>/` (Phase 1)

| file | content |
|---|---|
| `calibration.json` | `homography_px_to_m`, `homography_m_to_px`, 14 `keypoints_px` (null = not measured), keypoint sources, court dimensions, verification `checks`, `status: ok / failed` |
| `overlay.jpg`, `overlay_and_top_down.jpg` | keypoints and reprojected court lines on the reference frame; bird's-eye warp beside it |
| `top_down.jpg` | the frame warped into the 2-D court plane |
| `reference_frame.jpg`, `debug_mask.png`, `debug_lines.jpg` | input frame, white-line mask, candidate lines |

`data/output/` (Phase 3)

| file | content |
|---|---|
| `<video>_tracked.mp4` | H.264 video: ball marker (green detected, yellow predicted), fading trail, calibrated minimap top-right with the ball's ground projection, HUD `frame / status / confidence / court (x m, y m)`, optional court overlay |
| `<video>_tracked.csv` | one row per tracked frame: `frame, status, x_px, y_px, confidence, raw_x_px, raw_y_px, court_x_m, court_y_m, segment` |
| `raw_detections*.csv`, `threshold_sweep*.csv` | sweep-tool evidence |
| `<video>_calls.mp4` | ball trail, a ground marker at every bounce (green IN, red OUT, amber CLOSE) with its distance to the line, a call banner, and a minimap collecting all bounces |
| `<video>_calls.csv` | one row per bounce: `bounce, frame, time_s, half, court_x_m, court_y_m, call, leaning, clearance_cm, nearest_line, sigma_cm, close_band_cm, game, x_px, y_px, jump_vy_mps, tstat` |

Court coordinates are ground-plane projections through the homography: a ball in flight maps
beyond its true ground position (often past the far baseline). That is the intended input for a
later bounce detector, not a 3-D position.

Validation results with `model_best.pt`, threshold 0.5:

| clip | surface | calibration | detected / predicted / lost |
|---|---|---|---|
| ATP Paris, 125 f @ 30 fps | green hard | 14/14 kp, RMSE 1.3 px | 110 / 11 / 2 |
| Australian Open, 1650 f @ 50 fps | blue hard | 14/14, 1.2 px | 1477 / 125 / 46 |
| Roland Garros, 270 f @ 25 fps | clay | 10/14, 1.8 px | 227 / 20 / 21 |
| Wimbledon, 175 f @ 25 fps | grass | 14/14, 2.2 px | 124 / 19 / 30 |
| amateur indoor, 756 f @ 30 fps, wide-angle | hard | 12/14, 0.5 px (frame 411, angle 25, `--no-roi`) | 519 / 71 / 164 |

---

## 6. Testing and profiling

```bash
pytest -q          # 115 tests, ~20 s on CPU (MPS cases run when available)
```

The suite covers geometry, homography estimation and every verification check, the classical
detector on rendered synthetic courts, checkpoint loading (exact, wrapped, renamed, truncated),
heat-map peak extraction, the Kalman filter and tracker (gaps, outliers, re-initialisation,
candidate association, ROI), annotation and video writing, the CLI end to end, an "oracle"
test that recovers a known synthetic trajectory through the whole chain without any weights,
and bounce detection / line calls against a 3-D ball-flight and pinhole-camera simulator
(near and far bounces, racket hits that must not count, 25-in-30 fps repeated frames, line rules).

Profiling with `python -m cProfile -o run.prof track.py ...` on a 756-frame 1080p clip
(153 s wall clock): project Python code accounts for about 2 % of the time; 98 % is spent in
C extensions – torch MPS kernels and the GPU synchronisation inside `BallDetector.detect`
(~88 % inclusive, the network forward itself ~46 %), then OpenCV decode / resize / encode
(`pipeline` 6 %, `annotate` 5 %, `tensorize` 3 %, `video` 3 %). The tracker, Kalman filter and
calibration code together take well under a second. Calibration itself runs in under a second
per frame; its hypothesis search is the only Python-heavy stage (~0.4 s).

Speed levers, in order: half-precision inference on MPS, a smaller batch pipeline that overlaps
decode with inference, and skipping the annotated render when only the CSV is needed.

---

## 7. Layout

```
TennisTracker/
├── calibrate.py            Phase 1 CLI
├── prepare_video.py        Phase 2 CLI (tensor pipeline only)
├── track.py                Phase 3 CLI (end to end)
├── sweep_threshold.py      confidence-threshold sweep
├── line_calls.py           Phase 4 CLI (bounces, IN / OUT / CLOSE)
├── tennis_tracker/
│   ├── court/              geometry, line_detector, homography, calibration, manual, visualize
│   ├── inference/          video, preprocess, sequence, tensorize, pipeline, detector
│   ├── model/              tracknet (v1 / v2), weights
│   ├── tracking/           kalman, tracker, annotate, bounce, line_call, call_annotate
│   └── device.py
├── tests/                  115 pytest cases (+ synthetic court renderer, 3-D flight simulator)
├── weights/                model_best.pt, README.md, SHA256SUMS   (*.pt git-ignored)
├── data/calibration/       one folder per camera position
├── data/output/            tracked / calls videos, CSVs, sweep tables (git-ignored)
└── requirements.txt, pytest.ini, .gitignore
```

## References

* yastrebksv/TrackNet – PyTorch TrackNet for tennis (weights used here)
* ChgygLin/TrackNetV2-pytorch – 3-in / 3-out TrackNetV2 architecture
* nickluo/TrackNetV3 – background estimation module (future work)
* MichlF/sports_object_detection – YOLO + TrackNet + homography minimap
