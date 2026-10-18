"""Render line calls onto the video: ground marker, label, banner and minimap."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ..court import geometry as G
from ..court.calibration import CourtCalibration
from ..court.visualize import draw_minimap
from ..inference.video import VideoReader
from .annotate import AnnotatorConfig, TrajectoryAnnotator, VideoWriterMP4
from .line_call import CLOSE, IN, OUT, LineCall
from .tracker import TrackPoint

CALL_COLORS = {IN: (60, 200, 60), OUT: (40, 40, 230), CLOSE: (0, 190, 255)}   # BGR


@dataclass
class CallAnnotatorConfig:
    hold_s: float = 1.2               # how long a call stays on screen
    marker_radius_m: float = 0.15     # ground marker radius (drawn larger than the ball for visibility)
    show_banner: bool = True
    show_minimap: bool = True
    minimap_scale: float = 6.0        # px per metre
    trail: AnnotatorConfig = field(default_factory=lambda: AnnotatorConfig(draw_minimap=False))


def _ground_polygon(cal: CourtCalibration, x_m: float, y_m: float, r_m: float, n: int = 36) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([x_m + r_m * np.cos(a), y_m + r_m * np.sin(a)], axis=1)
    return cal.court_to_pixel(pts)


def _label(c: LineCall, show_in_distance_below_m: float = 0.5) -> str:
    cm = abs(c.clearance_m) * 100
    if c.call == CLOSE:
        return f"CLOSE {cm:.0f}+-{c.band_m * 100:.0f} cm"
    if c.call == IN and c.clearance_m > show_in_distance_below_m:
        return "IN"                      # a clear IN needs no number
    return f"{c.call} {cm:.0f} cm"


def _text(img, text, org, scale, color, thick):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_call(frame: np.ndarray, c: LineCall, cal: CourtCalibration, cfg: CallAnnotatorConfig,
              fade: float = 1.0) -> None:
    """Ground marker and label for one call (in place)."""
    color = CALL_COLORS[c.call]
    s = frame.shape[0] / 1080.0
    poly = _ground_polygon(cal, c.bounce.court_x_m, c.bounce.court_y_m, cfg.marker_radius_m)
    if np.all(np.isfinite(poly)):
        p = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [p], color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.45 * fade, frame, 1 - 0.45 * fade, 0, dst=frame)
        cv2.polylines(frame, [p], True, color, max(1, int(round(2 * s))), cv2.LINE_AA)
        top = p[:, 0, :][np.argmin(p[:, 0, 1])]
        _text(frame, _label(c), (int(top[0]) - int(60 * s), int(top[1]) - int(12 * s)),
              0.8 * s, color, max(1, int(round(2 * s))))


def draw_banner(frame: np.ndarray, c: LineCall) -> None:
    color = CALL_COLORS[c.call]
    h, w = frame.shape[:2]
    s = h / 1080.0
    title = c.call
    sub = f"{c.nearest_line}: {abs(c.clearance_m) * 100:.0f} cm {'inside' if c.clearance_m >= 0 else 'outside'}"
    if c.call == CLOSE:
        sub += f" (+-{c.band_m * 100:.0f} cm, leans {c.leaning})"
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.8 * s, max(2, int(4 * s)))
    (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, max(1, int(2 * s)))
    bw = max(tw, sw) + int(40 * s)
    x0, y0 = (w - bw) // 2, int(20 * s)
    y1 = y0 + th + sh + int(45 * s)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + bw, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, dst=frame)
    cv2.rectangle(frame, (x0, y0), (x0 + bw, y1), color, max(2, int(3 * s)))
    cv2.putText(frame, title, ((w - tw) // 2, y0 + th + int(15 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                1.8 * s, color, max(2, int(4 * s)), cv2.LINE_AA)
    cv2.putText(frame, sub, ((w - sw) // 2, y1 - int(14 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                0.7 * s, (235, 235, 235), max(1, int(2 * s)), cv2.LINE_AA)


def draw_call_minimap(frame: np.ndarray, calls_so_far: Sequence[LineCall], trail: Sequence[TrackPoint],
                      cal: CourtCalibration, cfg: CallAnnotatorConfig, margin: int = 12) -> None:
    spec = G.MinimapSpec(scale=cfg.minimap_scale, margin_m=1.5)
    mm = draw_minimap(spec)
    pts = [p for p in trail[-60:] if p.has_position]
    if pts:
        court = cal.pixel_to_court(np.array([[p.x, p.y] for p in pts]))
        mpx = spec.world_to_minimap(court)
        for i in range(1, len(mpx)):
            if np.all(np.isfinite(mpx[i - 1:i + 1])):
                cv2.line(mm, tuple(np.round(mpx[i - 1]).astype(int)), tuple(np.round(mpx[i]).astype(int)),
                         (200, 200, 200), 1, cv2.LINE_AA)
    for k, c in enumerate(calls_so_far):
        q = spec.world_to_minimap([[c.bounce.court_x_m, c.bounce.court_y_m]])[0]
        r = 5 if k == len(calls_so_far) - 1 else 3
        cv2.circle(mm, (int(round(q[0])), int(round(q[1]))), r, CALL_COLORS[c.call], -1, cv2.LINE_AA)
        cv2.circle(mm, (int(round(q[0])), int(round(q[1]))), r, (0, 0, 0), 1, cv2.LINE_AA)
    h, w = mm.shape[:2]
    H, W = frame.shape[:2]
    if h + margin > H or w + margin > W:
        return
    y0, x0 = margin, W - w - margin
    roi = frame[y0:y0 + h, x0:x0 + w]
    cv2.addWeighted(mm, 0.85, roi, 0.15, 0, dst=roi)
    cv2.rectangle(frame, (x0 - 1, y0 - 1), (x0 + w, y0 + h), (255, 255, 255), 1)


def render_line_calls_video(video_path: str | Path, points: Sequence[TrackPoint], calls: Sequence[LineCall],
                            output_path: str | Path, cal: CourtCalibration,
                            config: CallAnnotatorConfig | None = None) -> int:
    """Re-read the video and write it with ball trail, bounce markers, call banner and minimap."""
    cfg = config or CallAnnotatorConfig()
    by_frame = {p.frame_index: p for p in points}
    trail_ann = TrajectoryAnnotator(cal, cfg.trail)
    calls = sorted(calls, key=lambda c: c.bounce.frame)
    history: list[TrackPoint] = []
    with VideoReader(video_path) as reader:
        info = reader.info
        fps = info.fps if info.fps > 0 else 30.0
        hold = cfg.hold_s * fps
        with VideoWriterMP4(output_path, fps, info.size) as writer:
            for idx, frame, _ in reader:
                p = by_frame.get(idx)
                if p is not None:
                    history.append(p)
                    frame = trail_ann.draw(frame, history)
                done = [c for c in calls if c.bounce.frame <= idx + 0.5]
                active = [c for c in done if idx - c.bounce.frame <= hold]
                for c in active:
                    fade = 1.0 - 0.6 * max(0.0, (idx - c.bounce.frame) / hold)
                    draw_call(frame, c, cal, cfg, fade)
                if cfg.show_banner and active:
                    draw_banner(frame, active[-1])
                if cfg.show_minimap:
                    draw_call_minimap(frame, done, history, cal, cfg)
                writer.write(frame)
            return writer.frames_written
