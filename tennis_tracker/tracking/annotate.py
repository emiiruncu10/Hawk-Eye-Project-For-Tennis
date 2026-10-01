"""Draw the ball, its trail and a calibrated minimap; write the annotated MP4."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from ..court import geometry as G
from ..court.calibration import CourtCalibration
from ..court.visualize import draw_minimap
from ..inference.video import VideoReader
from .tracker import TrackPoint

COLORS = {"detected": (0, 255, 0), "predicted": (0, 200, 255), "lost": (0, 0, 255)}


@dataclass
class AnnotatorConfig:
    tail_length: int = 12
    radius: int = 7
    thickness: int = 2
    draw_minimap: bool = True
    minimap_scale: float = 6.0          # px per metre
    minimap_margin: int = 12
    minimap_trail: int = 60
    show_hud: bool = True
    draw_court_lines: bool = False      # reproject the court model onto the frame


class TrajectoryAnnotator:
    def __init__(self, calibration: CourtCalibration | None, config: AnnotatorConfig | None = None):
        self.cal = calibration
        self.cfg = config or AnnotatorConfig()
        self._spec = G.MinimapSpec(scale=self.cfg.minimap_scale, margin_m=1.5)
        self._minimap_base = draw_minimap(self._spec) if self.cfg.draw_minimap else None
        self._court_px = None
        if self.cal is not None and self.cfg.draw_court_lines:
            self._court_px = self.cal.keypoints_all_px

    def to_court_m(self, x: float, y: float) -> np.ndarray | None:
        """Ball pixel -> court metres (ground-plane projection) using the calibration."""
        if self.cal is None:
            return None
        return self.cal.pixel_to_court([[x, y]])[0]

    def draw(self, frame: np.ndarray, history: Sequence[TrackPoint]) -> np.ndarray:
        """Annotate ``frame`` with the last points of ``history`` (newest last)."""
        cfg = self.cfg
        out = frame.copy()
        if self._court_px is not None:
            for a, b in G.COURT_LINES.values():
                cv2.line(out, _pt(self._court_px[a]), _pt(self._court_px[b]), (255, 255, 255), 1, cv2.LINE_AA)
        tail = [p for p in history[-(cfg.tail_length + 1):] if p.has_position]
        # trail: older segments thinner / darker
        for i in range(1, len(tail)):
            a, b = tail[i - 1], tail[i]
            t = i / max(1, len(tail) - 1)
            col = tuple(int(c * (0.35 + 0.65 * t)) for c in COLORS[b.status])
            cv2.line(out, _pt((a.x, a.y)), _pt((b.x, b.y)), col, max(1, int(round(cfg.thickness * t)) + 1), cv2.LINE_AA)
        cur = history[-1] if history else None
        if cur is not None and cur.has_position:
            cv2.circle(out, _pt((cur.x, cur.y)), cfg.radius, COLORS[cur.status], -1, cv2.LINE_AA)
            cv2.circle(out, _pt((cur.x, cur.y)), cfg.radius + 2, (0, 0, 0), 1, cv2.LINE_AA)
        if cfg.show_hud and cur is not None:
            txt = f"frame {cur.frame_index}  {cur.status}  conf {cur.confidence:.2f}"
            if cur.has_position and self.cal is not None:
                m = self.to_court_m(cur.x, cur.y)
                txt += f"  court ({m[0]:.2f} m, {m[1]:.2f} m)"
            cv2.putText(out, txt, (12, out.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, txt, (12, out.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cur.status], 1, cv2.LINE_AA)
        if self._minimap_base is not None and self.cal is not None:
            self._paste_minimap(out, history)
        return out

    def _paste_minimap(self, out: np.ndarray, history: Sequence[TrackPoint]) -> None:
        mm = self._minimap_base.copy()
        pts = [p for p in history[-self.cfg.minimap_trail:] if p.has_position]
        if pts:
            court = self.cal.pixel_to_court(np.array([[p.x, p.y] for p in pts]))
            mpx = self._spec.world_to_minimap(court)
            for i in range(1, len(mpx)):
                cv2.line(mm, _pt(mpx[i - 1]), _pt(mpx[i]), (0, 200, 255), 1, cv2.LINE_AA)
            cv2.circle(mm, _pt(mpx[-1]), 4, COLORS[pts[-1].status], -1, cv2.LINE_AA)
        h, w = mm.shape[:2]
        H, W = out.shape[:2]
        if h + self.cfg.minimap_margin > H or w + self.cfg.minimap_margin > W:
            return
        y0, x0 = self.cfg.minimap_margin, W - w - self.cfg.minimap_margin
        roi = out[y0:y0 + h, x0:x0 + w]
        cv2.addWeighted(mm, 0.85, roi, 0.15, 0, dst=roi)
        cv2.rectangle(out, (x0 - 1, y0 - 1), (x0 + w, y0 + h), (255, 255, 255), 1)


def _pt(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


class VideoWriterMP4:
    """cv2.VideoWriter that prefers H.264 (avc1) and falls back to mp4v."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], codecs=("avc1", "mp4v")):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = (int(size[0]), int(size[1]))
        self.frames_written = 0
        self.codec = None
        self._w = None
        for c in codecs:
            w = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*c), float(fps), self.size)
            if w.isOpened():
                self._w, self.codec = w, c
                break
            w.release()
        if self._w is None:
            raise RuntimeError(f"no usable MP4 codec among {codecs}")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size)
        self._w.write(frame)
        self.frames_written += 1

    def release(self) -> None:
        if self._w is not None:
            self._w.release()
            self._w = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def render_annotated_video(video_path: str | Path, points: Sequence[TrackPoint], output_path: str | Path,
                           calibration: CourtCalibration | None, config: AnnotatorConfig | None = None,
                           start: int = 0, end: int | None = None,
                           progress: Callable[[int], None] | None = None) -> int:
    """Re-read the source video and write ``output_path`` with the trajectory drawn.

    Frames without a track point (e.g. the first two warm-up frames) are
    written unchanged. Returns the number of frames written.
    """
    by_frame = {p.frame_index: p for p in points}
    ann = TrajectoryAnnotator(calibration, config)
    history: list[TrackPoint] = []
    with VideoReader(video_path, start, end) as reader:
        info = reader.info
        with VideoWriterMP4(output_path, info.fps if info.fps > 0 else 30.0, info.size) as writer:
            for idx, frame, _ in reader:
                p = by_frame.get(idx)
                if p is not None:
                    history.append(p)
                    frame = ann.draw(frame, history)
                writer.write(frame)
                if progress is not None:
                    progress(idx)
            return writer.frames_written


def export_csv(points: Sequence[TrackPoint], path: str | Path,
               calibration: CourtCalibration | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "status", "x_px", "y_px", "confidence", "raw_x_px", "raw_y_px",
                    "court_x_m", "court_y_m", "segment"])
        for p in points:
            cx = cy = ""
            if p.has_position and calibration is not None:
                m = calibration.pixel_to_court([[p.x, p.y]])[0]
                cx, cy = f"{m[0]:.3f}", f"{m[1]:.3f}"
            w.writerow([p.frame_index, p.status,
                        "" if p.x is None else f"{p.x:.2f}", "" if p.y is None else f"{p.y:.2f}",
                        f"{p.confidence:.4f}",
                        "" if p.raw_x is None else f"{p.raw_x:.2f}", "" if p.raw_y is None else f"{p.raw_y:.2f}",
                        cx, cy, p.segment])
    return path
