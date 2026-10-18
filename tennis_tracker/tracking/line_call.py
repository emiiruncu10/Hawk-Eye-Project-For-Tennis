"""Line calls (IN / OUT / CLOSE) for detected bounces.

Rule: a ball that touches any part of a line is in. Court dimensions are
measured to the *outer* edges of the lines, while the calibration aligns
the court model with the *centres* of the painted lines (that is what the
line detector measures). The in/out boundary therefore lies half a line
width outside each model line, and the ball counts as touching it when its
contact patch reaches it.

For every bounce the signed *clearance* to the decisive boundary is

    clearance = distance from the contact patch edge to the outer edge of
                the nearest boundary line  (> 0 inside, < 0 outside)

and the call is

    CLOSE  if |clearance| < band,  band = max(k * sigma, min_band)
    IN     if clearance >= 0
    OUT    otherwise

where ``sigma`` is the 1-sigma uncertainty of the bounce position measured
perpendicular to that line (from the trajectory fit, the tracking noise and
the calibration error, mapped onto the court). A single camera cannot
resolve millimetre calls; CLOSE says so instead of guessing.

Scope: rally rules on the singles (default) or doubles court. Service-box
rules for serves are not applied yet; a serve is judged against the whole
court.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..court import geometry as G
from .bounce import Bounce

IN, OUT, CLOSE = "IN", "OUT", "CLOSE"


@dataclass(frozen=True)
class CourtRules:
    game: str = "singles"               # "singles" | "doubles"
    baseline_width_m: float = 0.10      # ITF: baselines up to 10 cm wide
    line_width_m: float = 0.05          # ITF: other lines 2.5 - 5 cm
    contact_half_width_m: float = 0.02  # half-width of the ball's contact patch on the ground
    close_k_sigma: float = 2.0          # CLOSE band = k * sigma ...
    min_close_band_m: float = 0.03      # ... but never narrower than this

    def __post_init__(self):
        if self.game not in ("singles", "doubles"):
            raise ValueError("game must be 'singles' or 'doubles'")

    @property
    def sidelines(self) -> tuple[float, float]:
        if self.game == "singles":
            return G.X_SINGLES_L, G.X_SINGLES_R
        return G.X_DOUBLES_L, G.X_DOUBLES_R


@dataclass
class LineCall:
    bounce: Bounce
    call: str                 # IN | OUT | CLOSE
    clearance_m: float        # > 0 inside, < 0 outside
    nearest_line: str         # "left sideline" | "right sideline" | "far baseline" | "near baseline"
    sigma_m: float            # 1-sigma position uncertainty perpendicular to that line
    band_m: float             # |clearance| below this is CLOSE
    game: str = "singles"

    @property
    def leaning(self) -> str:
        """IN/OUT side a CLOSE call leans to (by the point estimate)."""
        return IN if self.clearance_m >= 0 else OUT


def clearances(x_m: float, y_m: float, rules: CourtRules) -> dict[str, float]:
    """Signed clearance (m) to each boundary line; > 0 means inside."""
    xl, xr = rules.sidelines
    side = rules.line_width_m / 2.0 + rules.contact_half_width_m
    end = rules.baseline_width_m / 2.0 + rules.contact_half_width_m
    return {
        "left sideline": x_m - (xl - side),
        "right sideline": (xr + side) - x_m,
        "far baseline": y_m - (G.Y_BASE_FAR - end),
        "near baseline": (G.Y_BASE_NEAR + end) - y_m,
    }


def call_bounce(b: Bounce, rules: CourtRules | None = None) -> LineCall:
    rules = rules or CourtRules()
    c = clearances(b.court_x_m, b.court_y_m, rules)
    line = min(c, key=c.get)
    clearance = c[line]
    sigma = b.sigma_x_m if "sideline" in line else b.sigma_y_m
    band = max(rules.close_k_sigma * sigma, rules.min_close_band_m)
    if abs(clearance) < band:
        call = CLOSE
    else:
        call = IN if clearance >= 0 else OUT
    return LineCall(b, call, float(clearance), line, float(sigma), float(band), rules.game)


def call_bounces(bounces: list[Bounce], rules: CourtRules | None = None) -> list[LineCall]:
    return [call_bounce(b, rules) for b in bounces]


def export_calls_csv(calls: list[LineCall], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bounce", "frame", "time_s", "half", "court_x_m", "court_y_m", "call", "leaning",
                    "clearance_cm", "nearest_line", "sigma_cm", "close_band_cm", "game",
                    "x_px", "y_px", "jump_vy_mps", "tstat"])
        for i, c in enumerate(calls, 1):
            b = c.bounce
            w.writerow([i, f"{b.frame:.2f}", f"{b.time_s:.3f}", b.half, f"{b.court_x_m:.3f}", f"{b.court_y_m:.3f}",
                        c.call, c.leaning, f"{100 * c.clearance_m:.1f}", c.nearest_line, f"{100 * c.sigma_m:.1f}",
                        f"{100 * c.band_m:.1f}", c.game, f"{b.contact_x_px:.1f}", f"{b.contact_y_px:.1f}",
                        f"{b.jump_vy_mps:.2f}", f"{b.tstat:.1f}"])
    return path


def summarize(calls: list[LineCall]) -> dict[str, int]:
    out = {IN: 0, OUT: 0, CLOSE: 0}
    for c in calls:
        out[c.call] += 1
    return out
