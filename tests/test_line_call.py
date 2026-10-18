import csv

import numpy as np
import pytest

from tennis_tracker.court import geometry as G
from tennis_tracker.tracking.bounce import Bounce
from tennis_tracker.tracking.line_call import (
    CLOSE,
    IN,
    OUT,
    CourtRules,
    call_bounce,
    clearances,
    export_calls_csv,
    summarize,
)


def bounce(x, y, sx=0.01, sy=0.01):
    return Bounce(frame=10.0, time_s=0.2, x_px=0, y_px=0, contact_x_px=1, contact_y_px=2, court_x_m=x, court_y_m=y,
                  half="near" if y > G.NET_Y else "far", jump_vy_mps=-8, jump_vx_mps=0, tstat=9,
                  vy_before_mps=8, vy_after_mps=-2, px_per_m=80, cov_m=np.diag([sx ** 2, sy ** 2]))


def test_centre_of_court_is_in():
    c = call_bounce(bounce(5.0, 18.0))
    assert c.call == IN and c.clearance_m > 1.0


def test_touching_the_line_is_in():
    # contact point exactly on the centre of the left singles sideline
    c = call_bounce(bounce(G.X_SINGLES_L, 18.0))
    assert c.nearest_line == "left sideline"
    assert c.clearance_m == pytest.approx(0.025 + 0.02)      # half line width + contact patch half-width
    assert c.call == IN


def test_out_wide_and_long():
    c = call_bounce(bounce(G.X_SINGLES_R + 0.30, 18.0))
    assert c.call == OUT and c.nearest_line == "right sideline"
    assert c.clearance_m == pytest.approx(-(0.30 - 0.045))
    c = call_bounce(bounce(5.0, G.Y_BASE_NEAR + 0.5))
    assert c.call == OUT and c.nearest_line == "near baseline"
    assert c.clearance_m == pytest.approx(-(0.5 - 0.05 - 0.02))
    c = call_bounce(bounce(5.0, -0.4))
    assert c.call == OUT and c.nearest_line == "far baseline"


def test_close_band_uses_sigma_perpendicular_to_line():
    # 3 cm outside the boundary: sideline with small lateral sigma -> OUT
    x = G.X_SINGLES_L - 0.045 - 0.04
    assert call_bounce(bounce(x, 18.0, sx=0.01, sy=0.5)).call == OUT
    # same offset at the baseline with a large depth sigma -> CLOSE
    y = G.Y_BASE_NEAR + 0.07 + 0.04
    c = call_bounce(bounce(5.0, y, sx=0.01, sy=0.10))
    assert c.call == CLOSE and c.band_m == pytest.approx(0.20) and c.leaning == OUT


def test_min_close_band():
    x = G.X_SINGLES_L - 0.045 + 0.02          # 2 cm inside the boundary, tiny sigma
    c = call_bounce(bounce(x, 18.0, sx=0.001))
    assert c.band_m == pytest.approx(0.03) and c.call == CLOSE and c.leaning == IN


def test_singles_vs_doubles():
    b = bounce(0.6, 18.0)                     # in the doubles alley
    assert call_bounce(b, CourtRules("singles")).call == OUT
    assert call_bounce(b, CourtRules("doubles")).call == IN
    with pytest.raises(ValueError):
        CourtRules("mixed")


def test_clearances_all_four_lines():
    c = clearances(5.0, 10.0, CourtRules())
    assert set(c) == {"left sideline", "right sideline", "far baseline", "near baseline"}
    assert all(v > 0 for v in c.values())


def test_export_and_summary(tmp_path):
    calls = [call_bounce(bounce(5.0, 18.0)), call_bounce(bounce(5.0, 24.5)), call_bounce(bounce(G.X_SINGLES_L - 0.045, 18.0))]
    assert summarize(calls) == {IN: 1, OUT: 1, CLOSE: 1}
    p = export_calls_csv(calls, tmp_path / "calls.csv")
    rows = list(csv.DictReader(p.open()))
    assert [r["call"] for r in rows] == [IN, OUT, CLOSE]
    assert rows[1]["nearest_line"] == "near baseline" and float(rows[1]["clearance_cm"]) < 0
    assert rows[0]["game"] == "singles" and rows[0]["bounce"] == "1"
