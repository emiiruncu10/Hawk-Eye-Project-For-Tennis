import numpy as np

from tennis_tracker.court import geometry as G


def test_keypoint_count_and_extent():
    assert G.COURT_KEYPOINTS_M.shape == (14, 2)
    assert G.COURT_KEYPOINTS_M[:, 0].min() == 0.0
    assert np.isclose(G.COURT_KEYPOINTS_M[:, 0].max(), 10.97)
    assert G.COURT_KEYPOINTS_M[:, 1].min() == 0.0
    assert np.isclose(G.COURT_KEYPOINTS_M[:, 1].max(), 23.77)


def test_service_lines_are_6_40_from_net():
    assert np.isclose(G.NET_Y - G.Y_SERVICE_FAR, 6.40)
    assert np.isclose(G.Y_SERVICE_NEAR - G.NET_Y, 6.40)
    assert np.isclose(G.X_SINGLES_R - G.X_SINGLES_L, 8.23)


def test_lines_consistent_with_keypoint_table():
    for idx, (hn, vn) in G.KEYPOINT_FROM_LINES.items():
        y = dict((n, y) for n, y, _ in G.HORIZONTAL_MODEL_LINES)[hn]
        x = dict((n, x) for n, x, _ in G.VERTICAL_MODEL_LINES)[vn]
        assert np.allclose(G.COURT_KEYPOINTS_M[idx], (x, y))


def test_minimap_roundtrip():
    spec = G.MinimapSpec(scale=15.0, margin_m=1.0)
    pts = np.random.default_rng(0).uniform(0, 20, (10, 2))
    back = spec.minimap_to_world(spec.world_to_minimap(pts))
    assert np.allclose(back, pts)
    M = spec.world_to_minimap_matrix()
    hom = np.hstack([pts, np.ones((10, 1))]) @ M.T
    assert np.allclose(hom[:, :2] / hom[:, 2:], spec.world_to_minimap(pts))


def test_sampled_lines_cover_every_line():
    s = G.sample_model_lines_m(0.5)
    assert len(s) > 100
    assert s[:, 0].min() >= 0 and s[:, 0].max() <= G.COURT_WIDTH + 1e-9
