from __future__ import annotations

import numpy as np

from tda_risk.diagnostic_h1 import h1_persistence, weighted_landscape, window_landscape


def test_square_has_expected_h1_interval() -> None:
    square = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    intervals = h1_persistence(square)
    assert intervals.shape == (1, 2)
    assert np.allclose(intervals[0], [1.0, np.sqrt(2.0)])


def test_filled_triangle_has_no_positive_h1_interval() -> None:
    triangle = np.asarray([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]], dtype=float)
    assert h1_persistence(triangle).shape == (0, 2)


def test_landscape_and_window_feature_are_deterministic() -> None:
    grid = np.linspace(0.0, 3.0, 200)
    landscape = weighted_landscape(np.asarray([[1.0, 2.0]]), grid=grid)
    assert landscape.shape == (200,)
    assert landscape.max() > 0.0
    window = np.sin(np.linspace(0.0, 8.0 * np.pi, 250))
    first = window_landscape(window, grid=grid)
    second = window_landscape(window, grid=grid)
    assert np.allclose(first, second)
