"""Small exact H1 utilities for the exploratory order-shuffle diagnostic.

The main empirical pipeline continues to use :mod:`ripser`.  This module is
restricted to a deterministic point subsample of each delay-coordinate
cloud so that the post-hoc order-sensitivity diagnostic can be reproduced
without changing the selected forecasting model.
"""
from __future__ import annotations

import numpy as np


def h1_persistence(points: np.ndarray, *, tolerance: float = 1e-12) -> np.ndarray:
    """Compute finite one-dimensional Vietoris--Rips intervals over F2.

    This standard boundary-matrix reduction is intended for small point clouds
    used by the diagnostic only.  Simplices are ordered by filtration value,
    dimension, and lexicographic vertex order; zero-persistence intervals are
    discarded.
    """
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3:
        return np.empty((0, 2), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("points must be finite")

    differences = values[:, None, :] - values[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    n_points = values.shape[0]

    edges: list[tuple[float, int, int]] = []
    for upper in range(n_points):
        for lower in range(upper):
            edges.append((float(distances[lower, upper]), lower, upper))
    edges.sort(key=lambda item: (item[0], item[1], item[2]))
    edge_index = {(lower, upper): idx for idx, (_, lower, upper) in enumerate(edges)}

    # Reduce the edge-to-vertex boundary matrix. Zero columns create H1.
    vertex_pivots: dict[int, int] = {}
    births: dict[int, float] = {}
    for edge_position, (diameter, lower, upper) in enumerate(edges):
        column = (1 << lower) | (1 << upper)
        while column:
            pivot = column.bit_length() - 1
            previous = vertex_pivots.get(pivot)
            if previous is None:
                break
            column ^= previous
        if column:
            vertex_pivots[column.bit_length() - 1] = column
        else:
            births[edge_position] = diameter

    triangles: list[tuple[float, int, int, int, int, int, int]] = []
    for first in range(n_points - 2):
        for second in range(first + 1, n_points - 1):
            e12 = edge_index[(first, second)]
            for third in range(second + 1, n_points):
                e13 = edge_index[(first, third)]
                e23 = edge_index[(second, third)]
                diameter = max(
                    distances[first, second],
                    distances[first, third],
                    distances[second, third],
                )
                triangles.append((float(diameter), first, second, third, e12, e13, e23))
    triangles.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    edge_pivots: dict[int, int] = {}
    deaths: dict[int, float] = {}
    for diameter, _, _, _, e12, e13, e23 in triangles:
        column = (1 << e12) | (1 << e13) | (1 << e23)
        while column:
            pivot = column.bit_length() - 1
            previous = edge_pivots.get(pivot)
            if previous is None:
                break
            column ^= previous
        if column:
            pivot = column.bit_length() - 1
            edge_pivots[pivot] = column
            if pivot in births:
                deaths[pivot] = diameter

    intervals = [
        (birth, deaths[edge_position])
        for edge_position, birth in births.items()
        if edge_position in deaths and deaths[edge_position] > birth + tolerance
    ]
    if not intervals:
        return np.empty((0, 2), dtype=float)
    return np.asarray(intervals, dtype=float)


def weighted_landscape(
    intervals: np.ndarray,
    *,
    grid: np.ndarray,
    weights: tuple[float, ...] = (1.0, 0.5, 0.25),
) -> np.ndarray:
    """Evaluate a weighted average of the first landscape layers."""
    bars = np.asarray(intervals, dtype=float)
    x = np.asarray(grid, dtype=float)
    if bars.size == 0:
        return np.zeros_like(x)
    bars = bars[np.isfinite(bars).all(axis=1)]
    if bars.size == 0:
        return np.zeros_like(x)

    tents = np.maximum(
        0.0,
        np.minimum(x[None, :] - bars[:, 0, None], bars[:, 1, None] - x[None, :]),
    )
    n_layers = len(weights)
    if tents.shape[0] < n_layers:
        tents = np.vstack(
            [tents, np.zeros((n_layers - tents.shape[0], x.size), dtype=float)]
        )
    largest = np.partition(tents, -n_layers, axis=0)[-n_layers:]
    largest.sort(axis=0)
    largest = largest[::-1]
    normalized = np.asarray(weights, dtype=float)
    normalized /= normalized.sum()
    return np.average(largest, axis=0, weights=normalized)


def window_landscape(
    window: np.ndarray,
    *,
    tau: int = 3,
    embedding_dimension: int = 5,
    n_points: int = 60,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Create the diagnostic H1 landscape for one return window.

    The same within-window standardization, delay, embedding dimension, grid,
    and layer weights as the selected model are used.  A deterministic,
    evenly spaced point subsample makes the diagnostic computationally
    tractable; it is not used to replace the forecasting model.
    """
    returns = np.asarray(window, dtype=float).reshape(-1)
    if not np.isfinite(returns).all():
        raise ValueError("window must be finite")
    scale = float(returns.std(ddof=0))
    standardized = (returns - returns.mean()) / (scale if scale > 1e-8 else 1.0)
    usable = standardized.size - tau * (embedding_dimension - 1)
    if usable < 3:
        raise ValueError("window is too short for the requested embedding")
    cloud = np.column_stack(
        [standardized[offset * tau : offset * tau + usable] for offset in range(embedding_dimension)]
    )
    take = min(int(n_points), cloud.shape[0])
    indices = np.unique(np.rint(np.linspace(0, cloud.shape[0] - 1, take)).astype(int))
    sampled = cloud[indices]
    intervals = h1_persistence(sampled)
    if grid is None:
        grid = np.linspace(0.0, 3.0, 200)
    return weighted_landscape(intervals, grid=np.asarray(grid, dtype=float))
