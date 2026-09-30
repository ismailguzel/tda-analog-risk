"""History-only null-control mechanics.

The two primary controls target one complete-IID null through different
implementations: the topology placebo permutes feature rows within the
eligible history and then ranks candidate *dates*, while uniform retrieval
samples candidate dates directly.  Direct-permutation distance ties are
resolved by an independent seeded random key, never by candidate position.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class Selection:
    """Auditable result for one forecast date and one null draw."""

    control: str
    forecast_position: int
    candidate_positions: np.ndarray
    selected_positions: np.ndarray
    feature_origins: np.ndarray
    query_feature: np.ndarray
    assigned_features: np.ndarray
    distances: np.ndarray | None
    seed: int

    @property
    def selected_loss_positions(self) -> np.ndarray:
        """Positions whose following-day losses must be used as scenarios."""
        return self.selected_positions + 1

    @property
    def neighbor_checksum(self) -> str:
        return _checksum(self.selected_positions)

    @property
    def scenario_checksum(self) -> str:
        return _checksum(self.selected_loss_positions)


def eligible_history(
    forecast_position: int,
    n_rows: int,
    exclusion_radius: int,
    feature_matrix: np.ndarray,
    loss_values: np.ndarray,
) -> np.ndarray:
    """Return dates ``s`` with feature and next-day loss available.

    The boundary is ``s <= t - radius - 1``.  The loss check is performed on
    ``L[s+1]`` and never on the permuted feature origin.
    """
    if exclusion_radius < 0:
        raise ValueError("exclusion_radius must be non-negative")
    if forecast_position < 0 or forecast_position >= n_rows:
        raise IndexError("forecast_position is outside the panel")
    features = np.asarray(feature_matrix, dtype=float)
    losses = np.asarray(loss_values, dtype=float)
    if features.shape[0] != n_rows or losses.shape[0] != n_rows:
        raise ValueError("feature and loss arrays must have n_rows entries")
    last = min(n_rows - 2, forecast_position - exclusion_radius - 1)
    if last < 0:
        return np.array([], dtype=int)
    candidates = np.arange(last + 1, dtype=int)
    valid_feature = np.isfinite(features[candidates]).all(axis=1)
    valid_loss = np.isfinite(losses[candidates + 1])
    return candidates[valid_feature & valid_loss]


def _standardize_catalog(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return values, mean, scale


def _rank(
    candidate_positions: np.ndarray,
    distances: np.ndarray,
    k: int,
    tie_breaker: np.ndarray,
) -> np.ndarray:
    take = min(int(k), candidate_positions.size)
    if take < 1:
        raise ValueError("k must be positive and candidates must be non-empty")
    if tie_breaker.shape != distances.shape:
        raise ValueError("tie_breaker must have one value per candidate")
    # The random key is independent of candidate date and feature origin.
    order = np.lexsort((tie_breaker, distances))
    return order[:take]


def topology_history_permutation(
    query_feature: np.ndarray,
    candidate_positions: np.ndarray,
    feature_matrix: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    seed: int,
    permutation: np.ndarray | None = None,
    forecast_position: int = -1,
) -> Selection:
    """Break feature/date alignment while retaining the historical catalog.

    ``permutation[j]`` is the source row assigned to candidate date
    ``candidate_positions[j]``.  The query is never permuted.  The selected
    dates, not the feature origins, determine scenario losses.
    """
    positions = np.asarray(candidate_positions, dtype=int)
    if positions.size == 0:
        raise ValueError("candidate_positions must be non-empty")
    matrix = np.asarray(feature_matrix, dtype=float)
    query = np.asarray(query_feature, dtype=float)
    catalog = matrix[positions]
    if not np.isfinite(query).all() or not np.isfinite(catalog).all():
        raise ValueError("query and candidate features must be finite")
    if permutation is None:
        permutation = rng.permutation(positions.size)
    permutation = np.asarray(permutation, dtype=int)
    if not np.array_equal(np.sort(permutation), np.arange(positions.size)):
        raise ValueError("permutation must contain each candidate row exactly once")
    assigned = catalog[permutation]
    _, mean, scale = _standardize_catalog(catalog)
    standardized_query = (query - mean) / scale
    standardized_assigned = (assigned - mean) / scale
    distances = np.linalg.norm(standardized_assigned - standardized_query, axis=1)
    tie_breaker = rng.random(positions.size)
    selected_order = _rank(positions, distances, k, tie_breaker)
    return Selection(
        control="topology_history_permutation",
        forecast_position=forecast_position,
        candidate_positions=positions,
        selected_positions=positions[selected_order],
        feature_origins=positions[permutation[selected_order]],
        query_feature=query.copy(),
        assigned_features=assigned,
        distances=distances,
        seed=int(seed),
    )


def uniform_random_retrieval(
    candidate_positions: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    seed: int,
    forecast_position: int = -1,
) -> Selection:
    """Select historical candidate dates without computing topology distances."""
    positions = np.asarray(candidate_positions, dtype=int)
    take = min(int(k), positions.size)
    if take < 1:
        raise ValueError("k must be positive and candidates must be non-empty")
    selected = np.sort(rng.choice(positions, size=take, replace=False))
    return Selection(
        control="uniform_random_retrieval",
        forecast_position=forecast_position,
        candidate_positions=positions,
        selected_positions=selected,
        feature_origins=np.array([], dtype=int),
        query_feature=np.array([], dtype=float),
        assigned_features=np.empty((0, 0), dtype=float),
        distances=None,
        seed=int(seed),
    )


def _checksum(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()
