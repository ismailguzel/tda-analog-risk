"""Primary VaR(0.99) comparative inference.

The public functions operate on date-aligned arrays/data frames so inference
cannot accidentally compare different forecast panels.  Joint VaR--ES
inference remains separate in ``scripts/run_inference.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from .scoring import hac_variance


def pinball_loss(realized: np.ndarray, forecast: np.ndarray, level: float = 0.99) -> np.ndarray:
    y = np.asarray(realized, dtype=float)
    q = np.asarray(forecast, dtype=float)
    return np.where(y >= q, level * (y - q), (1.0 - level) * (q - y))


def hln_factor(n: int, horizon: int = 1) -> float:
    """Harvey--Leybourne--Newbold small-sample factor for forecast horizon h."""
    if n <= 1 or horizon < 1:
        return math.nan
    return float(np.sqrt((n + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / n) / n))


def dm_pinball(
    loss_difference: np.ndarray,
    bandwidth: int | None = None,
    horizon: int = 1,
) -> tuple[float, float]:
    """HLN-corrected Newey--West DM statistic and two-sided p-value."""
    d = np.asarray(loss_difference, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return math.nan, math.nan
    variance = hac_variance(d, bandwidth)
    if not np.isfinite(variance) or variance <= 0.0:
        return (0.0, 1.0) if abs(float(d.mean())) < 1e-15 else (math.copysign(math.inf, d.mean()), 0.0)
    raw = float(d.mean() / np.sqrt(variance / n))
    statistic = raw * hln_factor(n, horizon)
    return statistic, float(2.0 * norm.sf(abs(statistic)))


def paired_pinball_comparison(
    realized: np.ndarray,
    topology_forecast: np.ndarray,
    comparator_forecasts: pd.DataFrame | dict[str, np.ndarray],
    *,
    level: float = 0.99,
    bandwidth: int | None = None,
    horizon: int = 1,
    bootstrap_reps: int = 500,
    block_length: int = 10,
    seed: int = 20260927,
) -> pd.DataFrame:
    """Compare topology against each comparator on the same forecast dates."""
    y = np.asarray(realized, dtype=float)
    topology = np.asarray(topology_forecast, dtype=float)
    if isinstance(comparator_forecasts, pd.DataFrame):
        comparators = {str(name): comparator_forecasts[name].to_numpy(float) for name in comparator_forecasts}
    else:
        comparators = {str(name): np.asarray(values, dtype=float) for name, values in comparator_forecasts.items()}
    rows: list[dict[str, object]] = []
    raw_pvalues: list[float] = []
    for name, comparator in comparators.items():
        if comparator.shape != y.shape or topology.shape != y.shape:
            raise ValueError("all forecasts must have the same shape as realized")
        topo_loss = pinball_loss(y, topology, level)
        comp_loss = pinball_loss(y, comparator, level)
        difference = topo_loss - comp_loss
        statistic, pvalue = dm_pinball(difference, bandwidth, horizon)
        ci_low, ci_high = circular_block_ci(difference, block_length, bootstrap_reps, seed + len(rows))
        rows.append(
            {
                "topology": "topology",
                "comparator": name,
                "n_pairs": int(np.isfinite(difference).sum()),
                "mean_pinball_difference_topology_minus_comparator": float(np.nanmean(difference)),
                "dm_statistic": statistic,
                "dm_pvalue": pvalue,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "level": level,
                "horizon": horizon,
            }
        )
        raw_pvalues.append(pvalue)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["holm_pvalue"] = holm_adjust(np.asarray(raw_pvalues, dtype=float))
    return result


def circular_block_ci(
    values: np.ndarray,
    block_length: int = 10,
    bootstrap_reps: int = 500,
    seed: int = 20260927,
) -> tuple[float, float]:
    """Circular moving-block bootstrap CI for a mean loss difference."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return math.nan, math.nan
    block = max(1, min(int(block_length), x.size))
    rng = np.random.default_rng(seed)
    means = np.empty(int(bootstrap_reps), dtype=float)
    for b in range(means.size):
        sample: list[float] = []
        while len(sample) < x.size:
            start = int(rng.integers(0, x.size))
            sample.extend(float(x[(start + offset) % x.size]) for offset in range(block))
        means[b] = float(np.mean(sample[: x.size]))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    order = np.argsort(p[valid])
    sorted_p = p[valid][order]
    adjusted = np.minimum(1.0, np.maximum.accumulate(sorted_p * (len(sorted_p) - np.arange(len(sorted_p)))))
    indices = np.flatnonzero(valid)[order]
    out[indices] = adjusted
    return out


def var99_model_confidence_set(
    losses: pd.DataFrame,
    *,
    alpha: float = 0.10,
    bootstrap_reps: int = 500,
    block_length: int = 10,
    seed: int = 20260927,
) -> pd.DataFrame:
    """Range-MCS-style elimination using VaR(0.99) pinball losses.

    The input columns are model labels and rows are the same forecast dates.
    The output has one row per input label and records the sequential
    range-statistic decision; no model is selected by a heuristic score.
    """
    if losses.empty or losses.shape[1] < 2:
        raise ValueError("MCS needs at least two labelled model-loss columns")
    matrix = losses.to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("MCS losses must be finite and date-aligned")
    labels = list(map(str, losses.columns))
    current = list(range(matrix.shape[1]))
    removed: dict[int, tuple[float, float]] = {}
    rng = np.random.default_rng(seed)
    while len(current) > 1:
        active = matrix[:, current]
        means = active.mean(axis=0)
        observed = _range_statistic(active)
        centered = active - means
        bootstrap = np.empty(bootstrap_reps, dtype=float)
        for b in range(bootstrap_reps):
            sample = _circular_resample(centered, block_length, rng)
            bootstrap[b] = _range_statistic(sample)
        pvalue = float((1.0 + np.sum(bootstrap >= observed)) / (bootstrap_reps + 1.0))
        if pvalue >= alpha:
            break
        worst_local = int(np.argmax(means))
        worst = current.pop(worst_local)
        removed[worst] = (observed, pvalue)
    rows = []
    for index, label in enumerate(labels):
        stat, pvalue = removed.get(index, (math.nan, math.nan))
        rows.append(
            {
                "model": label,
                "included_in_mcs": index in current,
                "mean_pinball_loss": float(matrix[:, index].mean()),
                "mcs_range_statistic": stat,
                "mcs_pvalue": pvalue,
                "alpha": alpha,
                "n_dates": matrix.shape[0],
                "n_models": matrix.shape[1],
            }
        )
    return pd.DataFrame(rows)


def _range_statistic(values: np.ndarray) -> float:
    means = values.mean(axis=0)
    standard = np.sqrt(np.maximum(values.var(axis=0, ddof=1), 1e-18) / values.shape[0])
    return float(np.max(np.abs((means - means.mean()) / standard)))


def _circular_resample(values: np.ndarray, block_length: int, rng: np.random.Generator) -> np.ndarray:
    n = values.shape[0]
    block = max(1, min(int(block_length), n))
    rows: list[np.ndarray] = []
    while sum(part.shape[0] for part in rows) < n:
        start = int(rng.integers(0, n))
        indices = (start + np.arange(block)) % n
        rows.append(values[indices])
    return np.vstack(rows)[:n]
