"""Scoring and backtesting utilities used by the revision audit."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2


def fz0_score(
    realized: np.ndarray,
    var: np.ndarray,
    es: np.ndarray,
    level: float = 0.975,
) -> np.ndarray:
    """Positive-loss FZ0 score for jointly forecast VaR and ES.

    This is the orientation used throughout the revision: smaller is better.
    The score is defined only where all inputs are finite and ES is positive;
    invalid rows are returned as NaN so callers can report their sample size.
    """
    y = np.asarray(realized, dtype=float)
    q = np.asarray(var, dtype=float)
    e = np.asarray(es, dtype=float)
    out = np.full(np.broadcast(y, q, e).shape, np.nan, dtype=float)
    valid = np.isfinite(y) & np.isfinite(q) & np.isfinite(e) & (e > 0.0)
    tail = (y > q) & valid
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out[valid] = (
            tail[valid] * (y[valid] - q[valid]) / ((1.0 - level) * e[valid])
            + q[valid] / e[valid]
            + np.log(e[valid])
            - 1.0
        )
    return out


def pinball_score(realized: np.ndarray, var: np.ndarray, level: float) -> np.ndarray:
    y = np.asarray(realized, dtype=float)
    q = np.asarray(var, dtype=float)
    return np.where(y >= q, level * (y - q), (1.0 - level) * (q - y))


def kupiec_uc(exceedances: np.ndarray, level: float) -> tuple[float, float]:
    """Kupiec unconditional-coverage likelihood-ratio statistic and p-value."""
    x = np.asarray(exceedances, dtype=bool)
    n = int(x.size)
    k = int(x.sum())
    if n == 0:
        return math.nan, math.nan
    p = 1.0 - level
    phat = min(max(k / n, 1e-12), 1.0 - 1e-12)
    ll_null = (n - k) * math.log(1.0 - p) + k * math.log(p)
    ll_alt = (n - k) * math.log(1.0 - phat) + k * math.log(phat)
    statistic = max(0.0, -2.0 * (ll_null - ll_alt))
    pvalue = float(chi2.sf(statistic, df=1))
    return statistic, pvalue


def christoffersen_independence(exceedances: np.ndarray) -> tuple[float, float]:
    """Christoffersen first-order independence LR statistic and p-value."""
    x = np.asarray(exceedances, dtype=bool)
    if x.size < 2:
        return math.nan, math.nan
    n00 = n01 = n10 = n11 = 0
    for previous, current in zip(x[:-1], x[1:]):
        if not previous and not current:
            n00 += 1
        elif not previous and current:
            n01 += 1
        elif previous and not current:
            n10 += 1
        else:
            n11 += 1
    pi0 = (n01 / max(n00 + n01, 1))
    pi1 = (n11 / max(n10 + n11, 1))
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(prob: float, successes: int, failures: int) -> float:
        prob = min(max(prob, 1e-12), 1.0 - 1e-12)
        return successes * math.log(prob) + failures * math.log(1.0 - prob)

    ll_ind = ll(pi, n01 + n11, n00 + n10)
    ll_markov = ll(pi0, n01, n00) + ll(pi1, n11, n10)
    statistic = max(0.0, -2.0 * (ll_ind - ll_markov))
    pvalue = float(chi2.sf(statistic, df=1))
    return statistic, pvalue


def conditional_coverage(exceedances: np.ndarray, level: float) -> tuple[float, float]:
    uc_stat, uc_p = kupiec_uc(exceedances, level)
    ind_stat, ind_p = christoffersen_independence(exceedances)
    if not np.isfinite(uc_stat) or not np.isfinite(ind_stat):
        return math.nan, math.nan
    stat = uc_stat + ind_stat
    # LR_CC combines UC and first-order independence and therefore has two
    # restrictions under the null (Christoffersen, 1998).
    return stat, float(chi2.sf(stat, df=2))


def exceedance_durations(exceedances: np.ndarray) -> np.ndarray:
    x = np.asarray(exceedances, dtype=bool)
    hits = np.flatnonzero(x)
    if hits.size < 2:
        return np.array([], dtype=float)
    return np.diff(hits).astype(float)


def duration_geometric_test(exceedances: np.ndarray, level: float) -> tuple[float, float]:
    """LR test of geometric inter-exceedance durations.

    The test uses uncensored intervals between observed exceedances and compares
    the MLE geometric success probability with ``p=1-level``.  The reported
    reference distribution is chi-square(1); the finite-sample caveat is
    recorded in the methods report because no duration correction is applied.
    """
    durations = exceedance_durations(exceedances)
    if durations.size == 0:
        return math.nan, math.nan
    p0 = float(1.0 - level)
    total = float(durations.sum())
    phat = min(max(float(durations.size / total), 1e-12), 1.0 - 1e-12)
    p0 = min(max(p0, 1e-12), 1.0 - 1e-12)

    def loglik(prob: float) -> float:
        return durations.size * math.log(prob) + (total - durations.size) * math.log(1.0 - prob)

    statistic = max(0.0, 2.0 * (loglik(phat) - loglik(p0)))
    return statistic, float(chi2.sf(statistic, df=1))


def acerbi_szekely_z2(
    realized: np.ndarray,
    var: np.ndarray,
    es: np.ndarray,
    level: float = 0.975,
) -> tuple[float, float, int]:
    """Studentized Acerbi--Szekely Z2 ES moment test.

    For positive losses the Z2 contribution is
    ``loss * I(loss > VaR) / ((1-level) * ES) - 1``.  Under a calibrated
    forecast its mean is zero.  We use the recognized Z2 moment with a
    studentized asymptotic normal reference; the report states that this is
    not a small-sample or bootstrap correction.
    """
    y = np.asarray(realized, dtype=float)
    q = np.asarray(var, dtype=float)
    e = np.asarray(es, dtype=float)
    valid = np.isfinite(y) & np.isfinite(q) & np.isfinite(e) & (e > 0.0)
    if valid.sum() < 2:
        return math.nan, math.nan, int(valid.sum())
    alpha = 1.0 - level
    contributions = y[valid] * (y[valid] > q[valid]) / (alpha * e[valid]) - 1.0
    std = float(np.std(contributions, ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        statistic = 0.0 if abs(float(contributions.mean())) < 1e-15 else math.copysign(math.inf, float(contributions.mean()))
        return statistic, 1.0 if statistic == 0.0 else 0.0, int(valid.sum())
    statistic = float(np.sqrt(contributions.size) * contributions.mean() / std)
    return statistic, math.erfc(abs(statistic) / math.sqrt(2.0)), int(valid.sum())


def hac_variance(values: np.ndarray, bandwidth: int | None = None) -> float:
    """Newey-West long-run variance of a scalar series."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return math.nan
    centered = x - x.mean()
    if bandwidth is None:
        bandwidth = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    gamma0 = float(np.dot(centered, centered) / n)
    variance = gamma0
    for lag in range(1, min(bandwidth, n - 1) + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        variance += 2.0 * weight * gamma
    return max(variance, 0.0)


def dm_test(loss_difference: np.ndarray, bandwidth: int | None = None) -> tuple[float, float]:
    """HAC Diebold-Mariano-style statistic and two-sided normal p-value."""
    d = np.asarray(loss_difference, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return math.nan, math.nan
    variance = hac_variance(d, bandwidth)
    if not np.isfinite(variance) or variance <= 0.0:
        return (0.0, 1.0) if abs(float(d.mean())) < 1e-15 else (math.copysign(math.inf, d.mean()), 0.0)
    statistic = float(d.mean() / math.sqrt(variance / d.size))
    pvalue = math.erfc(abs(statistic) / math.sqrt(2.0))
    return statistic, pvalue
