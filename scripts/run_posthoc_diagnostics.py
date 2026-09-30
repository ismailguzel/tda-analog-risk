"""Run the two frozen post-hoc diagnostics added before resubmission.

The analyses do not alter model selection or the fixed evaluation forecasts:

1. an age-stratified recency-matched retrieval benchmark for the no-exclusion
   topology forecasts; and
2. a representation-level within-window order-shuffle diagnostic for H1.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "posthoc_diagnostics"
TOPOLOGY_ID = "topology_l250_k100"
AGE_EDGES = np.asarray([1, 21, 61, 126, 251, 501, 1001, 10**9], dtype=int)
AGE_LABELS = ("1-20", "21-60", "61-125", "126-250", "251-500", "501-1000", "1001+")
RECENCY_SEED = 2026093001
SHUFFLE_SEED = 2026093002


def _read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except (ImportError, ModuleNotFoundError):
        from tda_risk.parquet_fallback import MiniParquet

        reader = MiniParquet(path)
        payload = {column: reader.read_column(column) for column in columns}
        return pd.DataFrame(payload)


def _panel_and_state() -> tuple[pd.DataFrame, np.ndarray]:
    panel = pd.read_csv(ROOT / "data" / "panel_published.csv.gz", parse_dates=["date"])
    state = panel.copy()
    state["rv20_p"] = state["portfolio_return"].rolling(20).std(ddof=0) * np.sqrt(252.0)
    state["ret20_spy"] = (
        (1.0 + state["SPY_return"]).rolling(20).apply(np.prod, raw=True) - 1.0
    )
    state["VIX"] = state["^VIX_close"]
    state["Delta5_DGS10"] = state["DGS10"].diff(5)
    state["HYOAS"] = state["credit_spread_proxy"]
    state_valid = np.isfinite(
        state[["rv20_p", "ret20_spy", "VIX", "Delta5_DGS10", "T10Y2Y", "HYOAS"]]
    ).all(axis=1).to_numpy()
    return panel, state_valid


def _topology_rows() -> pd.DataFrame:
    columns = [
        "configuration_id",
        "forecast_date",
        "realized_loss",
        "pinball_99",
        "neighbor_indices",
    ]
    frame = _read_parquet_columns(
        ROOT / "results" / "final_evaluation" / "forecasts_radius0.parquet", columns
    )
    frame = frame.loc[frame["configuration_id"].astype(str).eq(TOPOLOGY_ID)].copy()
    # Arrow timestamps are normal pandas timestamps; the lightweight fallback
    # returns nanoseconds since epoch.
    if not np.issubdtype(frame["forecast_date"].dtype, np.datetime64):
        frame["forecast_date"] = pd.to_datetime(frame["forecast_date"].astype("int64"))
    else:
        frame["forecast_date"] = pd.to_datetime(frame["forecast_date"])
    frame = frame.sort_values("forecast_date").reset_index(drop=True)
    if len(frame) != 2848:
        raise RuntimeError(f"Expected 2,848 topology forecasts, found {len(frame)}")
    return frame


def _parse_indices(value: object) -> np.ndarray:
    if not isinstance(value, str) or not value:
        return np.empty(0, dtype=int)
    return np.fromiter((int(piece) for piece in value.split(";") if piece), dtype=int)


@dataclass
class RecencyQuery:
    realized_loss: float
    counts: np.ndarray
    pools: tuple[np.ndarray, ...]


def _prepare_recency_queries(
    frame: pd.DataFrame, panel: pd.DataFrame, state_valid: np.ndarray
) -> tuple[list[RecencyQuery], pd.DataFrame]:
    dates = pd.DatetimeIndex(panel["date"])
    positions = {pd.Timestamp(date): idx for idx, date in enumerate(dates)}
    returns = panel["portfolio_return"].to_numpy(dtype=float)
    losses = panel["portfolio_loss"].to_numpy(dtype=float)

    window_valid = np.zeros(len(panel), dtype=bool)
    for position in range(249, len(panel)):
        window_valid[position] = np.isfinite(returns[position - 249 : position + 1]).all()
    feature_and_state_valid = window_valid & state_valid

    queries: list[RecencyQuery] = []
    age_records: list[dict[str, float | int | str]] = []
    all_ages: list[int] = []
    jaccard_100: list[float] = []
    jaccard_250: list[float] = []

    for row in frame.itertuples(index=False):
        forecast_position = positions[pd.Timestamp(row.forecast_date)]
        neighbors = _parse_indices(row.neighbor_indices)
        if neighbors.size != 100:
            raise RuntimeError(
                f"Expected 100 topology neighbors at {row.forecast_date}, found {neighbors.size}"
            )
        ages = forecast_position - neighbors
        counts = np.histogram(ages, bins=AGE_EDGES)[0]

        candidates = np.arange(forecast_position, dtype=int)
        valid = feature_and_state_valid[:forecast_position] & np.isfinite(
            losses[1 : forecast_position + 1]
        )
        candidates = candidates[valid]
        candidate_ages = forecast_position - candidates
        pools = tuple(
            candidates[
                (candidate_ages >= AGE_EDGES[index])
                & (candidate_ages < AGE_EDGES[index + 1])
            ]
            for index in range(len(AGE_LABELS))
        )
        if any(pool.size < count for pool, count in zip(pools, counts)):
            raise RuntimeError(f"Age-matched pool is too small at {row.forecast_date}")
        if not set(neighbors).issubset(set(candidates)):
            raise RuntimeError(f"Stored topology neighbor is not eligible at {row.forecast_date}")

        queries.append(
            RecencyQuery(float(row.realized_loss), counts.astype(int), pools)
        )
        all_ages.extend(ages.tolist())
        neighbor_set = set(neighbors.tolist())
        recent100 = set(candidates[-100:].tolist())
        recent250 = set(candidates[-250:].tolist())
        jaccard_100.append(len(neighbor_set & recent100) / len(neighbor_set | recent100))
        jaccard_250.append(len(neighbor_set & recent250) / len(neighbor_set | recent250))

    age_array = np.asarray(all_ages, dtype=float)
    age_records.append(
        {
            "measure": "mean_age",
            "value": float(age_array.mean()),
        }
    )
    for quantile in (0.05, 0.25, 0.50, 0.75, 0.95):
        age_records.append(
            {
                "measure": f"age_q{int(100 * quantile):02d}",
                "value": float(np.quantile(age_array, quantile)),
            }
        )
    for cutoff in (20, 60, 125, 250):
        age_records.append(
            {
                "measure": f"share_age_le_{cutoff}",
                "value": float(np.mean(age_array <= cutoff)),
            }
        )
    age_records.extend(
        [
            {"measure": "mean_jaccard_recent_100", "value": float(np.mean(jaccard_100))},
            {"measure": "median_jaccard_recent_100", "value": float(np.median(jaccard_100))},
            {"measure": "mean_jaccard_recent_250", "value": float(np.mean(jaccard_250))},
            {"measure": "median_jaccard_recent_250", "value": float(np.median(jaccard_250))},
        ]
    )
    return queries, pd.DataFrame(age_records)


_RECENCY_QUERIES: list[RecencyQuery] | None = None
_RECENCY_LOSSES: np.ndarray | None = None


def _init_recency_worker(queries: list[RecencyQuery], losses: np.ndarray) -> None:
    global _RECENCY_QUERIES, _RECENCY_LOSSES
    _RECENCY_QUERIES = queries
    _RECENCY_LOSSES = losses


def _one_recency_draw_global(seed: int) -> tuple[float, float, float]:
    if _RECENCY_QUERIES is None or _RECENCY_LOSSES is None:
        raise RuntimeError("recency worker was not initialized")
    return _one_recency_draw(seed, _RECENCY_QUERIES, _RECENCY_LOSSES)


def _one_recency_draw(
    seed: int, queries: list[RecencyQuery], losses: np.ndarray
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    pinball: list[float] = []
    exceedances: list[float] = []
    forecasts: list[float] = []
    for query in queries:
        selected: list[np.ndarray] = []
        for count, pool in zip(query.counts, query.pools):
            if count == 0:
                continue
            selected.append(
                pool if count == pool.size else rng.choice(pool, size=int(count), replace=False)
            )
        positions = np.concatenate(selected)
        scenarios = losses[positions + 1]
        forecast = float(np.quantile(scenarios, 0.99))
        error = query.realized_loss - forecast
        pinball.append(0.99 * error if error >= 0.0 else -0.01 * error)
        exceedances.append(float(query.realized_loss > forecast))
        forecasts.append(forecast)
    return float(np.mean(pinball)), float(np.mean(exceedances)), float(np.mean(forecasts))


def run_recency_matched(draws: int, workers: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = _topology_rows()
    panel, state_valid = _panel_and_state()
    queries, age_summary = _prepare_recency_queries(frame, panel, state_valid)
    losses = panel["portfolio_loss"].to_numpy(dtype=float)
    seeds = [RECENCY_SEED + draw * 1000003 for draw in range(draws)]

    if workers <= 1:
        results = [_one_recency_draw(seed, queries, losses) for seed in seeds]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_recency_worker,
            initargs=(queries, losses),
        ) as executor:
            results = list(executor.map(_one_recency_draw_global, seeds, chunksize=4))

    draw_frame = pd.DataFrame(
        {
            "draw": np.arange(draws, dtype=int),
            "seed": seeds,
            "mean_pinball_99": [item[0] for item in results],
            "exceedance_rate_99": [item[1] for item in results],
            "mean_var_99": [item[2] for item in results],
        }
    )
    observed_loss = float(frame["pinball_99"].mean())
    observed_rate = float((frame["realized_loss"].to_numpy(dtype=float) > 0).mean())  # overwritten below
    observed_rate = float(
        pd.read_csv(ROOT / "results" / "final_evaluation" / "model_summary_radius0.csv")
        .loc[lambda x: x["configuration_id"].astype(str).eq(TOPOLOGY_ID), "exceedance_rate_99"]
        .iloc[0]
    )
    numerator = int(np.sum(draw_frame["mean_pinball_99"].to_numpy() <= observed_loss))
    pvalue = (numerator + 1.0) / (draws + 1.0)
    summary = pd.DataFrame(
        [
            {
                "benchmark": "age_stratified_recency_matched",
                "draws": draws,
                "topology_mean_pinball_99": observed_loss,
                "null_mean_pinball_99": float(draw_frame["mean_pinball_99"].mean()),
                "null_p05_pinball_99": float(draw_frame["mean_pinball_99"].quantile(0.05)),
                "null_p95_pinball_99": float(draw_frame["mean_pinball_99"].quantile(0.95)),
                "topology_minus_null_mean_pinball_99": observed_loss
                - float(draw_frame["mean_pinball_99"].mean()),
                "lower_tail_randomization_numerator": numerator,
                "lower_tail_randomization_pvalue": pvalue,
                "topology_exceedance_rate_99": observed_rate,
                "null_mean_exceedance_rate_99": float(draw_frame["exceedance_rate_99"].mean()),
                "null_p05_exceedance_rate_99": float(
                    draw_frame["exceedance_rate_99"].quantile(0.05)
                ),
                "null_p95_exceedance_rate_99": float(
                    draw_frame["exceedance_rate_99"].quantile(0.95)
                ),
                "age_bins": ";".join(AGE_LABELS),
                "seed_root": RECENCY_SEED,
            }
        ]
    )
    return draw_frame, summary, age_summary


def _shuffle_window_worker(payload: tuple[int, np.ndarray, int, int]) -> dict[str, float | int | str]:
    from tda_risk.diagnostic_h1 import window_landscape

    window_number, window, n_shuffles, n_points = payload
    seed = SHUFFLE_SEED + window_number * 1000003
    rng = np.random.default_rng(seed)
    original = window_landscape(window, n_points=n_points)
    shuffled = np.vstack(
        [window_landscape(rng.permutation(window), n_points=n_points) for _ in range(n_shuffles)]
    )
    original_distances = np.linalg.norm(shuffled - original[None, :], axis=1)
    differences = shuffled[:, None, :] - shuffled[None, :, :]
    pairwise = np.sqrt(np.sum(differences * differences, axis=2))
    triangle = pairwise[np.triu_indices(n_shuffles, k=1)]
    combined = np.vstack([original, shuffled])
    combined_differences = combined[:, None, :] - combined[None, :, :]
    combined_distances = np.sqrt(np.sum(combined_differences * combined_differences, axis=2))
    centrality = combined_distances.sum(axis=1) / n_shuffles
    original_percentile = (
        1.0 + float(np.sum(centrality[1:] <= centrality[0]))
    ) / (n_shuffles + 1.0)
    return {
        "window_number": window_number,
        "seed": seed,
        "mean_original_to_shuffle_distance": float(original_distances.mean()),
        "median_original_to_shuffle_distance": float(np.median(original_distances)),
        "mean_shuffle_to_shuffle_distance": float(triangle.mean()),
        "median_shuffle_to_shuffle_distance": float(np.median(triangle)),
        "distance_difference": float(original_distances.mean() - triangle.mean()),
        "original_centrality_percentile": float(original_percentile),
        "original_landscape_norm": float(np.linalg.norm(original)),
        "mean_shuffled_landscape_norm": float(np.mean(np.linalg.norm(shuffled, axis=1))),
    }


def _sign_flip_pvalues(
    values: np.ndarray, seed: int, repetitions: int = 100000
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    observed = float(values.mean())
    lower = 0
    upper = 0
    batch = 2000
    for start in range(0, repetitions, batch):
        take = min(batch, repetitions - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(take, values.size))
        permuted = (signs * values[None, :]).mean(axis=1)
        lower += int(np.sum(permuted <= observed))
        upper += int(np.sum(permuted >= observed))
    p_lower = (lower + 1.0) / (repetitions + 1.0)
    p_upper = (upper + 1.0) / (repetitions + 1.0)
    return p_lower, p_upper, min(1.0, 2.0 * min(p_lower, p_upper))


def run_temporal_shuffle(
    n_windows: int,
    n_shuffles: int,
    n_points: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _topology_rows()
    panel = pd.read_csv(ROOT / "data" / "panel_published.csv.gz", parse_dates=["date"])
    dates = pd.DatetimeIndex(panel["date"])
    positions = {pd.Timestamp(date): idx for idx, date in enumerate(dates)}
    returns = panel["portfolio_return"].to_numpy(dtype=float)

    selected_rows = np.unique(
        np.rint(np.linspace(0, len(frame) - 1, n_windows)).astype(int)
    )
    payloads: list[tuple[int, np.ndarray, int, int]] = []
    selected_dates: dict[int, str] = {}
    for window_number, row_index in enumerate(selected_rows):
        date = pd.Timestamp(frame.loc[row_index, "forecast_date"])
        position = positions[date]
        window = returns[position - 249 : position + 1]
        if window.size != 250 or not np.isfinite(window).all():
            raise RuntimeError(f"Invalid diagnostic window at {date.date()}")
        payloads.append((window_number, window, n_shuffles, n_points))
        selected_dates[window_number] = date.date().isoformat()

    if workers <= 1:
        records = [_shuffle_window_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_shuffle_window_worker, payloads))

    detail = pd.DataFrame(records).sort_values("window_number").reset_index(drop=True)
    detail.insert(1, "forecast_date", detail["window_number"].map(selected_dates))
    differences = detail["distance_difference"].to_numpy(dtype=float)
    rng = np.random.default_rng(SHUFFLE_SEED + 77)
    bootstrap_means = np.empty(5000, dtype=float)
    for draw in range(bootstrap_means.size):
        bootstrap_means[draw] = rng.choice(differences, size=differences.size, replace=True).mean()
    p_lower, p_upper, p_two_sided = _sign_flip_pvalues(
        differences, SHUFFLE_SEED + 99
    )
    top_decile_count = int(np.sum(detail["original_centrality_percentile"] >= 0.90))
    top_decile_test = binomtest(top_decile_count, n=len(detail), p=0.10, alternative="greater")
    summary = pd.DataFrame(
        [
            {
                "diagnostic": "within_window_temporal_shuffle_h1",
                "sampled_windows": len(detail),
                "shuffles_per_window": n_shuffles,
                "delay_tau": 3,
                "embedding_dimension": 5,
                "subsampled_delay_points": n_points,
                "landscape_grid_points": 200,
                "mean_original_to_shuffle_distance": float(
                    detail["mean_original_to_shuffle_distance"].mean()
                ),
                "mean_shuffle_to_shuffle_distance": float(
                    detail["mean_shuffle_to_shuffle_distance"].mean()
                ),
                "mean_distance_difference": float(differences.mean()),
                "median_distance_difference": float(np.median(differences)),
                "bootstrap_ci_low": float(np.quantile(bootstrap_means, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap_means, 0.975)),
                "lower_tail_sign_flip_pvalue": p_lower,
                "upper_tail_sign_flip_pvalue": p_upper,
                "two_sided_sign_flip_pvalue": p_two_sided,
                "original_top_decile_centrality_count": top_decile_count,
                "original_top_decile_centrality_share": top_decile_count / len(detail),
                "top_decile_binomial_pvalue": float(top_decile_test.pvalue),
                "seed_root": SHUFFLE_SEED,
            }
        ]
    )
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recency-draws", type=int, default=500)
    parser.add_argument("--shuffle-windows", type=int, default=100)
    parser.add_argument("--shuffle-draws", type=int, default=25)
    parser.add_argument("--shuffle-points", type=int, default=30)
    parser.add_argument("--recency-workers", type=int, default=8)
    parser.add_argument("--shuffle-workers", type=int, default=1)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    recency_draws, recency_summary, age_summary = run_recency_matched(
        args.recency_draws, args.recency_workers
    )
    recency_draws.to_csv(OUT / "recency_matched_draws.csv", index=False)
    recency_summary.to_csv(OUT / "recency_matched_summary.csv", index=False)
    age_summary.to_csv(OUT / "topology_neighbor_recency.csv", index=False)

    shuffle_detail, shuffle_summary = run_temporal_shuffle(
        args.shuffle_windows,
        args.shuffle_draws,
        args.shuffle_points,
        args.shuffle_workers,
    )
    shuffle_detail.to_csv(OUT / "temporal_shuffle_window_detail.csv", index=False)
    shuffle_summary.to_csv(OUT / "temporal_shuffle_summary.csv", index=False)
    print(recency_summary.to_string(index=False))
    print(shuffle_summary.to_string(index=False))


if __name__ == "__main__":
    main()
