"""Run the locked revised finalists on the fixed evaluation panel.

The script never writes to submitted result directories. It stores compact
forecast rows plus lossless compressed neighbor-index arrays by radius.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tda_risk.config import default_pipeline_config
from tda_risk.features import attach_state_zscores, build_state_panel
from tda_risk.methods import (
    DTWWindow,
    EuclideanKNN,
    FHSEWMA,
    FPCAWindow,
    GARCHStudentT,
    GapHS,
    MahalanobisKNN,
    RegimeHS,
    RollingHS,
    StandardizedDTWWindow,
    StandardizedFPCAWindow,
    StandardizedWindowEuclideanKNN,
    WindowEuclideanKNN,
    _dtw_top_positions,
    _euclidean_top_positions,
    _fpca_top_positions,
    _mahalanobis_top_positions,
    _scenario_losses_from_positions,
    _standardized_dtw_top_positions,
    _standardized_fpca_top_positions,
    _standardized_window_euclidean_top_positions,
    _window_euclidean_top_positions,
    clear_method_caches,
    empirical_var_es,
)
from tda_risk.topology import TopologyKNN, _topology_top_positions, clear_topology_caches


OUT = ROOT / "results" / "final_evaluation"
REGISTRY = ROOT / "revision_config" / "finalists.csv"
STATE_COLUMNS = default_pipeline_config().state_columns
K_MAX = 1000
RADIUS_VALUES = (0, 125, 250)


def load_data() -> pd.DataFrame:
    panel = pd.read_csv(ROOT / "data" / "panel_published.csv.gz", index_col=0, parse_dates=True)
    state = build_state_panel(panel)
    return attach_state_zscores(state, STATE_COLUMNS, min_history=60)


def parse_config(config_id: str, radius: int):
    cfg = default_pipeline_config()
    match = re.fullmatch(r"rolling_hs_w(\d+)", config_id)
    if match:
        return RollingHS(window=int(match.group(1)))
    match = re.fullmatch(r"fhs_ewma_l(\d+)", config_id)
    if match:
        return FHSEWMA(lambda_=int(match.group(1)) / 100.0)
    match = re.fullmatch(r"regime_hs_bins(\d+)", config_id)
    if match:
        return RegimeHS(regime_bins=int(match.group(1)))
    if config_id == "garch_t_w1500_r20":
        return GARCHStudentT(simulation_draws=20000, fit_window=1500, refit_interval=20)
    if config_id == "gap_hs_w250":
        return GapHS(window=250, exclusion_radius=radius)
    match = re.fullmatch(r"gap_hs_w250_r(125|250)", config_id)
    if match:
        return GapHS(window=250, exclusion_radius=int(match.group(1)))
    match = re.fullmatch(r"(s_euclidean|s_mahalanobis)_k(\d+)", config_id)
    if match:
        cls = EuclideanKNN if match.group(1) == "s_euclidean" else MahalanobisKNN
        return cls(k=int(match.group(2)), state_columns=STATE_COLUMNS, max_k=K_MAX, exclusion_radius=radius)
    match = re.fullmatch(r"(w|wz)_(euclidean|dtw|fpca)_l(\d+)_k(\d+)", config_id)
    if match:
        prefix, family, length, k = match.groups()
        length, k = int(length), int(k)
        standardized = prefix == "wz"
        if family == "euclidean":
            cls = StandardizedWindowEuclideanKNN if standardized else WindowEuclideanKNN
        elif family == "dtw":
            cls = StandardizedDTWWindow if standardized else DTWWindow
        else:
            cls = StandardizedFPCAWindow if standardized else FPCAWindow
        return cls(k=k, window_length=length, max_k=K_MAX, exclusion_radius=radius)
    match = re.fullmatch(r"topology_l(\d+)_k(\d+)", config_id)
    if match:
        length, k = map(int, match.groups())
        topology_cfg = replace(
            cfg.topology,
            window_lengths=(length,),
            feature_modes=("landscape_h1_top3_weighted",),
            input_modes=("portfolio_only",),
            alphas=(0.0,),
            n_jobs=1,
        )
        return TopologyKNN(
            k=k,
            state_columns=STATE_COLUMNS,
            max_k=K_MAX,
            window_length=length,
            input_mode="portfolio_only",
            feature_mode="landscape_h1_top3_weighted",
            alpha=0.0,
            topology=topology_cfg,
            training_end=cfg.splits.warmup_end,
            exclusion_radius=radius,
        )
    raise ValueError(f"Unknown locked configuration: {config_id}")


def neighbor_positions(data: pd.DataFrame, position: int, method: object, radius: int) -> np.ndarray:
    if isinstance(method, EuclideanKNN):
        return _euclidean_top_positions(data, position, STATE_COLUMNS, method.max_k, radius)[: method.k]
    if isinstance(method, MahalanobisKNN):
        return _mahalanobis_top_positions(data, position, STATE_COLUMNS, method.max_k, radius)[: method.k]
    if isinstance(method, StandardizedWindowEuclideanKNN):
        return _standardized_window_euclidean_top_positions(data, position, method.window_length, method.max_k, radius)[: method.k]
    if isinstance(method, WindowEuclideanKNN):
        return _window_euclidean_top_positions(data, position, method.window_length, method.max_k, radius)[: method.k]
    if isinstance(method, StandardizedDTWWindow):
        return _standardized_dtw_top_positions(data, position, method.window_length, method.band, method.max_k, radius)[: method.k]
    if isinstance(method, DTWWindow):
        return _dtw_top_positions(data, position, method.window_length, method.band, method.max_k, radius)[: method.k]
    if isinstance(method, StandardizedFPCAWindow):
        return _standardized_fpca_top_positions(data, position, method.window_length, method.n_components, method.max_k, radius)[: method.k]
    if isinstance(method, FPCAWindow):
        return _fpca_top_positions(data, position, method.window_length, method.n_components, method.max_k, radius)[: method.k]
    if isinstance(method, TopologyKNN):
        return _topology_top_positions(
            data,
            position,
            STATE_COLUMNS,
            method.window_length,
            method.input_mode,
            method.feature_mode,
            method.alpha,
            method.max_k,
            method.topology,
            method.training_end,
            use_null=False,
            exclusion_radius=radius,
        )[: method.k]
    return np.array([], dtype=int)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--radii", type=int, nargs="+", default=list(RADIUS_VALUES))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    registry = pd.read_csv(REGISTRY)
    config = default_pipeline_config()
    all_frames = []
    for radius in tuple(args.radii):
        clear_method_caches()
        clear_topology_caches()
        archived_path = (
            ROOT
            / "archive"
            / "superseded_results"
            / "revision_phase3_locked_evaluation"
            / f"forecasts_radius{radius}.parquet"
        )
        if archived_path.exists():
            archived = pd.read_parquet(archived_path)
        else:
            archived_csv = archived_path.with_suffix(".csv")
            archived = (
                pd.read_csv(archived_csv, parse_dates=["forecast_date", "realized_date"])
                if archived_csv.exists()
                else pd.DataFrame()
            )
        rows: list[dict[str, object]] = []
        config_ids = registry["configuration_id"].astype(str).tolist()
        if radius in (125, 250):
            config_ids.append(f"gap_hs_w250_r{radius}")
        for config_id in config_ids:
            if config_id.startswith("gap_hs_w250_r") and config_id != f"gap_hs_w250_r{radius}":
                continue
            if not archived.empty:
                cached = archived[archived["configuration_id"].astype(str).eq(config_id)]
                if not cached.empty:
                    rows.extend(cached.to_dict("records"))
                    print(
                        f"radius={radius} config={config_id} reused_exact_archived_rows={len(cached)}",
                        flush=True,
                    )
                    continue
            method = parse_config(config_id, radius)
            start = time.perf_counter()
            for position in range(len(data) - 1):
                date = data.index[position]
                if date < pd.Timestamp(config.splits.test_start):
                    continue
                realized = float(data["portfolio_loss"].iloc[position + 1])
                try:
                    forecast = method.forecast(data, position, config.risk)
                except (ValueError, RuntimeError):
                    continue
                neighbors = neighbor_positions(data, position, method, radius)
                scenarios = (
                    np.asarray(forecast.scenario_losses, dtype=float)
                    if forecast.scenario_losses is not None
                    else _scenario_losses_from_positions(data, neighbors)
                )
                if scenarios.size:
                    var975, es975 = empirical_var_es(scenarios, 0.975, 0.975)
                else:
                    var975, es975 = np.nan, np.nan
                if neighbors.size:
                    dates_text = ";".join(data.index[neighbors].strftime("%Y-%m-%d"))
                    indices_text = ";".join(map(str, neighbors.tolist()))
                else:
                    dates_text = ""
                    indices_text = ""
                rows.append(
                    {
                        "configuration_id": config_id,
                        "method": forecast.method,
                        "forecast_date": date,
                        "realized_date": data.index[position + 1],
                        "exclusion_radius": radius,
                        "realized_loss": realized,
                        "VaR_99": forecast.var,
                        "VaR_975": var975,
                        "ES_975": es975,
                        "scenario_count": forecast.scenario_count,
                        "var_exceedance_99": float(realized > forecast.var),
                        "pinball_99": float(np.where(realized >= forecast.var, 0.99 * (realized - forecast.var), 0.01 * (forecast.var - realized))),
                        "neighbor_indices": indices_text,
                        "neighbor_dates": dates_text,
                    }
                )
            elapsed = time.perf_counter() - start
            print(f"radius={radius} config={config_id} rows={sum(r['configuration_id'] == config_id for r in rows)} seconds={elapsed:.2f}", flush=True)
        frame = pd.DataFrame(rows)
        frame.to_parquet(OUT / f"forecasts_radius{radius}.parquet", index=False)
        all_frames.append(frame)
        summary = (
            frame.groupby(["exclusion_radius", "configuration_id", "method"], dropna=False)
        .agg(
            n_forecasts=("forecast_date", "size"),
            mean_pinball_99=("pinball_99", "mean"),
            exceedance_rate_99=("var_exceedance_99", "mean"),
            mean_var_99=("VaR_99", "mean"),
            mean_var_975=("VaR_975", "mean"),
            mean_es_975=("ES_975", "mean"),
            mean_scenarios=("scenario_count", "mean"),
        )
        .reset_index()
        )
        summary.to_csv(OUT / f"model_summary_radius{radius}.csv", index=False)


if __name__ == "__main__":
    main()
