"""Run frozen-panel blocked annual validation for the revised candidate grid."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tda_risk.config import default_pipeline_config
from tda_risk.features import attach_state_zscores, build_state_panel
from tda_risk.methods import (
    DTWWindow,
    EuclideanKNN,
    FHSEWMA,
    FPCAWindow,
    GARCHStudentT,
    MahalanobisKNN,
    RegimeHS,
    RollingHS,
    StandardizedDTWWindow,
    StandardizedFPCAWindow,
    StandardizedWindowEuclideanKNN,
    WindowEuclideanKNN,
    clear_method_caches,
)
from tda_risk.registry import load_candidate_registry
from tda_risk.topology import TopologyKNN, clear_topology_caches


STATE_COLUMNS = default_pipeline_config().state_columns
K_VALUES = (100, 250, 500, 750, 1000)
OUT_DIR = REPO_ROOT / "results" / "final_validation"
LOG_PATH = REPO_ROOT / "logs" / "final_analysis" / "commands.log"
REGISTRY_PATH = REPO_ROOT / "revision_config" / "recovery_candidate_registry.csv"


def pinball_loss(realized: np.ndarray, forecast: np.ndarray, alpha: float = 0.99) -> np.ndarray:
    realized = np.asarray(realized, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return np.where(realized >= forecast, alpha * (realized - forecast), (1.0 - alpha) * (forecast - realized))


def load_frozen_state_panel() -> pd.DataFrame:
    panel = pd.read_csv(REPO_ROOT / "data" / "panel_published.csv.gz", index_col=0, parse_dates=True)
    state = build_state_panel(panel)
    return attach_state_zscores(state, STATE_COLUMNS, min_history=60)


def _log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}\n")
    print(message, flush=True)


def _number(row: pd.Series, name: str, cast):
    value = row.get(name)
    if pd.isna(value) or value == "":
        return None
    return cast(value)


def _method_specs(config, registry_path: Path = REGISTRY_PATH) -> list[tuple[str, object]]:
    """Instantiate exactly the rows in the version-controlled registry."""
    registry = load_candidate_registry(registry_path)
    specs: list[tuple[str, object]] = []
    max_k = K_VALUES[-1]
    for _, row in registry.iterrows():
        config_id = str(row["configuration_id"])
        method = str(row["method"])
        k = _number(row, "k", int)
        length = _number(row, "window_length", int)
        if method == "rolling_hs":
            instance = RollingHS(window=length)
        elif method == "fhs_ewma":
            instance = FHSEWMA(lambda_=_number(row, "lambda", float))
        elif method == "regime_hs":
            instance = RegimeHS(regime_bins=_number(row, "regime_bins", int))
        elif method == "garch_t":
            instance = GARCHStudentT(
                simulation_draws=20000,
                fit_window=_number(row, "fit_window", int),
                refit_interval=_number(row, "refit_interval", int),
            )
        elif method == "s_euclidean":
            instance = EuclideanKNN(k=k, state_columns=STATE_COLUMNS, max_k=max_k)
        elif method == "s_mahalanobis":
            instance = MahalanobisKNN(k=k, state_columns=STATE_COLUMNS, max_k=max_k)
        elif method == "w_euclidean":
            instance = WindowEuclideanKNN(k=k, window_length=length, max_k=max_k)
        elif method == "wz_euclidean":
            instance = StandardizedWindowEuclideanKNN(k=k, window_length=length, max_k=max_k)
        elif method == "w_dtw":
            instance = DTWWindow(k=k, window_length=length, max_k=max_k)
        elif method == "wz_dtw":
            instance = StandardizedDTWWindow(k=k, window_length=length, max_k=max_k)
        elif method == "w_fpca":
            instance = FPCAWindow(k=k, window_length=length, max_k=max_k)
        elif method == "wz_fpca":
            instance = StandardizedFPCAWindow(k=k, window_length=length, max_k=max_k)
        elif method == "topology":
            topology_cfg = replace(
                config.topology,
                window_lengths=(length,),
                feature_modes=(str(row["feature_mode"]),),
                input_modes=(str(row["topology_input_mode"]),),
                alphas=(float(row["alpha"]),),
                n_jobs=1,
            )
            instance = TopologyKNN(
                k=k,
                state_columns=STATE_COLUMNS,
                max_k=max_k,
                window_length=length,
                input_mode=str(row["topology_input_mode"]),
                feature_mode=str(row["feature_mode"]),
                alpha=float(row["alpha"]),
                topology=topology_cfg,
                training_end=config.splits.warmup_end,
            )
        else:
            raise ValueError(f"registry contains unsupported method {method!r}")
        specs.append((config_id, instance))
    return specs


def _run_spec(data: pd.DataFrame, config, config_id: str, method: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dates = data.index
    for position in range(len(data) - 1):
        date = dates[position]
        if date.year not in (2012, 2013, 2014):
            continue
        realized = float(data["portfolio_loss"].iloc[position + 1])
        if not np.isfinite(realized):
            continue
        try:
            forecast = method.forecast(data, position, config.risk)
        except (ValueError, RuntimeError) as exc:
            continue
        rows.append(
            {
                "configuration_id": config_id,
                "method": forecast.method,
                "forecast_date": date,
                "realized_date": dates[position + 1],
                "fold": str(date.year),
                "scenario_count": forecast.scenario_count,
                "VaR_99": forecast.var,
                "ES_975": forecast.es,
                "realized_loss": realized,
                "pinball_99": float(pinball_loss(np.array([realized]), np.array([forecast.var]))[0]),
                "var_exceedance_99": float(realized > forecast.var),
                **forecast.params,
            }
        )
    return rows


def _selection_stability(summary: pd.DataFrame) -> pd.DataFrame:
    """Return fold, full-sample, leave-one-fold-out, and rank stability rows."""
    rows: list[dict[str, object]] = []
    summary = summary.copy()
    summary["status"] = summary.get("status", "completed")
    complete = summary[summary["status"].eq("completed")].copy()
    if complete.empty:
        return pd.DataFrame(rows)
    complete["rank"] = complete.groupby("fold")["mean_pinball_loss"].rank(
        method="min", ascending=True
    )
    for fold, group in complete.groupby("fold", sort=True):
        winner = group.sort_values(["mean_pinball_loss", "configuration_id"]).iloc[0]
        rows.append(
            {
                "analysis": "fold_winner",
                "fold": str(fold),
                "configuration_id": winner["configuration_id"],
                "method": winner["method"],
                "mean_pinball_loss": winner["mean_pinball_loss"],
                "rank": 1,
            }
        )
    overall = (
        complete.groupby(["configuration_id", "method"], as_index=False)
        .agg(mean_pinball_loss=("mean_pinball_loss", "mean"), n_folds=("fold", "nunique"))
        .sort_values(["mean_pinball_loss", "configuration_id"])
    )
    winner = overall.iloc[0]
    rows.append(
        {
            "analysis": "full_validation_winner",
            "fold": "all",
            "configuration_id": winner["configuration_id"],
            "method": winner["method"],
            "mean_pinball_loss": winner["mean_pinball_loss"],
            "rank": 1,
        }
    )
    for omitted in sorted(complete["fold"].astype(str).unique()):
        reduced = complete[complete["fold"].astype(str) != omitted]
        reduced_overall = (
            reduced.groupby(["configuration_id", "method"], as_index=False)
            .agg(mean_pinball_loss=("mean_pinball_loss", "mean"), n_folds=("fold", "nunique"))
            .sort_values(["mean_pinball_loss", "configuration_id"])
        )
        if reduced_overall.empty:
            continue
        selected = reduced_overall.iloc[0]
        rows.append(
            {
                "analysis": "leave_one_fold_out",
                "fold": f"omit_{omitted}",
                "configuration_id": selected["configuration_id"],
                "method": selected["method"],
                "mean_pinball_loss": selected["mean_pinball_loss"],
                "rank": 1,
            }
        )
    rank_summary = (
        complete.groupby(["configuration_id", "method"], as_index=False)
        .agg(
            mean_rank=("rank", "mean"),
            worst_rank=("rank", "max"),
            rank_std=("rank", "std"),
            n_folds=("fold", "nunique"),
        )
        .fillna({"rank_std": 0.0})
    )
    for _, value in rank_summary.sort_values(["mean_rank", "configuration_id"]).iterrows():
        rows.append(
            {
                "analysis": "rank_variation",
                "fold": "all",
                "configuration_id": value["configuration_id"],
                "method": value["method"],
                "mean_rank": value["mean_rank"],
                "worst_rank": value["worst_rank"],
                "rank_std": value["rank_std"],
                "n_folds": value["n_folds"],
            }
        )
    return pd.DataFrame(rows)


def _write_finalist_registry(summary: pd.DataFrame) -> None:
    complete = summary[summary["status"].eq("completed")].copy()
    overall = (
        complete.groupby(["configuration_id", "method"], as_index=False)
        .agg(
            mean_pinball_loss=("mean_pinball_loss", "mean"),
            worst_fold_loss=("mean_pinball_loss", "max"),
            mean_exceedance_rate=("exceedance_rate", "mean"),
            mean_scenarios=("mean_scenarios", "mean"),
            runtime_seconds=("runtime_seconds", "max"),
            n_folds=("fold", "nunique"),
        )
    )
    rows = []
    for method, group in overall.groupby("method", sort=True):
        winner = group.sort_values(
            ["mean_pinball_loss", "worst_fold_loss", "configuration_id"]
        ).iloc[0]
        rows.append(
            {
                "registry_version": "phase_b2_final_v1",
                "family": method,
                "configuration_id": winner["configuration_id"],
                "method": method,
                "mean_pinball_loss": winner["mean_pinball_loss"],
                "worst_fold_loss": winner["worst_fold_loss"],
                "mean_exceedance_rate": winner["mean_exceedance_rate"],
                "mean_scenarios": winner["mean_scenarios"],
                "runtime_seconds": winner["runtime_seconds"],
                "n_folds": winner["n_folds"],
                "selection_rule": "min mean annual pinball; tie worst fold; tie configuration_id",
            }
        )
    pd.DataFrame(rows).sort_values("method").to_csv(
        REPO_ROOT / "revision_config" / "finalists.csv", index=False
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = default_pipeline_config()
    data = load_frozen_state_panel()
    _log(f"validation start rows={len(data)} panel={data.index.min().date()}..{data.index.max().date()}")
    clear_method_caches()
    clear_topology_caches()
    archived_checkpoint_dir = (
        REPO_ROOT / "archive" / "superseded_results" / "revision_phase2_validation" / "checkpoints"
    )
    reused_archived = 0
    fresh_runs = 0
    all_rows: list[dict[str, object]] = []
    for config_id, method in _method_specs(config):
        archived_checkpoint = archived_checkpoint_dir / f"{config_id}.csv"
        if archived_checkpoint.exists():
            cached = pd.read_csv(archived_checkpoint, parse_dates=["forecast_date", "realized_date"])
            all_rows.extend(cached.to_dict("records"))
            reused_archived += 1
            _log(f"reused exact archived checkpoint {config_id} rows={len(cached)}")
            continue
        started = time.perf_counter()
        rows = _run_spec(data, config, config_id, method)
        elapsed = time.perf_counter() - started
        for row in rows:
            row["runtime_seconds"] = elapsed
        all_rows.extend(rows)
        fresh_runs += 1
        _log(f"completed {config_id} rows={len(rows)} seconds={elapsed:.2f}")
    forecasts = pd.DataFrame(all_rows)
    if forecasts.empty:
        raise RuntimeError("Validation produced no rows.")
    forecasts["fold"] = forecasts["fold"].astype(str)
    raw_summary = (
        forecasts.groupby(["configuration_id", "method", "fold"], dropna=False)
        .agg(
            n_forecasts=("forecast_date", "size"),
            mean_pinball_loss=("pinball_99", "mean"),
            exceedance_rate=("var_exceedance_99", "mean"),
            mean_var=("VaR_99", "mean"),
            mean_scenarios=("scenario_count", "mean"),
            runtime_seconds=("runtime_seconds", "max"),
        )
        .reset_index()
    )
    registered = load_candidate_registry(REGISTRY_PATH)[["configuration_id", "method"]].rename(
        columns={"method": "registry_method"}
    )
    summary = registered.merge(raw_summary, on=["configuration_id"], how="left")
    summary["method"] = summary["registry_method"]
    summary = summary.drop(columns=["registry_method"])
    summary["status"] = np.where(summary["n_forecasts"].notna(), "completed", "infeasible")
    summary.to_csv(OUT_DIR / "validation_fold_summary.csv", index=False)
    forecasts.to_parquet(OUT_DIR / "all_configurations.parquet", index=False)
    pd.DataFrame(_selection_stability(summary)).to_csv(OUT_DIR / "selection_stability.csv", index=False)
    _write_finalist_registry(summary)
    run_summary = {
        "registry": str(REGISTRY_PATH.relative_to(REPO_ROOT)),
        "registered_configurations": int(len(registered)),
        "completed_configurations": int(summary.loc[summary["status"].eq("completed"), "configuration_id"].nunique()),
        "infeasible_configurations": int(summary.loc[summary["status"].eq("infeasible"), "configuration_id"].nunique()),
        "forecast_rows": int(len(forecasts)),
        "reused_exact_archived_checkpoints": reused_archived,
        "fresh_configurations_run": fresh_runs,
        "validation_folds": ["2012", "2013", "2014"],
    }
    (REPO_ROOT / "logs" / "final_analysis" / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n", encoding="utf-8"
    )
    _log(f"validation complete forecasts={len(forecasts)} configs={forecasts.configuration_id.nunique()}")


if __name__ == "__main__":
    main()
