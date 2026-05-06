from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .methods import BaseMethod


def run_backtest(
    data: pd.DataFrame,
    methods: list[BaseMethod],
    config: PipelineConfig,
) -> pd.DataFrame:
    results: list[dict[str, object]] = []
    skip_events: list[dict[str, object]] = []

    start_date = pd.Timestamp(config.splits.validation_start)
    start_positions = np.flatnonzero(data.index >= start_date)
    if start_positions.size == 0:
        raise RuntimeError("Validation start date is outside the panel range.")
    start_position = int(start_positions[0])

    for position in range(start_position, len(data) - 1):
        realized_loss = data["portfolio_loss"].iloc[position + 1]
        if not np.isfinite(realized_loss):
            continue

        forecast_date = data.index[position]
        for method in methods:
            try:
                forecast = method.forecast(data, position, config.risk)
            except ValueError as exc:
                skip_events.append(
                    {
                        "forecast_date": forecast_date,
                        "realized_date": data.index[position + 1],
                        "split": infer_split(forecast_date, config),
                        "method": method.name,
                        "error": str(exc),
                    }
                )
                continue

            row = {
                "forecast_date": forecast_date,
                "realized_date": data.index[position + 1],
                "split": infer_split(forecast_date, config),
                "method": forecast.method,
                "scenario_count": forecast.scenario_count,
                "VaR_99": forecast.var,
                "ES_975": forecast.es,
                "realized_loss": realized_loss,
                "var_exceedance_99": float(realized_loss > forecast.var),
                "es_exceedance_975": float(realized_loss > forecast.es),
            }
            for key, value in forecast.params.items():
                row[key] = value
            results.append(row)

    if not results:
        raise RuntimeError("Backtest did not produce any forecasts.")

    forecasts = pd.DataFrame(results)
    skip_frame = pd.DataFrame(
        skip_events,
        columns=["forecast_date", "realized_date", "split", "method", "error"],
    )
    forecasts.attrs["skip_events"] = skip_frame
    if skip_frame.empty:
        forecasts.attrs["skip_summary"] = pd.DataFrame(
            columns=["split", "method", "error", "n_skips"]
        )
    else:
        forecasts.attrs["skip_summary"] = (
            skip_frame.groupby(["split", "method", "error"], dropna=False)
            .size()
            .rename("n_skips")
            .reset_index()
            .sort_values(["split", "method", "n_skips"], ascending=[True, True, False])
        )
    return forecasts


def summarize_backtest(forecasts: pd.DataFrame) -> pd.DataFrame:
    return (
        forecasts.groupby(["split", "method"], dropna=False)
        .agg(
            n_forecasts=("forecast_date", "size"),
            mean_scenarios=("scenario_count", "mean"),
            mean_var_99=("VaR_99", "mean"),
            mean_es_975=("ES_975", "mean"),
            exceedance_rate_99=("var_exceedance_99", "mean"),
            exceedance_rate_es_975=("es_exceedance_975", "mean"),
            mean_realized_loss=("realized_loss", "mean"),
        )
        .reset_index()
        .sort_values(["split", "method"])
    )


def infer_split(date: pd.Timestamp, config: PipelineConfig) -> str:
    if date < pd.Timestamp(config.splits.validation_start):
        return "pre_validation"
    if date <= pd.Timestamp(config.splits.validation_end):
        return "validation"
    if date >= pd.Timestamp(config.splits.test_start):
        return "test"
    return "transition"
