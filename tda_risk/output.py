from __future__ import annotations

from pathlib import Path
import json
import shutil

import pandas as pd

from .config import OutputConfig


def prepare_output_dir(output_dir: Path) -> None:
    """Remove stale artifacts before writing a fresh experiment run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def write_backtest_outputs(
    *,
    output_dir: Path,
    output_config: OutputConfig,
    panel: pd.DataFrame,
    state_panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    summary: pd.DataFrame,
    run_metadata: dict[str, object] | None = None,
) -> None:
    panel.to_csv(output_dir / output_config.panel_file, index_label="date")
    state_panel.to_csv(output_dir / output_config.state_file, index_label="date")
    forecasts.to_csv(output_dir / output_config.forecast_file, index=False)
    summary.to_csv(output_dir / output_config.summary_file, index=False)

    metadata = dict(run_metadata or {})
    if not panel.empty:
        metadata.setdefault("panel_start_date", panel.index.min().date().isoformat())
        metadata.setdefault("panel_end_date", panel.index.max().date().isoformat())
        metadata.setdefault("panel_observation_count", int(panel.shape[0]))
    if not forecasts.empty:
        metadata.setdefault(
            "forecast_start_date",
            pd.to_datetime(forecasts["forecast_date"], utc=False).min().date().isoformat(),
        )
        metadata.setdefault(
            "forecast_end_date",
            pd.to_datetime(forecasts["forecast_date"], utc=False).max().date().isoformat(),
        )
        metadata.setdefault(
            "realized_end_date",
            pd.to_datetime(forecasts["realized_date"], utc=False).max().date().isoformat(),
        )
        metadata.setdefault("forecast_row_count", int(forecasts.shape[0]))
    if metadata:
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    skip_events = forecasts.attrs.get("skip_events")
    if isinstance(skip_events, pd.DataFrame):
        skip_events.to_csv(output_dir / output_config.skip_events_file, index=False)

    skip_summary = forecasts.attrs.get("skip_summary")
    if isinstance(skip_summary, pd.DataFrame):
        skip_summary.to_csv(output_dir / output_config.skip_summary_file, index=False)
