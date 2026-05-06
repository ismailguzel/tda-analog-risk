from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tda_risk.backtest import run_backtest, summarize_backtest
from tda_risk.config import default_pipeline_config
from tda_risk.data import build_research_panel
from tda_risk.features import attach_state_zscores, build_state_panel
from tda_risk.methods import build_baseline_methods
from tda_risk.output import prepare_output_dir, write_backtest_outputs


def main() -> None:
    config = default_pipeline_config()
    output_dir = config.output.root_dir
    prepare_output_dir(output_dir)

    panel_result = build_research_panel(config.data)
    panel = panel_result.panel
    state_panel = build_state_panel(panel)
    state_panel = attach_state_zscores(
        state_panel,
        state_columns=config.state_columns,
        min_history=config.min_zscore_history,
    )

    methods = build_baseline_methods(config.grid, state_columns=config.state_columns)
    forecasts = run_backtest(state_panel, methods, config)
    summary = summarize_backtest(forecasts)

    write_backtest_outputs(
        output_dir=output_dir,
        output_config=config.output,
        panel=panel,
        state_panel=state_panel,
        forecasts=forecasts,
        summary=summary,
        run_metadata={
            "pipeline": "baseline_pipeline",
            "data_metadata": panel_result.metadata,
        },
    )

    print("Baseline pipeline completed.")
    print(f"Outputs written to: {output_dir}")
    print(summary.to_string(index=False))
    skip_summary = forecasts.attrs.get("skip_summary")
    if hasattr(skip_summary, "empty") and not skip_summary.empty:
        print("\nSkipped forecasts:")
        print(skip_summary.to_string(index=False))


if __name__ == "__main__":
    main()
