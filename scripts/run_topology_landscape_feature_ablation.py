from __future__ import annotations

from dataclasses import replace
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
from tda_risk.topology import build_topology_methods


def main() -> None:
    config = default_pipeline_config()
    ablation_topology = replace(
        config.topology,
        window_lengths=(125,),
        alphas=(0.15,),
        feature_modes=(
            "landscape1",
            "landscape_h1_top3_weighted",
            "landscape_h1_top3_concat",
            "landscape_h0h1_top1_concat",
            "landscape_h0h1_top3_concat",
        ),
        input_modes=("portfolio_only",),
        landscape_num_steps=200,
        n_jobs=1,
    )
    ablation_grid = replace(config.grid, euclidean_k=(1000,))
    output = replace(config.output, root_dir=Path("results/topology_landscape_feature_ablation"))
    config = replace(config, topology=ablation_topology, grid=ablation_grid, output=output)

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

    methods = [
        method
        for method in build_baseline_methods(config.grid, state_columns=config.state_columns)
        if method.name in {"rolling_hs", "fhs_ewma", "garch_t"}
        or (method.name == "euclidean_knn" and getattr(method, "k", None) == 1000)
        or (method.name == "random_knn" and getattr(method, "k", None) == 1000)
    ]
    methods.extend(
        build_topology_methods(
            config.topology,
            state_columns=config.state_columns,
            k_values=config.grid.euclidean_k,
            training_end=config.splits.warmup_end,
        )
    )

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
            "pipeline": "topology_landscape_feature_ablation",
            "data_metadata": panel_result.metadata,
            "alpha": 0.15,
            "landscape_num_steps": 200,
            "k": 1000,
        },
    )

    print("Topology landscape feature ablation completed.")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()