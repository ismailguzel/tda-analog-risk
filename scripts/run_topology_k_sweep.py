from __future__ import annotations

import argparse
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
from tda_risk.methods import build_baseline_methods, clear_method_caches
from tda_risk.output import prepare_output_dir, write_backtest_outputs
from tda_risk.topology import build_topology_methods, clear_topology_caches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical topology K-sweep appendix experiment."
    )
    parser.add_argument(
        "--k-grid",
        type=str,
        default="100,250,500,750,1000",
        help="Comma-separated K values for the appendix sweep.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/topology_k_sweep"),
        help="Root directory for per-k outputs.",
    )
    return parser.parse_args()


def _parse_k_grid(raw: str) -> tuple[int, ...]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("k-grid cannot be empty.")
    if min(values) <= 0:
        raise ValueError("k values must be strictly positive.")
    return tuple(values)


def main() -> None:
    args = parse_args()
    k_grid = _parse_k_grid(args.k_grid)

    config = default_pipeline_config()
    panel_result = build_research_panel(config.data)
    panel = panel_result.panel
    state_panel = build_state_panel(panel)
    state_panel = attach_state_zscores(
        state_panel,
        state_columns=config.state_columns,
        min_history=config.min_zscore_history,
    )

    for k_value in k_grid:
        output_dir = args.output_root / f"k{k_value}"
        prepare_output_dir(output_dir)

        run_cfg = replace(
            config,
            grid=replace(
                config.grid,
                euclidean_k=(k_value,),
                analogue_window_lengths=(60, 125),
            ),
            topology=replace(
                config.topology,
                window_lengths=(125,),
                alphas=(0.15,),
                feature_modes=("landscape1",),
                input_modes=("portfolio_only",),
                landscape_num_steps=200,
                landscape_topk_weights=(1.0, 0.5, 0.25),
                n_jobs=1,
            ),
            output=replace(config.output, root_dir=output_dir),
        )

        methods = build_baseline_methods(run_cfg.grid, state_columns=run_cfg.state_columns)
        methods.extend(
            build_topology_methods(
                run_cfg.topology,
                state_columns=run_cfg.state_columns,
                k_values=run_cfg.grid.euclidean_k,
                training_end=run_cfg.splits.warmup_end,
            )
        )

        clear_method_caches()
        clear_topology_caches()
        forecasts = run_backtest(state_panel, methods, run_cfg)
        summary = summarize_backtest(forecasts)
        write_backtest_outputs(
            output_dir=output_dir,
            output_config=run_cfg.output,
            panel=panel,
            state_panel=state_panel,
            forecasts=forecasts,
            summary=summary,
            run_metadata={
                "pipeline": "topology_k_sweep",
                "data_metadata": panel_result.metadata,
                "k_value": k_value,
                "feature_mode": "landscape1",
                "alpha": 0.15,
                "window_length": 125,
                "landscape_num_steps": 200,
            },
        )
        print(f"[done] k={k_value} -> {output_dir}")

    print("Topology K-sweep completed.")
    print(f"Outputs written under: {args.output_root}")


if __name__ == "__main__":
    main()