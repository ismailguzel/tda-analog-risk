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
from tda_risk.methods import build_baseline_methods
from tda_risk.output import prepare_output_dir, write_backtest_outputs
from tda_risk.topology import _topology_worker_count, build_topology_methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked finalist topology pipeline."
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of worker processes for topology feature precomputation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/topology_pipeline"),
        help="Directory for pipeline outputs.",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="landscape_h1_top3_weighted",
        help="Topology feature mode for the finalist-style validation run.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="State-vs-topology weight alpha for the finalist-style validation run.",
    )
    parser.add_argument(
        "--landscape-steps",
        type=int,
        default=200,
        help="Number of landscape discretization steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = default_pipeline_config()
    finalist_topology = replace(
        config.topology,
        window_lengths=(125,),
        alphas=(args.alpha,),
        feature_modes=(args.feature_mode,),
        input_modes=("portfolio_only",),
        landscape_num_steps=args.landscape_steps,
        landscape_topk_weights=(1.0, 0.5, 0.25),
        n_jobs=max(args.n_jobs, 1),
    )
    finalist_grid = replace(config.grid, euclidean_k=(1000,))
    output = replace(config.output, root_dir=args.output_dir)
    config = replace(config, topology=finalist_topology, grid=finalist_grid, output=output)
    effective_topology_workers = _topology_worker_count(config.topology)

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
            "pipeline": "topology_pipeline",
            "data_metadata": panel_result.metadata,
            "topology_workers_requested": config.topology.n_jobs,
            "topology_workers_effective": effective_topology_workers,
            "finalist_config": {
                "window_length": 125,
                "alpha": args.alpha,
                "feature_mode": args.feature_mode,
                "input_mode": "portfolio_only",
                "landscape_num_steps": args.landscape_steps,
                "k": 1000,
            },
        },
    )

    print("Finalist topology pipeline completed.")
    print(f"Topology feature workers: {config.topology.n_jobs}")
    print(
        f"Finalist config: mode={args.feature_mode}, "
        f"window=125, alpha={args.alpha}, k=1000, steps={args.landscape_steps}"
    )
    print(f"Outputs written to: {output_dir}")
    print(summary.to_string(index=False))
    skip_summary = forecasts.attrs.get("skip_summary")
    if hasattr(skip_summary, "empty") and not skip_summary.empty:
        print("\nSkipped forecasts:")
        print(skip_summary.to_string(index=False))


if __name__ == "__main__":
    main()
