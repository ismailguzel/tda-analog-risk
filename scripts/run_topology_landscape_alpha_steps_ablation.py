from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd

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


FEATURE_MODES = ("landscape1", "landscape_h1_top3_weighted")
LANDSCAPE_STEPS_GRID = (100, 200, 300)
ALPHA_GRID = (0.15, 0.30, 0.50)
K_GRID = (1000,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical topology alpha/steps appendix ablation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/topology_landscape_alpha_steps_ablation"),
        help="Directory for alpha/steps appendix outputs.",
    )
    return parser.parse_args()


def _annotate_config(
    frame: pd.DataFrame,
    *,
    feature_mode: str,
    landscape_steps: int,
    alpha: float,
) -> pd.DataFrame:
    out = frame.copy()
    out["ablation_feature_mode"] = feature_mode
    out["ablation_landscape_steps"] = landscape_steps
    out["ablation_alpha"] = alpha
    return out


def main() -> None:
    args = parse_args()
    config = default_pipeline_config()
    prepare_output_dir(args.output_dir)

    panel_result = build_research_panel(config.data)
    panel = panel_result.panel
    state_panel = build_state_panel(panel)
    state_panel = attach_state_zscores(
        state_panel,
        state_columns=config.state_columns,
        min_history=config.min_zscore_history,
    )

    baseline_grid = replace(config.grid, euclidean_k=K_GRID)
    baseline_methods = [
        method
        for method in build_baseline_methods(baseline_grid, state_columns=config.state_columns)
        if method.name in {"rolling_hs", "fhs_ewma", "garch_t"}
        or (method.name == "euclidean_knn" and getattr(method, "k", None) in K_GRID)
        or (method.name == "random_knn" and getattr(method, "k", None) in K_GRID)
    ]

    all_forecasts: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []

    for feature_mode in FEATURE_MODES:
        for landscape_steps in LANDSCAPE_STEPS_GRID:
            for alpha in ALPHA_GRID:
                topo_cfg = replace(
                    config.topology,
                    window_lengths=(125,),
                    alphas=(alpha,),
                    feature_modes=(feature_mode,),
                    input_modes=("portfolio_only",),
                    landscape_num_steps=landscape_steps,
                    landscape_topk_weights=(1.0, 0.5, 0.25),
                    n_jobs=1,
                )
                run_cfg = replace(config, topology=topo_cfg, grid=baseline_grid)

                clear_method_caches()
                clear_topology_caches()
                methods = list(baseline_methods)
                methods.extend(
                    build_topology_methods(
                        run_cfg.topology,
                        state_columns=run_cfg.state_columns,
                        k_values=run_cfg.grid.euclidean_k,
                        training_end=run_cfg.splits.warmup_end,
                    )
                )

                forecasts = run_backtest(state_panel, methods, run_cfg)
                summary = summarize_backtest(forecasts)
                forecasts.attrs = {}
                summary.attrs = {}
                all_forecasts.append(
                    _annotate_config(
                        forecasts,
                        feature_mode=feature_mode,
                        landscape_steps=landscape_steps,
                        alpha=alpha,
                    )
                )
                all_summary.append(
                    _annotate_config(
                        summary,
                        feature_mode=feature_mode,
                        landscape_steps=landscape_steps,
                        alpha=alpha,
                    )
                )
                print(f"[done] mode={feature_mode} steps={landscape_steps} alpha={alpha:.2f}")

    forecasts_out = pd.concat(all_forecasts, ignore_index=True)
    summary_out = pd.concat(all_summary, ignore_index=True)
    forecasts_out.to_csv(args.output_dir / "forecasts.csv", index=False)
    summary_out.to_csv(args.output_dir / "summary.csv", index=False)

    focus = summary_out[
        (summary_out["split"] == "test")
        & (summary_out["method"].isin({"topology_knn", "topology_placebo_knn"}))
    ].sort_values(
        [
            "ablation_feature_mode",
            "ablation_landscape_steps",
            "ablation_alpha",
            "method",
        ]
    )
    focus.to_csv(args.output_dir / "summary_topology_focus.csv", index=False)

    (args.output_dir / "run_notes.txt").write_text(
        "Canonical appendix ablation: feature_mode x landscape_steps x alpha\n",
        encoding="utf-8",
    )

    print("Topology alpha/steps ablation completed.")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()