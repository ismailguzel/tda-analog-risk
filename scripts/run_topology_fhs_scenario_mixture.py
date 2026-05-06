from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tda_risk.backtest import infer_split
from tda_risk.config import default_pipeline_config
from tda_risk.data import build_research_panel
from tda_risk.features import attach_state_zscores, build_state_panel
from tda_risk.methods import _scenario_losses_from_positions, ewma_sigma
from tda_risk.topology import _topology_top_positions, clear_topology_caches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run scenario-level topology+FHS mixture: merge scenario distributions "
            "per date, select weight on validation, evaluate on test."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/topology_fhs_scenario_mixture"),
        help="Directory for scenario-mixture outputs.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0",
        help="Comma-separated topology mixture weights.",
    )
    parser.add_argument(
        "--fhs-lambdas",
        type=str,
        default="0.94,0.97",
        help="Comma-separated FHS lambdas.",
    )
    parser.add_argument(
        "--selection-objective",
        type=str,
        choices=("fz", "coverage_then_fz", "weighted_combo"),
        default="coverage_then_fz",
        help="Objective for validation-based model selection.",
    )
    parser.add_argument(
        "--target-var-exceedance",
        type=float,
        default=0.01,
        help="Target exceedance level used by non-fz objectives.",
    )
    parser.add_argument(
        "--coverage-weight",
        type=float,
        default=0.7,
        help="Coverage weight for weighted_combo objective.",
    )
    parser.add_argument(
        "--topology-k",
        type=int,
        default=1000,
        help="Number of topology neighbor scenarios.",
    )
    parser.add_argument(
        "--topology-window-length",
        type=int,
        default=125,
        help="Topology window length.",
    )
    parser.add_argument(
        "--topology-alpha",
        type=float,
        default=0.0,
        help="Topology alpha.",
    )
    parser.add_argument(
        "--topology-feature-mode",
        type=str,
        default="landscape_h1_top3_weighted",
        help="Topology feature mode.",
    )
    parser.add_argument(
        "--topology-input-mode",
        type=str,
        default="portfolio_only",
        help="Topology input mode.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Topology feature precompute workers.",
    )
    return parser.parse_args()


def _parse_float_grid(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Grid cannot be empty.")
    return sorted(set(values))


def _fz_style_score(
    losses: np.ndarray,
    var_values: np.ndarray,
    es_values: np.ndarray,
    alpha_var: float = 0.99,
    alpha_es: float = 0.975,
) -> np.ndarray:
    var_term = np.where(
        losses >= var_values,
        alpha_var * (losses - var_values),
        (1.0 - alpha_var) * (var_values - losses),
    )
    es_term = np.where(
        losses >= es_values,
        losses - es_values,
        alpha_es * (es_values - losses),
    )
    return var_term + es_term


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cum = np.cumsum(sorted_weights)
    if cum[-1] <= 0.0:
        raise ValueError("Total weight must be positive.")
    target = q * cum[-1]
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(max(idx, 0), sorted_values.size - 1)
    return float(sorted_values[idx])


def _weighted_var_es(
    losses: np.ndarray,
    weights: np.ndarray,
    var_confidence: float,
    es_confidence: float,
) -> tuple[float, float]:
    clean_mask = np.isfinite(losses) & np.isfinite(weights) & (weights > 0.0)
    clean_losses = losses[clean_mask]
    clean_weights = weights[clean_mask]
    if clean_losses.size == 0:
        raise ValueError("No finite weighted scenario losses.")
    clean_weights = clean_weights / clean_weights.sum()
    var_value = _weighted_quantile(clean_losses, clean_weights, var_confidence)
    es_threshold = _weighted_quantile(clean_losses, clean_weights, es_confidence)
    tail_mask = clean_losses >= es_threshold
    if not np.any(tail_mask):
        es_value = es_threshold
    else:
        tail_losses = clean_losses[tail_mask]
        tail_weights = clean_weights[tail_mask]
        es_value = float(np.sum(tail_weights * tail_losses) / np.sum(tail_weights))
    return var_value, es_value


def _selection_score(
    objective: str,
    *,
    mean_fz: float,
    exceedance_99: float,
    target_exceedance: float,
    coverage_weight: float,
) -> float:
    coverage_gap = abs(exceedance_99 - target_exceedance)
    if objective == "fz":
        return mean_fz
    if objective == "coverage_then_fz":
        return coverage_gap * 1_000.0 + mean_fz
    if objective == "weighted_combo":
        return coverage_weight * coverage_gap + (1.0 - coverage_weight) * mean_fz
    raise ValueError(f"Unknown objective: {objective}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (0.0 <= args.target_var_exceedance <= 1.0):
        raise ValueError("--target-var-exceedance must be in [0,1].")
    if not (0.0 <= args.coverage_weight <= 1.0):
        raise ValueError("--coverage-weight must be in [0,1].")

    weights_grid = _parse_float_grid(args.weights)
    lambda_grid = _parse_float_grid(args.fhs_lambdas)
    config = default_pipeline_config()
    topo_cfg = replace(
        config.topology,
        window_lengths=(args.topology_window_length,),
        alphas=(args.topology_alpha,),
        feature_modes=(args.topology_feature_mode,),
        input_modes=(args.topology_input_mode,),
        n_jobs=max(args.n_jobs, 1),
    )
    config = replace(config, topology=topo_cfg)

    panel = build_research_panel(config.data).panel
    state_panel = attach_state_zscores(
        build_state_panel(panel),
        state_columns=config.state_columns,
        min_history=config.min_zscore_history,
    )
    losses_all = state_panel["portfolio_loss"].to_numpy(dtype=float)
    returns_all = state_panel["portfolio_return"].to_numpy(dtype=float)

    clear_topology_caches()
    start_date = pd.Timestamp(config.splits.validation_start)
    start_positions = np.flatnonzero(state_panel.index >= start_date)
    if start_positions.size == 0:
        raise RuntimeError("Validation start date is outside panel range.")
    start_pos = int(start_positions[0])

    # Precompute topology and FHS scenario sets per position.
    base_rows: list[dict[str, object]] = []
    for position in range(start_pos, len(state_panel) - 1):
        realized_loss = float(losses_all[position + 1])
        if not np.isfinite(realized_loss):
            continue
        forecast_date = state_panel.index[position]
        split = infer_split(forecast_date, config)
        if split not in {"validation", "test"}:
            continue

        try:
            top_positions = _topology_top_positions(
                state_panel,
                position,
                state_columns=config.state_columns,
                window_length=args.topology_window_length,
                input_mode=args.topology_input_mode,
                feature_mode=args.topology_feature_mode,
                alpha=args.topology_alpha,
                max_k=args.topology_k,
                topology=config.topology,
                training_end=config.splits.warmup_end,
                use_null=False,
            )
            topo_losses = _scenario_losses_from_positions(state_panel, top_positions[: args.topology_k])
        except ValueError:
            continue
        if topo_losses.size == 0:
            continue

        returns_hist = returns_all[: position + 1]
        returns_hist = returns_hist[np.isfinite(returns_hist)]
        if returns_hist.size < 100:
            continue

        for lambda_ in lambda_grid:
            sigma = ewma_sigma(returns_hist, lambda_)
            residuals = returns_hist / sigma
            sigma_next = float(
                np.sqrt(lambda_ * sigma[-1] ** 2 + (1.0 - lambda_) * returns_hist[-1] ** 2)
            )
            fhs_losses = -(sigma_next * residuals)
            fhs_losses = fhs_losses[np.isfinite(fhs_losses)]
            if fhs_losses.size == 0:
                continue

            base_rows.append(
                {
                    "position": position,
                    "forecast_date": forecast_date,
                    "realized_date": state_panel.index[position + 1],
                    "split": split,
                    "realized_loss": realized_loss,
                    "lambda": lambda_,
                    "topo_losses": topo_losses,
                    "fhs_losses": fhs_losses,
                }
            )

    if not base_rows:
        raise RuntimeError("No valid rows for scenario mixture.")

    # Build candidate forecasts for each (lambda, weight).
    candidate_rows: list[dict[str, object]] = []
    for row in base_rows:
        topo_losses = row["topo_losses"]  # type: ignore[assignment]
        fhs_losses = row["fhs_losses"]  # type: ignore[assignment]
        for w_topo in weights_grid:
            topo_count = topo_losses.size
            fhs_count = fhs_losses.size
            mix_losses = np.concatenate([topo_losses, fhs_losses])
            mix_weights = np.concatenate(
                [
                    np.full(topo_count, w_topo / topo_count, dtype=float),
                    np.full(fhs_count, (1.0 - w_topo) / fhs_count, dtype=float),
                ]
            )
            var99, es975 = _weighted_var_es(
                mix_losses,
                mix_weights,
                var_confidence=config.risk.var_confidence,
                es_confidence=config.risk.es_confidence,
            )
            realized_loss = float(row["realized_loss"])
            candidate_rows.append(
                {
                    "forecast_date": row["forecast_date"],
                    "realized_date": row["realized_date"],
                    "split": row["split"],
                    "method": "hybrid_topology_fhs_scenario",
                    "lambda": row["lambda"],
                    "hybrid_weight_topology": w_topo,
                    "scenario_count_topology": topo_count,
                    "scenario_count_fhs": fhs_count,
                    "scenario_count_total": topo_count + fhs_count,
                    "VaR_99": var99,
                    "ES_975": es975,
                    "realized_loss": realized_loss,
                    "var_exceedance_99": float(realized_loss > var99),
                    "es_exceedance_975": float(realized_loss > es975),
                    "fz_style_score": float(
                        _fz_style_score(
                            np.array([realized_loss], dtype=float),
                            np.array([var99], dtype=float),
                            np.array([es975], dtype=float),
                        )[0]
                    ),
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    validation = candidates[candidates["split"] == "validation"].copy()
    test = candidates[candidates["split"] == "test"].copy()

    scan = (
        validation.groupby(["lambda", "hybrid_weight_topology"], dropna=False)
        .agg(
            n_validation_forecasts=("realized_date", "size"),
            validation_fz_style_mean=("fz_style_score", "mean"),
            validation_exceedance_rate_99=("var_exceedance_99", "mean"),
            validation_exceedance_rate_es_975=("es_exceedance_975", "mean"),
        )
        .reset_index()
    )
    scan["validation_coverage_gap_99"] = (
        scan["validation_exceedance_rate_99"] - args.target_var_exceedance
    ).abs()
    scan["validation_selection_score"] = scan.apply(
        lambda r: _selection_score(
            args.selection_objective,
            mean_fz=float(r["validation_fz_style_mean"]),
            exceedance_99=float(r["validation_exceedance_rate_99"]),
            target_exceedance=args.target_var_exceedance,
            coverage_weight=args.coverage_weight,
        ),
        axis=1,
    )
    scan = scan.sort_values("validation_selection_score")
    scan.to_csv(output_dir / "validation_weight_scan.csv", index=False)

    best = scan.iloc[0]
    best_lambda = float(best["lambda"])
    best_weight = float(best["hybrid_weight_topology"])

    selected_validation = validation[
        (validation["lambda"] == best_lambda)
        & (validation["hybrid_weight_topology"] == best_weight)
    ].copy()
    selected_test = test[
        (test["lambda"] == best_lambda) & (test["hybrid_weight_topology"] == best_weight)
    ].copy()
    hybrid_forecasts = pd.concat([selected_validation, selected_test], ignore_index=True).sort_values(
        "realized_date"
    )
    hybrid_forecasts.to_csv(output_dir / "hybrid_forecasts.csv", index=False)

    summary = (
        hybrid_forecasts.groupby("split", dropna=False)
        .agg(
            n_forecasts=("realized_date", "size"),
            mean_var_99=("VaR_99", "mean"),
            mean_es_975=("ES_975", "mean"),
            exceedance_rate_99=("var_exceedance_99", "mean"),
            exceedance_rate_es_975=("es_exceedance_975", "mean"),
            mean_realized_loss=("realized_loss", "mean"),
            mean_fz_style=("fz_style_score", "mean"),
        )
        .reset_index()
        .sort_values("split")
    )
    summary.to_csv(output_dir / "hybrid_summary.csv", index=False)

    selected = pd.DataFrame(
        [
            {
                "selection_objective": args.selection_objective,
                "target_var_exceedance": args.target_var_exceedance,
                "coverage_weight": args.coverage_weight,
                "selected_lambda": best_lambda,
                "selected_hybrid_weight_topology": best_weight,
                "validation_selection_score": float(best["validation_selection_score"]),
                "validation_mean_fz_style_score": float(best["validation_fz_style_mean"]),
                "validation_exceedance_rate_99": float(best["validation_exceedance_rate_99"]),
                "validation_exceedance_rate_es_975": float(best["validation_exceedance_rate_es_975"]),
                "n_validation_forecasts": int(selected_validation.shape[0]),
                "n_test_forecasts": int(selected_test.shape[0]),
                "topology_k": args.topology_k,
                "topology_window_length": args.topology_window_length,
                "topology_feature_mode": args.topology_feature_mode,
                "topology_input_mode": args.topology_input_mode,
                "topology_alpha": args.topology_alpha,
            }
        ]
    )
    selected.to_csv(output_dir / "selected_hybrid_config.csv", index=False)

    print(f"Scenario-mixture outputs written to: {output_dir}")
    print(selected.to_string(index=False))
    print("\nTop validation candidates:")
    print(scan.head(10).to_string(index=False))
    print("\nTest summary:")
    print(summary[summary["split"] == "test"].to_string(index=False))


if __name__ == "__main__":
    main()
