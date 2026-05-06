from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys

import numpy as np
from dtaidistance import dtw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tda_risk.backtest import run_backtest, summarize_backtest
from tda_risk.config import RiskConfig, TopologyConfig, default_pipeline_config
from tda_risk.data import build_research_panel
from tda_risk.features import attach_state_zscores, build_state_panel
from tda_risk.methods import (
    BaseMethod,
    MethodForecast,
    _scenario_losses_from_positions,
    _select_top_positions,
    _window_knn_inputs,
    empirical_var_es,
)
from tda_risk.output import prepare_output_dir
from tda_risk.topology import (
    _get_topology_feature_matrix,
    _select_embedding_parameters,
    _standardize_topology_features,
    clear_topology_caches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation: topology distance blended with DTW distance."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/topology_dtw_hybrid_ablation"),
        help="Directory for ablation outputs.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="0.0,0.1,0.25,0.5,0.75,0.9,1.0",
        help="Comma-separated topology weights in [0,1].",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1000,
        help="Neighbor count k for scenario extraction.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=125,
        help="Window length used for both topology and DTW windows.",
    )
    parser.add_argument(
        "--dtw-band",
        type=int,
        default=10,
        help="Sakoe-Chiba band for DTW.",
    )
    return parser.parse_args()


def _parse_weights(raw: str) -> tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    if not vals:
        raise ValueError("Weight grid is empty.")
    for v in vals:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Invalid weight {v}. Must be in [0,1].")
    return vals


@dataclass(frozen=True)
class TopologyDTWHybridKNN(BaseMethod):
    k: int
    max_k: int
    window_length: int
    input_mode: str
    feature_mode: str
    topology_weight: float
    dtw_band: int
    topology: TopologyConfig
    training_end: str
    name: str = "topology_dtw_hybrid_knn"

    def forecast(self, data, position: int, risk: RiskConfig) -> MethodForecast:
        tau, embedding_dimension = _select_embedding_parameters(
            data,
            self.topology,
            self.training_end,
            self.input_mode,
        )
        feature_matrix = _get_topology_feature_matrix(
            data,
            window_length=self.window_length,
            input_mode=self.input_mode,
            feature_mode=self.feature_mode,
            topology=self.topology,
            training_end=self.training_end,
            use_null=False,
        )
        current_feature = feature_matrix[position]
        if not np.isfinite(current_feature).all():
            raise ValueError("Current topology representation is not available.")

        current_window, history_windows, candidate_positions = _window_knn_inputs(
            data,
            position,
            self.window_length,
        )
        feature_mask = np.isfinite(feature_matrix[candidate_positions]).all(axis=1)
        if not feature_mask.any():
            raise ValueError("No valid candidates after topology feature filtering.")

        candidate_positions = candidate_positions[feature_mask]
        history_windows = history_windows[feature_mask]
        history_features = feature_matrix[candidate_positions]

        # DTW distances on trailing return windows.
        stacked = np.vstack([current_window, history_windows]).astype(np.double, copy=False)
        dtw_distances = np.asarray(
            dtw.distance_matrix_fast(
                stacked,
                block=((0, 1), (1, stacked.shape[0])),
                compact=True,
                window=self.dtw_band,
                parallel=False,
                use_pruning=True,
            ),
            dtype=float,
        )

        # Topology distances on standardized persistence-landscape features.
        history_features, current_feature = _standardize_topology_features(
            history_features,
            current_feature,
        )
        topo_distances = np.linalg.norm(history_features - current_feature, axis=1)
        combined = self.topology_weight * topo_distances + (1.0 - self.topology_weight) * dtw_distances

        top_positions = _select_top_positions(candidate_positions, combined, self.max_k)
        scenario_losses = _scenario_losses_from_positions(data, top_positions[: self.k])
        var, es = empirical_var_es(scenario_losses, risk.var_confidence, risk.es_confidence)
        return MethodForecast(
            self.name,
            {
                "k": self.k,
                "window_length": self.window_length,
                "feature_mode": self.feature_mode,
                "topology_input_mode": self.input_mode,
                "tau": tau,
                "embedding_dimension": embedding_dimension,
                "topology_weight": self.topology_weight,
                "dtw_band": self.dtw_band,
            },
            scenario_losses.size,
            var,
            es,
        )


def main() -> None:
    args = parse_args()
    cfg = default_pipeline_config()
    output_dir = args.output_dir
    prepare_output_dir(output_dir)

    panel = build_research_panel(cfg.data).panel
    state_panel = attach_state_zscores(
        build_state_panel(panel),
        state_columns=cfg.state_columns,
        min_history=cfg.min_zscore_history,
    )

    # Retained topology setup, but without state-vector blending.
    topo_cfg = replace(
        cfg.topology,
        window_lengths=(args.window_length,),
        alphas=(0.0,),
        feature_modes=("landscape_h1_top3_weighted",),
        input_modes=("portfolio_only",),
        landscape_num_steps=200,
        n_jobs=1,
    )
    run_cfg = replace(cfg, topology=topo_cfg)

    k = args.k
    max_k = args.k
    dtw_band = args.dtw_band
    topology_weights = _parse_weights(args.weights)
    methods: list[BaseMethod] = []
    clear_topology_caches()
    for w in topology_weights:
        methods.append(
            TopologyDTWHybridKNN(
                k=k,
                max_k=max_k,
                window_length=args.window_length,
                input_mode="portfolio_only",
                feature_mode="landscape_h1_top3_weighted",
                topology_weight=w,
                dtw_band=dtw_band,
                topology=run_cfg.topology,
                training_end=run_cfg.splits.warmup_end,
            )
        )

    forecasts = run_backtest(state_panel, methods, run_cfg)
    summary = summarize_backtest(forecasts)
    forecasts.to_csv(output_dir / "forecasts.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    # Keep legacy method-level summary, but also write config-level summaries
    # so different topology_weight values are not collapsed into one row.
    detail_cols = ["split", "topology_weight", "dtw_band", "k"]
    summary_by_config = (
        forecasts.groupby(detail_cols, dropna=False)
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
        .sort_values(detail_cols)
    )
    summary_by_config.to_csv(output_dir / "summary_by_config.csv", index=False)

    test_focus = (
        summary_by_config[summary_by_config["split"] == "test"]
        .copy()
        .sort_values("exceedance_rate_99")
    )
    test_focus.to_csv(output_dir / "summary_test_focus_by_config.csv", index=False)
    print(f"DTW-topology hybrid ablation written to: {output_dir}")
    print(test_focus.to_string(index=False))


if __name__ == "__main__":
    main()
