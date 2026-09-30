"""Check distance cutoffs for the complete-IID uniform-subset shortcut.

This diagnostic reads the frozen panel and computes the locked topology
feature catalog in memory. It does not generate forecasts, null draws, or
any scientific result used for model selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tda_risk.config import TopologyConfig, default_pipeline_config
from tda_risk.features import attach_state_zscores, build_state_panel
from tda_risk.methods import eligible_candidate_positions
from tda_risk.null_controls import eligible_history
from tda_risk.topology import _get_topology_feature_matrix, clear_topology_caches


DEFAULT_OUTPUT = ROOT / "results" / "final_diagnostics" / "cutoff_tie_diagnostic.csv"
MACHINE_TOLERANCE_MULTIPLIER = 100.0


def load_state_panel() -> pd.DataFrame:
    panel = pd.read_csv(ROOT / "data" / "panel_published.csv.gz", index_col=0, parse_dates=True)
    config = default_pipeline_config()
    return attach_state_zscores(build_state_panel(panel), config.state_columns, min_history=60)


def build_locked_features(data: pd.DataFrame) -> np.ndarray:
    config = default_pipeline_config()
    topology = TopologyConfig(
        window_lengths=(250,),
        feature_modes=("landscape_h1_top3_weighted",),
        input_modes=("portfolio_only",),
        alphas=(0.0,),
        n_jobs=1,
    )
    clear_topology_caches()
    return _get_topology_feature_matrix(
        data,
        window_length=250,
        input_mode="portfolio_only",
        feature_mode="landscape_h1_top3_weighted",
        topology=topology,
        training_end=config.splits.warmup_end,
        use_null=False,
    )


def _candidate_positions(data: pd.DataFrame, features: np.ndarray, position: int, radius: int) -> np.ndarray:
    losses = data["portfolio_loss"].to_numpy(dtype=float)
    candidates = eligible_history(position, len(data), radius, features, losses)
    boundary = eligible_candidate_positions(position, len(data), radius)
    expected = boundary[
        np.isfinite(features[boundary]).all(axis=1)
        & np.isfinite(losses[boundary + 1])
    ]
    np.testing.assert_array_equal(candidates, expected)
    return candidates


def _standardized_distances(features: np.ndarray, position: int, candidates: np.ndarray) -> np.ndarray:
    catalog = np.asarray(features[candidates], dtype=float)
    query = np.asarray(features[position], dtype=float)
    mean = catalog.mean(axis=0)
    scale = catalog.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return np.linalg.norm((catalog - mean) / scale - (query - mean) / scale, axis=1)


def scan_catalog(
    data: pd.DataFrame,
    features: np.ndarray,
    *,
    sample_type: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    radius: int,
    k: int,
) -> dict[str, object]:
    exact = 0
    near = 0
    gaps: list[float] = []
    n_queries = 0
    for position, date in enumerate(data.index[:-1]):
        if not (start <= date <= end) or not np.isfinite(features[position]).all():
            continue
        candidates = _candidate_positions(data, features, position, radius)
        if candidates.size <= k:
            continue
        distances = np.sort(_standardized_distances(features, position, candidates))
        gap = float(distances[k] - distances[k - 1])
        scale = max(1.0, abs(float(distances[k - 1])), abs(float(distances[k])))
        tolerance = MACHINE_TOLERANCE_MULTIPLIER * np.finfo(float).eps * scale
        exact += int(distances[k] == distances[k - 1])
        near += int(abs(gap) <= tolerance)
        gaps.append(gap)
        n_queries += 1
    return {
        "sample_type": sample_type,
        "exclusion_radius": radius,
        "window_length": 250,
        "k": k,
        "number_queries": n_queries,
        "exact_tie_count": exact,
        "near_tie_count": near,
        "exact_tie_rate": exact / n_queries if n_queries else np.nan,
        "near_tie_rate": near / n_queries if n_queries else np.nan,
        "min_cutoff_gap": min(gaps) if gaps else np.nan,
    }


def run(output: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    data = load_state_panel()
    features = build_locked_features(data)
    config = default_pipeline_config()
    rows: list[dict[str, object]] = []
    rows.extend(
        scan_catalog(
            data,
            features,
            sample_type="locked_evaluation",
            start=pd.Timestamp(config.splits.test_start),
            end=pd.Timestamp("2026-04-30"),
            radius=radius,
            k=100,
        )
        for radius in (0, 125, 250)
    )
    rows.extend(
        scan_catalog(
            data,
            features,
            sample_type="validation_k_sensitivity",
            start=pd.Timestamp(config.splits.validation_start),
            end=pd.Timestamp(config.splits.validation_end),
            radius=radius,
            k=k,
        )
        for radius in (0, 125)
        for k in (100, 250, 500, 750, 1000)
    )
    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = run(args.output)
    print(frame.to_string(index=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
