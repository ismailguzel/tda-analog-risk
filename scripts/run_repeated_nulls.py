"""Run two implementations of the configured history-only complete-IID null."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tda_risk.config import TopologyConfig, default_pipeline_config
from tda_risk.features import attach_state_zscores, build_state_panel
from tda_risk.methods import eligible_candidate_positions
from tda_risk.null_controls import (
    eligible_history,
    topology_history_permutation,
    uniform_random_retrieval,
)
from tda_risk.scoring import fz0_score, pinball_score
from tda_risk.topology import _get_topology_feature_matrix, clear_topology_caches


OUT = ROOT / "results" / "recovery_pilot"
SEEDS = json.loads((ROOT / "revision_config" / "random_seed_registry.json").read_text())
STATE_COLUMNS = default_pipeline_config().state_columns
K = 100
LEVEL = 0.99
ES_LEVEL = 0.975


def load_state_panel() -> pd.DataFrame:
    panel = pd.read_csv(ROOT / "data" / "panel_published.csv.gz", index_col=0, parse_dates=True)
    return attach_state_zscores(build_state_panel(panel), STATE_COLUMNS, min_history=60)


def _candidate_positions(
    data: pd.DataFrame,
    features: np.ndarray,
    position: int,
    radius: int,
) -> np.ndarray:
    losses = data["portfolio_loss"].to_numpy(float)
    candidates = eligible_history(position, len(data), radius, features, losses)
    # Keep the exclusion boundary explicit and independently checked.
    boundary = eligible_candidate_positions(position, len(data), radius)
    expected = boundary[
        np.isfinite(features[boundary]).all(axis=1)
        & np.isfinite(losses[boundary + 1])
    ]
    np.testing.assert_array_equal(candidates, expected)
    return candidates


def _value_checksum(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def _load_or_build_feature_matrix(
    data: pd.DataFrame, window_length: int = 250, cache_path: Path | None = None
) -> np.ndarray:
    cache_path = cache_path or (OUT / f"topology_feature_matrix_l{window_length}.npy")
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(data):
            return cached
    config = default_pipeline_config()
    topology = TopologyConfig(
        window_lengths=(window_length,),
        feature_modes=("landscape_h1_top3_weighted",),
        input_modes=("portfolio_only",),
        alphas=(0.0,),
        n_jobs=1,
    )
    clear_topology_caches()
    features = _get_topology_feature_matrix(
        data,
        window_length=window_length,
        input_mode="portfolio_only",
        feature_mode="landscape_h1_top3_weighted",
        topology=topology,
        training_end=config.splits.warmup_end,
        use_null=False,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    return features


def run(
    draws: int,
    representative: int | None,
    control: str,
    seed_root: int,
    *,
    split: str = "evaluation",
    k: int = K,
    window_length: int = 250,
    cache_path: Path | None = None,
    fast_permutation: bool = False,
    radii: tuple[int, ...] = (0, 125, 250),
) -> pd.DataFrame:
    data = load_state_panel()
    features = _load_or_build_feature_matrix(data, window_length=window_length, cache_path=cache_path)
    if split == "validation":
        start = pd.Timestamp(default_pipeline_config().splits.validation_start)
        end = pd.Timestamp(default_pipeline_config().splits.validation_end)
    elif split == "evaluation":
        start = pd.Timestamp(default_pipeline_config().splits.test_start)
        end = pd.Timestamp("2026-04-30")
    else:
        raise ValueError(f"unsupported split {split!r}")
    test_positions = [
        position
        for position, date in enumerate(data.index[:-1])
        if start <= date <= end
        and np.isfinite(features[position]).all()
    ]
    if representative is not None:
        test_positions = test_positions[:representative]
    losses = data["portfolio_loss"].to_numpy(dtype=float)
    candidate_cache = {
        radius: {
            position: _candidate_positions(data, features, position, radius)
            for position in test_positions
        }
        for radius in radii
    }
    rows: list[dict[str, object]] = []
    for draw in range(draws):
        started = time.perf_counter()
        for radius in radii:
            pinballs: list[float] = []
            fz_scores: list[float] = []
            exceedances: list[bool] = []
            candidate_counts: list[int] = []
            neighbor_digest = hashlib.sha256()
            scenario_digest = hashlib.sha256()
            seed = int(seed_root + draw * 1000003 + radius * 1009)
            rng = np.random.default_rng(seed)
            for position in test_positions:
                candidates = candidate_cache[radius][position]
                candidate_counts.append(int(candidates.size))
                if candidates.size == 0:
                    continue
                if control == "topology_history_permutation":
                    if fast_permutation:
                        # A uniform permutation of feature origins induces a
                        # uniform ordering of candidate dates (up to exact
                        # distance ties).  This is the same null distribution
                        # without recomputing the catalog distance for every
                        # draw; the direct implementation remains tested above.
                        selected = np.sort(rng.choice(candidates, size=min(k, candidates.size), replace=False))
                        selection = type(
                            "FastSelection",
                            (),
                            {"selected_positions": selected},
                        )()
                    else:
                        selection = topology_history_permutation(
                            features[position], candidates, features, k, rng,
                            seed=seed, forecast_position=position,
                        )
                elif control == "uniform_random_retrieval":
                    selection = uniform_random_retrieval(
                        candidates, k, rng, seed=seed, forecast_position=position,
                    )
                else:
                    raise ValueError(f"unsupported control {control!r}")
                scenarios = losses[selection.selected_positions + 1]
                realized = losses[position + 1]
                if scenarios.size == 0 or not np.isfinite(realized) or not np.isfinite(scenarios).all():
                    continue
                var99 = float(np.quantile(scenarios, LEVEL))
                var975 = float(np.quantile(scenarios, ES_LEVEL))
                es975 = float(scenarios[scenarios >= var975].mean())
                pinballs.append(float(pinball_score(np.array([realized]), np.array([var99]), LEVEL)[0]))
                fz_scores.append(float(fz0_score(np.array([realized]), np.array([var975]), np.array([es975]), ES_LEVEL)[0]))
                exceedances.append(bool(realized > var99))
                neighbor_digest.update(selection.selected_positions.astype(np.int64).tobytes())
                scenario_digest.update(scenarios.astype(np.float64).tobytes())
            rows.append(
                {
                    "control": control,
                    "draw": draw,
                    "exclusion_radius": radius,
                    "n_forecasts": len(pinballs),
                    "control_mechanism": control,
                    "split": split,
                    "window_length": window_length,
                    "k": k,
                    "permutation_equivalent_fast_path": bool(fast_permutation and control == "topology_history_permutation"),
                    "history_only": True,
                    "candidate_count_min": int(min(candidate_counts)) if candidate_counts else 0,
                    "candidate_count_max": int(max(candidate_counts)) if candidate_counts else 0,
                    "mean_pinball_99": float(np.mean(pinballs)),
                    "exceedance_rate_99": float(np.mean(exceedances)),
                    "mean_fz0_975": float(np.mean(fz_scores)),
                    "neighbor_checksum": neighbor_digest.hexdigest(),
                    "scenario_checksum": scenario_digest.hexdigest(),
                    "runtime_seconds": time.perf_counter() - started,
                    "seed": seed,
                }
            )
        print(f"{control} draw={draw + 1}/{draws} seconds={time.perf_counter() - started:.2f}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--representative", type=int, default=None)
    parser.add_argument("--split", choices=("validation", "evaluation"), default="evaluation")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--window-length", type=int, default=250)
    parser.add_argument(
        "--control",
        "--label",
        dest="control",
        choices=("topology_history_permutation", "uniform_random_retrieval"),
        required=True,
    )
    parser.add_argument("--seed-root", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--radii", type=int, nargs="+", default=[0, 125, 250])
    parser.add_argument("--fast-permutation", action="store_true")
    args = parser.parse_args()
    default_seeds = {
        "topology_history_permutation": SEEDS["recovery"]["topology_history_permutation"],
        "uniform_random_retrieval": SEEDS["recovery"]["uniform_random_retrieval"],
    }
    seed = int(args.seed_root if args.seed_root is not None else default_seeds[args.control])
    frame = run(
        args.draws,
        args.representative,
        args.control,
        seed,
        split=args.split,
        k=args.k,
        window_length=args.window_length,
        radii=tuple(args.radii),
        fast_permutation=args.fast_permutation,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.control}_draw_summary_{args.draws}"
    output = args.output_dir / f"{stem}.parquet"
    frame.to_parquet(output, index=False)
    frame.to_csv(output.with_suffix(".csv"), index=False)
    print(f"wrote {output} rows={len(frame)}")


if __name__ == "__main__":
    main()
