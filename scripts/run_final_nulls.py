"""Run the frozen null analysis for locked radii and validation k-sensitivity."""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_repeated_nulls import run
from tda_risk.scoring import fz0_score


CONTROLS = ("topology_history_permutation", "uniform_random_retrieval")
SEEDS = json.loads((ROOT / "revision_config" / "random_seed_registry.json").read_text())["recovery"]
OUT = ROOT / "results" / "final_nulls"
K_VALUES = (100, 250, 500, 750, 1000)
LOCKED_EVALUATION_NULL_RADII = (0, 125, 250)
VALIDATION_K_SENSITIVITY_RADII = (0, 125)
DRAW_COUNT = 500
NULL_CONFIG_VERSION = "final-history-only-null-v2"
LOCKED_CONFIGURATION = "topology_l250_k100"


def _summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(group_columns, as_index=False)
        .agg(
            draws=("draw", "nunique"),
            n_forecasts=("n_forecasts", "mean"),
            candidate_count_min=("candidate_count_min", "min"),
            candidate_count_max=("candidate_count_max", "max"),
            mean_pinball_99=("mean_pinball_99", "mean"),
            sd_pinball_99=("mean_pinball_99", "std"),
            p05_pinball_99=("mean_pinball_99", lambda x: x.quantile(0.05)),
            p50_pinball_99=("mean_pinball_99", "median"),
            p95_pinball_99=("mean_pinball_99", lambda x: x.quantile(0.95)),
            mean_exceedance_rate_99=("exceedance_rate_99", "mean"),
            mean_fz0_975=("mean_fz0_975", "mean"),
            sd_fz0_975=("mean_fz0_975", "std"),
            p05_fz0_975=("mean_fz0_975", lambda x: x.quantile(0.05)),
            p50_fz0_975=("mean_fz0_975", "median"),
            p95_fz0_975=("mean_fz0_975", lambda x: x.quantile(0.95)),
        )
        .fillna(0.0)
    )


def _real_topology_scores() -> pd.DataFrame:
    rows = []
    for radius in LOCKED_EVALUATION_NULL_RADII:
        frame = pd.read_parquet(ROOT / "results" / "final_evaluation" / f"forecasts_radius{radius}.parquet")
        frame = frame[frame["configuration_id"].eq("topology_l250_k100")].copy()
        realized = frame["realized_loss"].to_numpy(float)
        var975 = frame["VaR_975"].to_numpy(float)
        es975 = frame["ES_975"].to_numpy(float)
        fz0 = fz0_score(realized, var975, es975, 0.975)
        rows.append(
            {
                "exclusion_radius": radius,
                "real_mean_pinball_99": float(frame["pinball_99"].mean()),
                "real_mean_fz0_975": float(np.mean(fz0)),
            }
        )
    return pd.DataFrame(rows)


def _randomization_tests(frame: pd.DataFrame) -> pd.DataFrame:
    real = _real_topology_scores()
    rows = []
    for _, real_row in real.iterrows():
        for control in CONTROLS:
            sample = frame[
                frame["control"].eq(control) & frame["exclusion_radius"].eq(real_row["exclusion_radius"])
            ]
            draws = len(sample)
            rows.append(
                {
                    "control": control,
                    "exclusion_radius": int(real_row["exclusion_radius"]),
                    "draws": draws,
                    "real_mean_pinball_99": real_row["real_mean_pinball_99"],
                    "null_mean_pinball_99": sample["mean_pinball_99"].mean(),
                    "pinball_randomization_p_lower": (1 + int((sample["mean_pinball_99"] <= real_row["real_mean_pinball_99"]).sum())) / (draws + 1),
                    "real_mean_fz0_975": real_row["real_mean_fz0_975"],
                    "null_mean_fz0_975": sample["mean_fz0_975"].mean(),
                    "fz0_randomization_p_lower": (1 + int((sample["mean_fz0_975"] <= real_row["real_mean_fz0_975"]).sum())) / (draws + 1),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "topology_feature_matrix_l250.npy"

    main_frames = []
    for control in CONTROLS:
        seed = int(SEEDS[control])
        frame = run(
            DRAW_COUNT,
            None,
            control,
            seed,
            split="evaluation",
            k=100,
            window_length=250,
            cache_path=cache_path,
            fast_permutation=(control == "topology_history_permutation"),
            radii=LOCKED_EVALUATION_NULL_RADII,
        )
        main_frames.append(frame)
    main = pd.concat(main_frames, ignore_index=True)
    main.to_parquet(OUT / "main_null_draws.parquet", index=False)
    _summary(main, ["control", "exclusion_radius"]).to_csv(OUT / "main_null_summary.csv", index=False)
    _randomization_tests(main).to_csv(OUT / "randomization_tests.csv", index=False)
    metadata = {
        "code_configuration_version": NULL_CONFIG_VERSION,
        "split": "evaluation",
        "draws_per_control_radius": DRAW_COUNT,
        "controls": list(CONTROLS),
        "radii": list(LOCKED_EVALUATION_NULL_RADII),
        "seed_roots": {control: int(SEEDS[control]) for control in CONTROLS},
        "locked_configuration": LOCKED_CONFIGURATION,
        "window_length": 250,
        "k": 100,
        "outputs": {
            "main_null_draws.parquet": {
                "radii": list(LOCKED_EVALUATION_NULL_RADII),
                "draws_per_control_radius": DRAW_COUNT,
            }
        },
    }
    (OUT / "null_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    sensitivity_frames = []
    for control in CONTROLS:
        base_seed = int(SEEDS[control])
        for k in K_VALUES:
            frame = run(
                250,
                None,
                control,
                base_seed + k * 10007,
                split="validation",
                k=k,
                window_length=250,
                cache_path=cache_path,
                fast_permutation=(control == "topology_history_permutation"),
                radii=VALIDATION_K_SENSITIVITY_RADII,
            )
            sensitivity_frames.append(frame)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    sensitivity.to_parquet(OUT / "k_sensitivity_draws.parquet", index=False)
    _summary(sensitivity, ["control", "exclusion_radius", "k"]).to_csv(
        OUT / "k_sensitivity_summary.csv", index=False
    )
    shutil.rmtree(cache_dir)
    print(f"wrote final null outputs: main_rows={len(main)} sensitivity_rows={len(sensitivity)}")


if __name__ == "__main__":
    main()
