"""Block-bootstrap significance test for topology vs. null-placebo VaR exceedance gap.

Tests H₀: exceedance_rate(topology_knn) == exceedance_rate(topo_null) on the test
split, using a circular block bootstrap (block size = 21 trading days ≈ 1 month).

One-sided p-value: P(bootstrap_diff ≥ observed_diff | H₀)
where diff = exc_rate(topo_null) − exc_rate(topology_knn)  (positive = topology wins).

Usage
-----
    python scripts/run_placebo_significance_test.py
    python scripts/run_placebo_significance_test.py \
        --forecasts results/topology_pipeline/forecasts.csv \
        --n-bootstrap 10000 \
        --block-size 21
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FORECASTS = REPO_ROOT / "results" / "topology_pipeline" / "forecasts.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT / "results" / "final_results_package" / "table_placebo_significance.csv"
)

METHOD_A = "topology_knn"   # treatment
METHOD_B = "topology_placebo_knn"      # null placebo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block-bootstrap significance test: topology vs. null placebo."
    )
    parser.add_argument(
        "--forecasts",
        type=Path,
        default=DEFAULT_FORECASTS,
        help="Path to forecasts CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
        help="Number of bootstrap replications (default: 5000).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=21,
        help="Block size in trading days (default: 21 ≈ 1 month).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260427,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def _circular_block_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    block_size: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return bootstrap distribution of (mean(b) - mean(a)).

    Both series must be aligned on the same dates (same length).
    Circular block bootstrap: blocks wrap around the end of the series.
    """
    n = len(a)
    n_blocks = int(np.ceil(n / block_size))
    diffs = b - a  # daily difference series
    boot_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([
            np.arange(s, s + block_size) % n for s in starts
        ])[:n]
        boot_means[i] = diffs[idx].mean()

    return boot_means


def main() -> None:
    args = parse_args()

    if not args.forecasts.exists():
        raise FileNotFoundError(f"Forecasts file not found: {args.forecasts}")

    print(f"Reading: {args.forecasts}")
    fc = pd.read_csv(args.forecasts)

    # Filter to test split and relevant methods
    test = fc[fc["split"] == "test"].copy()
    available_methods = test["method"].unique()
    for m in (METHOD_A, METHOD_B):
        if m not in available_methods:
            raise ValueError(
                f"Method '{m}' not found in forecasts. "
                f"Available: {sorted(available_methods)}"
            )

    test_a = test[test["method"] == METHOD_A].sort_values("forecast_date").reset_index(drop=True)
    test_b = test[test["method"] == METHOD_B].sort_values("forecast_date").reset_index(drop=True)

    # Align on forecast_date (inner join — same dates only)
    merged = pd.merge(
        test_a[["forecast_date", "var_exceedance_99"]].rename(
            columns={"var_exceedance_99": "exc_a"}
        ),
        test_b[["forecast_date", "var_exceedance_99"]].rename(
            columns={"var_exceedance_99": "exc_b"}
        ),
        on="forecast_date",
    )

    if len(merged) == 0:
        raise RuntimeError("No overlapping forecast dates between the two methods.")

    print(f"  Aligned observations: {len(merged)}")

    exc_a = merged["exc_a"].to_numpy(dtype=float)
    exc_b = merged["exc_b"].to_numpy(dtype=float)

    rate_a = exc_a.mean()
    rate_b = exc_b.mean()
    observed_diff = rate_b - rate_a  # positive = placebo worse = topology wins

    print(f"\n  {METHOD_A} VaR(99%) exceedance rate : {rate_a:.6f}")
    print(f"  {METHOD_B}   VaR(99%) exceedance rate : {rate_b:.6f}")
    print(f"  Observed difference (null − topology) : {observed_diff:+.6f}")

    # Bootstrap under H₀: shuffle diff series
    rng = np.random.default_rng(args.seed)
    boot_diffs = _circular_block_bootstrap(
        exc_a, exc_b, args.block_size, args.n_bootstrap, rng
    )

    # One-sided p-value: P(boot_diff >= observed_diff | H₀)
    # H₀ is imposed by centering: boot diffs already centered at observed diff
    # Use standard approach: center bootstrap distribution at 0
    centered = boot_diffs - boot_diffs.mean()
    p_value_one_sided = float((centered >= observed_diff).mean())
    p_value_two_sided = float((np.abs(centered) >= np.abs(observed_diff)).mean())

    # 95% CI for the true diff
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    print(f"\n  Bootstrap reps        : {args.n_bootstrap}")
    print(f"  Block size            : {args.block_size} days")
    print(f"  p-value (one-sided)   : {p_value_one_sided:.4f}")
    print(f"  p-value (two-sided)   : {p_value_two_sided:.4f}")
    print(f"  95% CI for diff       : [{ci_lower:.6f}, {ci_upper:.6f}]")

    result = pd.DataFrame([
        {
            "method_a":          METHOD_A,
            "method_b":          METHOD_B,
            "n_obs":             len(merged),
            "exc_rate_a":        rate_a,
            "exc_rate_b":        rate_b,
            "diff_b_minus_a":    observed_diff,
            "block_size":        args.block_size,
            "n_bootstrap":       args.n_bootstrap,
            "p_value_one_sided": p_value_one_sided,
            "p_value_two_sided": p_value_two_sided,
            "ci_lower_95":       ci_lower,
            "ci_upper_95":       ci_upper,
            "seed":              args.seed,
        }
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\nResult written to: {args.output}")


if __name__ == "__main__":
    main()
