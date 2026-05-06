"""Extended stress-period analysis across five market stress windows.

Reads the topology pipeline forecasts CSV and computes per-method exceedance
metrics within each stress window.  Results are written to a single tidy CSV
that can be turned directly into a paper table.

Usage
-----
    python scripts/run_extended_stress_analysis.py
    python scripts/run_extended_stress_analysis.py \
        --forecasts results/topology_pipeline/forecasts.csv \
        --output-dir results/extended_stress
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# Stress windows: label -> (start, end) inclusive on realized_date
STRESS_WINDOWS: dict[str, tuple[str, str]] = {
    "covid_crash":          ("2020-02-15", "2020-04-30"),
    "inflation_rates_shock": ("2022-01-01", "2022-12-31"),
    "2018_q4_selloff":      ("2018-10-01", "2018-12-31"),
    "2023_banking_stress":  ("2023-03-08", "2023-05-05"),
    "2020_rates_crash":     ("2020-03-09", "2020-03-31"),
}

# Descriptive labels for the paper table
WINDOW_LABELS: dict[str, str] = {
    "covid_crash":           "COVID crash (Feb–Apr 2020)",
    "inflation_rates_shock": "Inflation/rates shock (2022)",
    "2018_q4_selloff":       "2018 Q4 equity selloff",
    "2023_banking_stress":   "2023 banking stress (SVB)",
    "2020_rates_crash":      "2020 rates flash crash (Mar)",
}

# Methods to include in the output (in display order)
FOCUS_METHODS = [
    "topology_knn",
    "topo_null",
    "fhs_ewma",
    "rolling_hs",
    "euclidean_knn",
    "window_euclidean_knn",
    "mahalanobis_knn",
    "dtw_window",
    "fpca_window",
    "garch_t",
    "regime_hs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended stress-period analysis.")
    parser.add_argument(
        "--forecasts",
        type=Path,
        default=REPO_ROOT / "results" / "topology_pipeline" / "forecasts.csv",
        help="Path to forecasts CSV (must contain split, method, realized_date, "
             "VaR_99, ES_975, realized_loss, var_exceedance_99, es_exceedance_975).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "extended_stress",
        help="Directory where output CSV is written.",
    )
    return parser.parse_args()


def _compute_stress_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    test = forecasts[forecasts["split"] == "test"].copy()
    test["realized_date"] = pd.to_datetime(test["realized_date"], utc=False)

    records: list[pd.DataFrame] = []
    for key, (start, end) in STRESS_WINDOWS.items():
        mask = (test["realized_date"] >= pd.Timestamp(start)) & (
            test["realized_date"] <= pd.Timestamp(end)
        )
        period = test.loc[mask]
        if period.empty:
            print(f"  WARNING: no test forecasts in window '{key}' ({start} – {end})")
            continue

        agg = (
            period.groupby("method", dropna=False)
            .agg(
                n_forecasts=("method", "size"),
                mean_var_99=("VaR_99", "mean"),
                mean_es_975=("ES_975", "mean"),
                exceedance_rate_99=("var_exceedance_99", "mean"),
                exceedance_rate_es_975=("es_exceedance_975", "mean"),
                mean_realized_loss=("realized_loss", "mean"),
            )
            .reset_index()
        )
        agg.insert(0, "stress_period_key", key)
        agg.insert(1, "stress_period_label", WINDOW_LABELS[key])
        records.append(agg)

    if not records:
        raise RuntimeError("No stress windows produced any output — check date ranges.")

    combined = pd.concat(records, ignore_index=True)

    # Order by window definition order, then by exceedance rate within each window
    window_order = {k: i for i, k in enumerate(STRESS_WINDOWS)}
    combined["_window_order"] = combined["stress_period_key"].map(window_order)
    combined = combined.sort_values(
        ["_window_order", "exceedance_rate_99"], ascending=[True, True]
    ).drop(columns=["_window_order"])

    return combined


def _focus_subset(table: pd.DataFrame) -> pd.DataFrame:
    present = [m for m in FOCUS_METHODS if m in table["method"].values]
    return table[table["method"].isin(present)].copy()


def main() -> None:
    args = parse_args()

    if not args.forecasts.exists():
        raise FileNotFoundError(
            f"Forecasts file not found: {args.forecasts}\n"
            "Run scripts/run_topology_pipeline.py first."
        )

    print(f"Reading forecasts from: {args.forecasts}")
    forecasts = pd.read_csv(args.forecasts)

    required_cols = {
        "split", "method", "realized_date", "VaR_99", "ES_975",
        "realized_loss", "var_exceedance_99", "es_exceedance_975",
    }
    missing = required_cols - set(forecasts.columns)
    if missing:
        raise ValueError(f"Forecasts CSV is missing columns: {missing}")

    print(f"  {len(forecasts):,} rows, methods: {sorted(forecasts['method'].unique())}")

    stress_table = _compute_stress_table(forecasts)
    focus_table = _focus_subset(stress_table)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_all = args.output_dir / "table_stress_extended_all_methods.csv"
    out_focus = args.output_dir / "table_stress_extended.csv"

    stress_table.to_csv(out_all, index=False)
    focus_table.to_csv(out_focus, index=False)

    print(f"\nExtended stress table written:")
    print(f"  All methods : {out_all}")
    print(f"  Focus subset: {out_focus}")
    print(f"\nWindow summary (focus methods):")
    for key in STRESS_WINDOWS:
        sub = focus_table[focus_table["stress_period_key"] == key]
        if sub.empty:
            print(f"  {key}: (no data)")
            continue
        best = sub.loc[sub["exceedance_rate_99"].idxmin()]
        worst = sub.loc[sub["exceedance_rate_99"].idxmax()]
        topo = sub[sub["method"] == "topology_knn"]
        topo_str = (
            f"topology_knn={topo['exceedance_rate_99'].iloc[0]:.4f}"
            if not topo.empty
            else "topology_knn=N/A"
        )
        print(
            f"  {WINDOW_LABELS[key]}: "
            f"best={best['method']}({best['exceedance_rate_99']:.4f}), "
            f"worst={worst['method']}({worst['exceedance_rate_99']:.4f}), "
            f"{topo_str}"
        )


if __name__ == "__main__":
    main()
