"""Build ablation summary tables from pre-computed ablation run results.

Aggregates three ablation dimensions into a single tidy CSV:
  1. k sensitivity  (from results/topology_k_sweep/)
  2. Feature mode   (from results/topology_landscape_alpha_steps_ablation/)
  3. Alpha (α) sensitivity (from results/topology_landscape_alpha_steps_ablation/)

The output is used in Appendix A of the paper.

Usage
-----
    python scripts/build_ablation_summary_table.py
    python scripts/build_ablation_summary_table.py \
        --output results/final_results_package/table_ablation_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── directory helpers ────────────────────────────────────────────────────────

K_SWEEP_DIR = REPO_ROOT / "results" / "topology_k_sweep"
ALPHA_STEPS_DIR = (
    REPO_ROOT / "results" / "topology_landscape_alpha_steps_ablation"
)
FEATURE_ABLATION_DIR = (
    REPO_ROOT / "results" / "topology_landscape_feature_ablation"
)
MULTISERIES_DIR = REPO_ROOT / "results" / "topology_multiseries_ablation"
OUTPUT_DEFAULT = (
    REPO_ROOT / "results" / "final_results_package" / "table_ablation_summary.csv"
)

DISPLAY_NAMES = {
    "landscape1":                 "landscape1 (single H₁ bar)",
    "landscape_h1_top3_weighted": "landscape_h1_top3_weighted (finalist)",
    "landscape_h1_top3_concat":   "landscape_h1_top3_concat",
    "landscape_h0h1_top1_concat": "landscape_h0h1_top1_concat",
    "landscape_h0h1_top3_concat": "landscape_h0h1_top3_concat",
}

INPUT_MODE_NAMES = {
    "portfolio_only": "portfolio_only (finalist)",
    "portfolio_plus_components": "portfolio_plus_components",
    "components_only": "components_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ablation summary table.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DEFAULT,
        help="Path for the output CSV.",
    )
    return parser.parse_args()


# ── sub-table builders ───────────────────────────────────────────────────────

def _build_k_sweep() -> pd.DataFrame:
    """k sensitivity: topology_knn test-split metrics for each k in the sweep."""
    records = []
    for k_dir in sorted(K_SWEEP_DIR.iterdir()):
        if not k_dir.is_dir():
            continue
        summary_path = k_dir / "summary.csv"
        if not summary_path.exists():
            print(f"  WARNING: {summary_path} missing — skipping")
            continue
        df = pd.read_csv(summary_path)
        row = df[(df["split"] == "test") & (df["method"] == "topology_knn")]
        if row.empty:
            print(f"  WARNING: no topology_knn test row in {summary_path}")
            continue
        r = row.iloc[0]
        # infer k from directory name or from mean_scenarios
        k_label = k_dir.name  # e.g. "k1000"
        k_value = int(k_label.replace("k", "")) if k_label.startswith("k") else None
        records.append(
            {
                "ablation": "k_sensitivity",
                "dimension": "k (number of analogues)",
                "value_label": str(k_value) if k_value else k_label,
                "value_numeric": k_value,
                "var_exceedance_99": r["exceedance_rate_99"],
                "es_exceedance_975": r["exceedance_rate_es_975"],
                "note": "landscape1, α=0.15 (fixed)",
            }
        )
    df_out = (
        pd.DataFrame(records)
        .sort_values("value_numeric")
        .reset_index(drop=True)
    )
    return df_out


def _needs_forecast_fallback(frame: pd.DataFrame, required_columns: list[str]) -> bool:
    return any(column not in frame.columns for column in required_columns)


def _group_from_forecasts(
    forecasts_path: Path,
    group_columns: list[str],
) -> pd.DataFrame:
    forecasts = pd.read_csv(forecasts_path)
    missing = [column for column in group_columns if column not in forecasts.columns]
    if missing:
        raise KeyError(
            f"Missing grouping columns {missing} in forecasts file: {forecasts_path}"
        )
    grouped = (
        forecasts.groupby(group_columns, dropna=False)
        .agg(
            n_forecasts=("forecast_date", "size"),
            exceedance_rate_99=("var_exceedance_99", "mean"),
            exceedance_rate_es_975=("es_exceedance_975", "mean"),
        )
        .reset_index()
    )
    return grouped


def _build_feature_mode(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Feature mode comparison at fixed α=0.15, steps=200, test split."""
    sub = feature_df[
        (feature_df["split"] == "test")
        & (feature_df["method"] == "topology_knn")
    ].copy()
    if sub.empty:
        print("  WARNING: no rows for feature-mode comparison — check ablation CSV")
        return pd.DataFrame()
    records = []
    for _, r in sub.iterrows():
        fmode = r.get("feature_mode", r.get("ablation_feature_mode"))
        records.append(
            {
                "ablation": "feature_mode",
                "dimension": "persistence feature mode",
                "value_label": DISPLAY_NAMES.get(fmode, fmode),
                "value_numeric": None,
                "var_exceedance_99": r["exceedance_rate_99"],
                "es_exceedance_975": r["exceedance_rate_es_975"],
                "note": "α=0.15, L=125, steps=200, k=1000 (fixed)",
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values("var_exceedance_99")
        .reset_index(drop=True)
    )


def _build_alpha_sensitivity(alpha_steps_df: pd.DataFrame) -> pd.DataFrame:
    """α sensitivity for the finalist feature mode at steps=200, test split."""
    sub = alpha_steps_df[
        (alpha_steps_df["split"] == "test")
        & (alpha_steps_df["method"] == "topology_knn")
        & (alpha_steps_df["ablation_feature_mode"] == "landscape_h1_top3_weighted")
        & (alpha_steps_df["ablation_landscape_steps"] == 200)
    ].copy()
    if sub.empty:
        print("  WARNING: no rows for alpha sensitivity — check ablation CSV")
        return pd.DataFrame()
    records = []
    for _, r in sub.iterrows():
        alpha_val = r["ablation_alpha"]
        records.append(
            {
                "ablation": "alpha_sensitivity",
                "dimension": "α (state vs topology weight)",
                "value_label": f"α={alpha_val:.2f}",
                "value_numeric": float(alpha_val),
                "var_exceedance_99": r["exceedance_rate_99"],
                "es_exceedance_975": r["exceedance_rate_es_975"],
                "note": "landscape_h1_top3_weighted, L=125, steps=200, k=1000 (fixed)",
            }
        )
    # Append the locked α=0.0 reference row from the canonical topology pipeline.
    finalist_path = REPO_ROOT / "results" / "topology_pipeline" / "summary.csv"
    if finalist_path.exists():
        df_fin = pd.read_csv(finalist_path)
        fin_row = df_fin[(df_fin["split"] == "test") & (df_fin["method"] == "topology_knn")]
        if not fin_row.empty:
            r = fin_row.iloc[0]
            records.append(
                {
                    "ablation": "alpha_sensitivity",
                    "dimension": "α (state vs topology weight)",
                    "value_label": "α=0.00 (canonical)",
                    "value_numeric": 0.0,
                    "var_exceedance_99": r["exceedance_rate_99"],
                    "es_exceedance_975": r["exceedance_rate_es_975"],
                    "note": "landscape_h1_top3_weighted, L=125, steps=200, k=1000 (canonical run)",
                }
            )
    return (
        pd.DataFrame(records)
        .sort_values("value_numeric")
        .reset_index(drop=True)
    )


def _build_multiseries(multiseries_df: pd.DataFrame) -> pd.DataFrame:
    """Input-mode comparison at fixed α=0.15, steps=200, k=1000."""
    sub = multiseries_df[
        (multiseries_df["split"] == "test")
        & (multiseries_df["method"] == "topology_knn")
    ].copy()
    if sub.empty:
        print("  WARNING: no rows for multiseries comparison — check multiseries CSV")
        return pd.DataFrame()

    records = []
    for _, r in sub.iterrows():
        input_mode = r["topology_input_mode"]
        records.append(
            {
                "ablation": "input_mode",
                "dimension": "topology input mode",
                "value_label": INPUT_MODE_NAMES.get(input_mode, input_mode),
                "value_numeric": None,
                "var_exceedance_99": r["exceedance_rate_99"],
                "es_exceedance_975": r["exceedance_rate_es_975"],
                "note": "landscape1, α=0.15, L=125, steps=200, k=1000 (fixed)",
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values("var_exceedance_99")
        .reset_index(drop=True)
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── k sweep ──────────────────────────────────────────────────────────────
    if not K_SWEEP_DIR.exists():
        print(f"WARNING: k-sweep directory not found: {K_SWEEP_DIR}")
        k_table = pd.DataFrame()
    else:
        k_table = _build_k_sweep()
        print(f"k-sweep: {len(k_table)} rows")
        print(k_table[["value_label", "var_exceedance_99", "es_exceedance_975"]].to_string(index=False))

    # ── alpha/steps ablation ─────────────────────────────────────────────────
    alpha_steps_path = ALPHA_STEPS_DIR / "summary_topology_focus.csv"
    if not alpha_steps_path.exists():
        print(f"WARNING: alpha/steps ablation not found: {alpha_steps_path}")
        alpha_table = pd.DataFrame()
    else:
        alpha_steps_df = pd.read_csv(alpha_steps_path)
        alpha_table = _build_alpha_sensitivity(alpha_steps_df)
        print(f"\nAlpha sensitivity: {len(alpha_table)} rows")
        if not alpha_table.empty:
            print(alpha_table[["value_label", "var_exceedance_99"]].to_string(index=False))

    feature_path = FEATURE_ABLATION_DIR / "summary.csv"
    feature_forecasts_path = FEATURE_ABLATION_DIR / "forecasts.csv"
    if not feature_path.exists() and not feature_forecasts_path.exists():
        print(f"WARNING: feature ablation not found: {feature_path}")
        feature_table = pd.DataFrame()
    else:
        if feature_path.exists():
            feature_df = pd.read_csv(feature_path)
        else:
            feature_df = pd.DataFrame()
        if feature_df.empty or _needs_forecast_fallback(feature_df, ["feature_mode"]):
            feature_df = _group_from_forecasts(
                feature_forecasts_path,
                ["split", "method", "feature_mode"],
            )
        feature_table = _build_feature_mode(feature_df)
        print(f"\nFeature mode: {len(feature_table)} rows")
        if not feature_table.empty:
            print(feature_table[["value_label", "var_exceedance_99"]].to_string(index=False))

    multiseries_path = MULTISERIES_DIR / "summary.csv"
    multiseries_forecasts_path = MULTISERIES_DIR / "forecasts.csv"
    if not multiseries_path.exists() and not multiseries_forecasts_path.exists():
        print(f"WARNING: multiseries ablation not found: {multiseries_path}")
        multiseries_table = pd.DataFrame()
    else:
        if multiseries_path.exists():
            multiseries_df = pd.read_csv(multiseries_path)
        else:
            multiseries_df = pd.DataFrame()
        if multiseries_df.empty or _needs_forecast_fallback(multiseries_df, ["topology_input_mode"]):
            multiseries_df = _group_from_forecasts(
                multiseries_forecasts_path,
                ["split", "method", "topology_input_mode"],
            )
        multiseries_table = _build_multiseries(multiseries_df)
        print(f"\nInput mode: {len(multiseries_table)} rows")
        if not multiseries_table.empty:
            print(multiseries_table[["value_label", "var_exceedance_99"]].to_string(index=False))

    # ── combine and write ─────────────────────────────────────────────────────
    parts = [t for t in [k_table, feature_table, alpha_table, multiseries_table] if not t.empty]
    if not parts:
        raise RuntimeError("No ablation data found — run ablation scripts first.")

    combined = pd.concat(parts, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"\nAblation summary written to: {args.output}")
    print(f"Total rows: {len(combined)}")


if __name__ == "__main__":
    main()
