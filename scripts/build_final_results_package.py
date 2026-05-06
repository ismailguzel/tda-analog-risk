from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "results" / "baseline_pipeline"
TOPOLOGY_DIR = REPO_ROOT / "results" / "topology_pipeline"
OUTPUT_DIR = REPO_ROOT / "results" / "final_results_package"
TOPOLOGY_FORMAL_TESTS_DIR = TOPOLOGY_DIR / "formal_tests"
VALIDATION_FINALISTS_DIR = REPO_ROOT / "results" / "validation_finalists"
HYBRID_COMPARISON_PATH = OUTPUT_DIR / "table_hybrid_comparison_matched.csv"

MAIN_MODEL_ORDER = [
    "topology_knn",
    "fhs_ewma",
    "rolling_hs",
    "garch_t",
    "euclidean_knn",
    "dtw_window",
]
MAIN_MODEL_LABELS = {
    "topology_knn": "w-Topology",
    "fhs_ewma": "FHS-EWMA",
    "rolling_hs": "Rolling HS",
    "garch_t": "GARCH-t",
    "euclidean_knn": "s-Euclidean",
    "dtw_window": "w-DTW",
}


def _locked_main_spec_filters() -> dict[str, dict[str, object]]:
    return {
        "topology_knn": {"k": 1000},
        "fhs_ewma": {"lambda": 0.94},
        "rolling_hs": {"window": 250},
    }


def _apply_locked_filter(frame: pd.DataFrame, method: str, filters: dict[str, object]) -> pd.DataFrame:
    subset = frame[frame["method"] == method].copy()
    for column, expected in filters.items():
        if column in subset.columns:
            subset = subset[subset[column] == expected]
    return subset


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def _stress_windows() -> dict[str, tuple[str, str]]:
    return {
        "covid_crash": ("2020-02-15", "2020-04-30"),
        "inflation_rates_shock": ("2022-01-01", "2022-12-31"),
    }


def _copy_tree(source_dir: Path, destination_dir: Path) -> None:
    if not source_dir.exists():
        return
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)


def _method_sort_key(series: pd.Series) -> pd.Series:
    order_map = {method: idx for idx, method in enumerate(MAIN_MODEL_ORDER)}
    return series.map(order_map).fillna(999).astype(int)


def build_method_tables(
    baseline_summary: pd.DataFrame,
    topology_summary: pd.DataFrame,
) -> None:
    baseline_test = baseline_summary[baseline_summary["split"] == "test"].copy()
    topology_test = topology_summary[topology_summary["split"] == "test"].copy()

    # Table 1: full baseline-grid aggregates.
    baseline_test.sort_values(["exceedance_rate_99", "exceedance_rate_es_975"]).to_csv(
        OUTPUT_DIR / "table_test_baseline_grid.csv",
        index=False,
    )

    # Table 2: finalist head-to-head comparison (single retained topology run).
    finalist_head_to_head = topology_test.sort_values(
        ["exceedance_rate_99", "exceedance_rate_es_975"]
    )
    finalist_head_to_head.to_csv(
        OUTPUT_DIR / "table_test_finalist_head_to_head.csv",
        index=False,
    )

    # Table 3: focused placebo + nearest-neighbor comparators.
    focus_methods = {
        "topology_knn",
        "topology_placebo_knn",
        "euclidean_knn",
        "window_euclidean_knn",
        "random_knn",
        "mahalanobis_knn",
        "dtw_window",
        "fpca_window",
        "rolling_hs",
        "fhs_ewma",
        "regime_hs",
    }
    finalist_head_to_head[finalist_head_to_head["method"].isin(focus_methods)].to_csv(
        OUTPUT_DIR / "table_test_focus_methods.csv",
        index=False,
    )


def build_stress_tables(topology_forecasts: pd.DataFrame) -> None:
    forecasts = topology_forecasts.copy()
    forecasts["realized_date"] = pd.to_datetime(forecasts["realized_date"], utc=False)
    forecasts["forecast_date"] = pd.to_datetime(forecasts["forecast_date"], utc=False)
    test_forecasts = forecasts[forecasts["split"] == "test"].copy()

    records: list[pd.DataFrame] = []
    for label, (start, end) in _stress_windows().items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        mask = (test_forecasts["realized_date"] >= start_ts) & (
            test_forecasts["realized_date"] <= end_ts
        )
        period = test_forecasts.loc[mask].copy()
        if period.empty:
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
        agg.insert(0, "stress_period", label)
        records.append(agg)

    if records:
        stress_table = pd.concat(records, ignore_index=True).sort_values(
            ["stress_period", "exceedance_rate_99", "exceedance_rate_es_975"]
        )
        stress_table.to_csv(OUTPUT_DIR / "table_stress_period_summary.csv", index=False)

    # Monthly exceedance panel for plotting rolling/temporal diagnostics.
    monthly = test_forecasts.copy()
    monthly["month"] = monthly["realized_date"].dt.to_period("M").astype(str)
    monthly_agg = (
        monthly.groupby(["month", "method"], dropna=False)
        .agg(
            n_forecasts=("method", "size"),
            exceedance_rate_99=("var_exceedance_99", "mean"),
            exceedance_rate_es_975=("es_exceedance_975", "mean"),
            mean_realized_loss=("realized_loss", "mean"),
            mean_var_99=("VaR_99", "mean"),
            mean_es_975=("ES_975", "mean"),
        )
        .reset_index()
        .sort_values(["month", "method"])
    )
    monthly_agg.to_csv(OUTPUT_DIR / "figure_monthly_exceedance_panel.csv", index=False)

    # Qualitative case-study panel around COVID crash onset.
    case_start = pd.Timestamp("2020-02-15")
    case_end = pd.Timestamp("2020-04-30")
    case_methods = [
        "topology_knn",
        "topology_placebo_knn",
        "euclidean_knn",
        "random_knn",
        "rolling_hs",
        "fhs_ewma",
    ]
    case_panel = test_forecasts[
        (test_forecasts["realized_date"] >= case_start)
        & (test_forecasts["realized_date"] <= case_end)
        & (test_forecasts["method"].isin(case_methods))
    ].copy()
    case_panel.sort_values(["realized_date", "method"]).to_csv(
        OUTPUT_DIR / "case_study_covid_window_daily.csv",
        index=False,
    )


def build_notes() -> None:
    notes = """# Final Results Package Notes

This folder contains paper-facing artifacts generated from:

- `results/baseline_pipeline`
- `results/topology_pipeline`

Key outputs:

- `table_main_selected_models.csv`: main-text test table using validation-selected benchmark specifications.
- `table_main_hybrid_comparison.csv`: main-text matched-date comparison for topology, FHS, and scenario mixture.
- `figure_main_monthly_selected_models.csv`: reduced monthly panel for the main text figure.
- `figure_main_covid_selected_models.csv`: reduced COVID-window panel for the main text figure.
- `table_test_baseline_grid.csv`: aggregate test metrics from the full baseline grid.
- `table_test_finalist_head_to_head.csv`: pooled test metrics from the retained topology run (appendix/supporting only).
- `table_test_focus_methods.csv`: focused table for topology/placebo/core baselines.
- `table_stress_period_summary.csv`: method metrics inside predefined stress windows.
- `figure_monthly_exceedance_panel.csv`: monthly exceedance panel for plotting.
- `case_study_covid_window_daily.csv`: day-level panel for qualitative stress-case reading.
- `formal_tests/`: canonical topology formal-test exports copied from `results/topology_pipeline/formal_tests`.
- `package_metadata.json`: run window and source-artifact summary for the package build.

Notes:

- `docs/arxiv/mainfigures.py` generates a synthetic mechanism figure for exposition; it is intentionally kept separate from the empirical package.
- `window_euclidean_knn` is a supporting raw-window $L^2$ comparator added to separate window-level retrieval effects from topology-specific representation effects; it is not included in the current main-text selected-model table by default.
"""
    (OUTPUT_DIR / "README.md").write_text(notes, encoding="utf-8")


def build_main_text_tables() -> None:
    finalists_path = VALIDATION_FINALISTS_DIR / "test_finalists.csv"
    if finalists_path.exists():
        finalists = _read_csv(finalists_path)
        selected = finalists[finalists["method"].isin(MAIN_MODEL_ORDER)].copy()
        if not selected.empty:
            selected.insert(0, "model_label", selected["method"].map(MAIN_MODEL_LABELS))
            selected = selected.sort_values(
                ["method"],
                key=_method_sort_key,
            )
            selected = selected[
                [
                    "model_label",
                    "method",
                    "test_n_forecasts",
                    "test_mean_var_99",
                    "test_mean_es_975",
                    "test_exceedance_rate_99",
                    "test_exceedance_rate_es_975",
                    "test_fz_style_mean_score",
                    "k",
                    "window",
                    "window_length",
                    "lambda",
                    "dist",
                    "sim_draws",
                    "refit_interval",
                    "topology_input_mode",
                    "feature_mode",
                    "alpha",
                ]
            ]
            selected.to_csv(OUTPUT_DIR / "table_main_selected_models.csv", index=False)

    if HYBRID_COMPARISON_PATH.exists():
        hybrid = _read_csv(HYBRID_COMPARISON_PATH)
        subset = hybrid[hybrid["comparison_group"] == "scenario_mixture_coverage_then_fz"].copy()
        preferred = subset[subset["model_label"].isin([
            "topology_only_matched",
            "fhs_matched_selected_lambda",
            "scenario_mixture",
        ])].copy()
        if not preferred.empty:
            label_map = {
                "topology_only_matched": "Topology only (matched dates)",
                "fhs_matched_selected_lambda": "FHS-EWMA (matched dates)",
                "scenario_mixture": "Topology-FHS scenario mixture",
            }
            preferred.insert(0, "display_label", preferred["model_label"].map(label_map))
            preferred.to_csv(OUTPUT_DIR / "table_main_hybrid_comparison.csv", index=False)


def build_main_text_figure_panels() -> None:
    monthly_path = OUTPUT_DIR / "figure_monthly_exceedance_panel.csv"
    if monthly_path.exists():
        monthly = _read_csv(monthly_path)
        reduced_parts: list[pd.DataFrame] = []
        for method, filters in _locked_main_spec_filters().items():
            reduced_parts.append(_apply_locked_filter(monthly, method, filters))
        reduced = pd.concat(reduced_parts, ignore_index=True) if reduced_parts else pd.DataFrame()
        if not reduced.empty:
            reduced.insert(0, "model_label", reduced["method"].map(MAIN_MODEL_LABELS))
            reduced.to_csv(OUTPUT_DIR / "figure_main_monthly_selected_models.csv", index=False)

    covid_path = OUTPUT_DIR / "case_study_covid_window_daily.csv"
    if covid_path.exists():
        covid = _read_csv(covid_path)
        reduced_parts = []
        for method, filters in _locked_main_spec_filters().items():
            reduced_parts.append(_apply_locked_filter(covid, method, filters))
        reduced = pd.concat(reduced_parts, ignore_index=True) if reduced_parts else pd.DataFrame()
        if not reduced.empty:
            reduced.insert(0, "model_label", reduced["method"].map(MAIN_MODEL_LABELS))
            reduced.to_csv(OUTPUT_DIR / "figure_main_covid_selected_models.csv", index=False)


def build_package_metadata(
    baseline_summary: pd.DataFrame,
    topology_summary: pd.DataFrame,
    topology_forecasts: pd.DataFrame,
) -> None:
    topology_run_metadata_path = TOPOLOGY_DIR / "run_metadata.json"
    topology_run_metadata: dict[str, object] = {}
    if topology_run_metadata_path.exists():
        topology_run_metadata = json.loads(topology_run_metadata_path.read_text(encoding="utf-8"))

    test_forecasts = topology_forecasts[topology_forecasts["split"] == "test"].copy()
    package_metadata = {
        "sources": {
            "baseline_dir": str(BASELINE_DIR.relative_to(REPO_ROOT)),
            "topology_dir": str(TOPOLOGY_DIR.relative_to(REPO_ROOT)),
            "formal_tests_dir": str(TOPOLOGY_FORMAL_TESTS_DIR.relative_to(REPO_ROOT)),
        },
        "baseline_test_methods": sorted(
            baseline_summary.loc[baseline_summary["split"] == "test", "method"].astype(str).unique().tolist()
        ),
        "topology_test_methods": sorted(
            topology_summary.loc[topology_summary["split"] == "test", "method"].astype(str).unique().tolist()
        ),
        "test_window": {
            "forecast_start_date": pd.to_datetime(test_forecasts["forecast_date"], utc=False).min().date().isoformat(),
            "forecast_end_date": pd.to_datetime(test_forecasts["forecast_date"], utc=False).max().date().isoformat(),
            "realized_end_date": pd.to_datetime(test_forecasts["realized_date"], utc=False).max().date().isoformat(),
            "n_test_rows": int(test_forecasts.shape[0]),
        },
        "topology_run_metadata": topology_run_metadata,
    }
    (OUTPUT_DIR / "package_metadata.json").write_text(
        json.dumps(package_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_summary = _read_csv(BASELINE_DIR / "summary.csv")
    topology_summary = _read_csv(TOPOLOGY_DIR / "summary.csv")
    topology_forecasts = _read_csv(TOPOLOGY_DIR / "forecasts.csv")

    build_method_tables(baseline_summary, topology_summary)
    build_stress_tables(topology_forecasts)
    build_main_text_tables()
    build_main_text_figure_panels()
    _copy_tree(TOPOLOGY_FORMAL_TESTS_DIR, OUTPUT_DIR / "formal_tests")
    build_package_metadata(baseline_summary, topology_summary, topology_forecasts)
    build_notes()

    print(f"Final results package written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
