from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build matched-date test comparison tables for topology-only, FHS, "
            "and scenario-mixture hybrids."
        )
    )
    parser.add_argument(
        "--topology-forecasts",
        type=Path,
        default=Path("results/topology_pipeline/forecasts.csv"),
        help="Path to finalist topology forecasts.csv.",
    )
    parser.add_argument(
        "--scenario-coverage-forecasts",
        type=Path,
        default=Path("results/topology_fhs_scenario_mixture/hybrid_forecasts.csv"),
        help="Path to scenario-mixture forecasts for coverage_then_fz run.",
    )
    parser.add_argument(
        "--scenario-weighted-forecasts",
        type=Path,
        default=Path("results/topology_fhs_scenario_mixture_weightedcombo/hybrid_forecasts.csv"),
        help="Path to scenario-mixture forecasts for weighted_combo run.",
    )
    parser.add_argument(
        "--scenario-coverage-config",
        type=Path,
        default=Path("results/topology_fhs_scenario_mixture/selected_hybrid_config.csv"),
        help="Path to selected config for coverage_then_fz run.",
    )
    parser.add_argument(
        "--scenario-weighted-config",
        type=Path,
        default=Path("results/topology_fhs_scenario_mixture_weightedcombo/selected_hybrid_config.csv"),
        help="Path to selected config for weighted_combo run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hybrid_comparison_table.csv"),
        help="Output CSV file path.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    return pd.read_csv(path)


def _read_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return _read_csv(path)


def _summarize_forecasts(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        raise ValueError("Cannot summarize empty forecast frame.")
    return {
        "n_forecasts": int(frame.shape[0]),
        "mean_var_99": float(frame["VaR_99"].mean()),
        "mean_es_975": float(frame["ES_975"].mean()),
        "exceedance_rate_99": float((frame["realized_loss"] > frame["VaR_99"]).mean()),
        "exceedance_rate_es_975": float((frame["realized_loss"] > frame["ES_975"]).mean()),
        "mean_fz_style": float(frame["fz_style_score"].mean()) if "fz_style_score" in frame else "",
    }


def _legacy_full_sample_rows(topology_forecasts: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, label in (("topology_knn", "topology_only_fullsample"), ("fhs_ewma", "fhs_fullsample")):
        frame = topology_forecasts[
            (topology_forecasts["split"] == "test") & (topology_forecasts["method"] == method)
        ].copy()
        if frame.empty:
            continue
        summary = _summarize_forecasts(frame)
        rows.append(
            {
                "comparison_group": "full_sample_reference",
                "model_label": label,
                "source_run": "topology_pipeline",
                "selection_objective": "",
                "selected_lambda": "",
                "selected_weight_topology": "",
                **summary,
            }
        )
    return rows


def _scenario_run_rows(
    topology_forecasts: pd.DataFrame,
    scenario_forecasts: pd.DataFrame,
    config: pd.DataFrame,
    *,
    run_name: str,
) -> list[dict[str, object]]:
    cfg = config.iloc[0]
    selected_lambda = float(cfg.get("selected_lambda", float("nan")))
    selected_weight = float(cfg.get("selected_hybrid_weight_topology", float("nan")))
    selection_objective = str(cfg.get("selection_objective", ""))

    scenario_test = scenario_forecasts[scenario_forecasts["split"] == "test"].copy()
    if scenario_test.empty:
        raise ValueError(f"Missing test rows in scenario forecasts: {run_name}")
    scenario_test["realized_date"] = pd.to_datetime(scenario_test["realized_date"], utc=False)
    date_set = set(scenario_test["realized_date"])

    topo_test = topology_forecasts[
        (topology_forecasts["split"] == "test")
        & (topology_forecasts["method"] == "topology_knn")
    ].copy()
    fhs_test = topology_forecasts[
        (topology_forecasts["split"] == "test")
        & (topology_forecasts["method"] == "fhs_ewma")
        & (topology_forecasts["lambda"].astype(float) == selected_lambda)
    ].copy()
    topo_test["realized_date"] = pd.to_datetime(topo_test["realized_date"], utc=False)
    fhs_test["realized_date"] = pd.to_datetime(fhs_test["realized_date"], utc=False)

    topo_matched = topo_test[topo_test["realized_date"].isin(date_set)].copy()
    fhs_matched = fhs_test[fhs_test["realized_date"].isin(date_set)].copy()
    if topo_matched.empty or fhs_matched.empty:
        raise ValueError(f"Matched-date component rows missing for {run_name}")

    rows: list[dict[str, object]] = []
    for label, frame in (
        ("topology_only_matched", topo_matched),
        ("fhs_matched_selected_lambda", fhs_matched),
        ("scenario_mixture", scenario_test),
    ):
        summary = _summarize_forecasts(frame)
        rows.append(
            {
                "comparison_group": run_name,
                "model_label": label,
                "source_run": run_name if label == "scenario_mixture" else "topology_pipeline",
                "selection_objective": selection_objective,
                "selected_lambda": selected_lambda,
                "selected_weight_topology": selected_weight,
                **summary,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    topology_forecasts = _read_csv(args.topology_forecasts)
    rows: list[dict[str, object]] = []
    rows.extend(_legacy_full_sample_rows(topology_forecasts))

    coverage_forecasts = _read_optional(args.scenario_coverage_forecasts)
    coverage_config = _read_optional(args.scenario_coverage_config)
    if coverage_forecasts is not None and coverage_config is not None:
        rows.extend(
            _scenario_run_rows(
                topology_forecasts,
                coverage_forecasts,
                coverage_config,
                run_name="scenario_mixture_coverage_then_fz",
            )
        )

    weighted_forecasts = _read_optional(args.scenario_weighted_forecasts)
    weighted_config = _read_optional(args.scenario_weighted_config)
    if weighted_forecasts is not None and weighted_config is not None:
        rows.extend(
            _scenario_run_rows(
                topology_forecasts,
                weighted_forecasts,
                weighted_config,
                run_name="scenario_mixture_weighted_combo",
            )
        )

    if not rows:
        raise RuntimeError("No rows built; check input paths.")

    out = pd.DataFrame(rows)
    out = out.sort_values(["comparison_group", "model_label"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Wrote comparison table: {args.output}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
