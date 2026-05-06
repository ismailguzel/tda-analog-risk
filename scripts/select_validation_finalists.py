from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORECASTS = REPO_ROOT / "results" / "topology_pipeline" / "forecasts.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "validation_finalists"
TARGET_VAR_EXCEEDANCE = 0.01

PARAM_COLUMNS_BY_METHOD: dict[str, tuple[str, ...]] = {
    "rolling_hs": ("window",),
    "fhs_ewma": ("lambda",),
    "regime_hs": ("regime_bins",),
    "garch_t": ("dist", "sim_draws", "refit_interval"),
    "euclidean_knn": ("k",),
    "window_euclidean_knn": ("k", "window_length"),
    "random_knn": ("k",),
    "mahalanobis_knn": ("k",),
    "dtw_window": ("k", "window_length"),
    "fpca_window": ("k", "window_length"),
    "topology_knn": (
        "k",
        "window_length",
        "topology_input_mode",
        "feature_mode",
        "alpha",
    ),
    "topology_placebo_knn": (
        "k",
        "window_length",
        "topology_input_mode",
        "feature_mode",
        "alpha",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one locked finalist per method family on validation and report the frozen "
            "test comparison."
        )
    )
    parser.add_argument(
        "--forecasts",
        type=Path,
        default=DEFAULT_FORECASTS,
        help="Path to a forecasts.csv file containing validation and test rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where finalist tables will be written.",
    )
    parser.add_argument(
        "--target-var-exceedance",
        type=float,
        default=TARGET_VAR_EXCEEDANCE,
        help="Target VaR exceedance rate used in the validation selection score.",
    )
    parser.add_argument(
        "--include-topology-placebo",
        action="store_true",
        help="Retain topology placebo as its own finalist family in the output tables.",
    )
    return parser.parse_args()


def fz_style_score(
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


def _family_param_columns(frame: pd.DataFrame, method: str) -> list[str]:
    configured = PARAM_COLUMNS_BY_METHOD.get(method, ())
    return [column for column in configured if column in frame.columns]


def _selection_score(mean_fz: float, exceedance_99: float, target_exceedance: float) -> float:
    return abs(exceedance_99 - target_exceedance) * 1000.0 + mean_fz


def _summarize_candidates(frame: pd.DataFrame, target_exceedance: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, method_frame in frame.groupby("method", sort=True):
        param_columns = _family_param_columns(method_frame, method)
        group_columns = ["method", *param_columns]

        for group_key, candidate in method_frame.groupby(group_columns, dropna=False, sort=True):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row: dict[str, object] = dict(zip(group_columns, group_key))
            losses = candidate["realized_loss"].to_numpy(dtype=float)
            var_values = candidate["VaR_99"].to_numpy(dtype=float)
            es_values = candidate["ES_975"].to_numpy(dtype=float)
            scores = fz_style_score(losses, var_values, es_values)
            exceedance_99 = float(candidate["var_exceedance_99"].mean())
            exceedance_es = float(candidate["es_exceedance_975"].mean())
            row.update(
                {
                    "n_forecasts": int(candidate.shape[0]),
                    "mean_scenarios": float(candidate["scenario_count"].mean()),
                    "mean_var_99": float(candidate["VaR_99"].mean()),
                    "mean_es_975": float(candidate["ES_975"].mean()),
                    "exceedance_rate_99": exceedance_99,
                    "exceedance_rate_es_975": exceedance_es,
                    "mean_realized_loss": float(candidate["realized_loss"].mean()),
                    "fz_style_mean_score": float(scores.mean()),
                    "selection_score": _selection_score(
                        mean_fz=float(scores.mean()),
                        exceedance_99=exceedance_99,
                        target_exceedance=target_exceedance,
                    ),
                }
            )
            rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No candidate rows were available for validation selection.")
    return summary.sort_values(["method", "selection_score", "fz_style_mean_score"])


def _choose_finalists(candidate_summary: pd.DataFrame, include_topology_placebo: bool) -> pd.DataFrame:
    keep = candidate_summary.copy()
    if not include_topology_placebo:
        keep = keep[keep["method"] != "topology_placebo_knn"].copy()
    finalists = keep.groupby("method", as_index=False, sort=True).first()
    return finalists.sort_values(["selection_score", "fz_style_mean_score", "method"])


def _filter_by_candidate(frame: pd.DataFrame, finalist_row: pd.Series) -> pd.DataFrame:
    method = str(finalist_row["method"])
    filtered = frame[frame["method"] == method].copy()
    for column in _family_param_columns(filtered, method):
        value = finalist_row[column]
        if pd.isna(value):
            filtered = filtered[filtered[column].isna()]
        else:
            filtered = filtered[filtered[column] == value]
    return filtered


def _build_frozen_test_table(
    forecasts: pd.DataFrame,
    finalists: pd.DataFrame,
    target_exceedance: float,
) -> pd.DataFrame:
    test = forecasts[forecasts["split"] == "test"].copy()
    rows: list[dict[str, object]] = []
    for _, finalist in finalists.iterrows():
        candidate = _filter_by_candidate(test, finalist)
        if candidate.empty:
            continue
        losses = candidate["realized_loss"].to_numpy(dtype=float)
        var_values = candidate["VaR_99"].to_numpy(dtype=float)
        es_values = candidate["ES_975"].to_numpy(dtype=float)
        scores = fz_style_score(losses, var_values, es_values)
        row = finalist.to_dict()
        row.update(
            {
                "test_n_forecasts": int(candidate.shape[0]),
                "test_mean_scenarios": float(candidate["scenario_count"].mean()),
                "test_mean_var_99": float(candidate["VaR_99"].mean()),
                "test_mean_es_975": float(candidate["ES_975"].mean()),
                "test_exceedance_rate_99": float(candidate["var_exceedance_99"].mean()),
                "test_exceedance_rate_es_975": float(candidate["es_exceedance_975"].mean()),
                "test_mean_realized_loss": float(candidate["realized_loss"].mean()),
                "test_fz_style_mean_score": float(scores.mean()),
                "test_selection_score": _selection_score(
                    mean_fz=float(scores.mean()),
                    exceedance_99=float(candidate["var_exceedance_99"].mean()),
                    target_exceedance=target_exceedance,
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["test_selection_score", "test_fz_style_mean_score", "method"]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    forecasts = pd.read_csv(args.forecasts)
    validation = forecasts[forecasts["split"] == "validation"].copy()
    if validation.empty:
        raise RuntimeError("No validation rows found in forecasts.csv.")

    validation_candidates = _summarize_candidates(validation, args.target_var_exceedance)
    validation_candidates.to_csv(args.output_dir / "validation_candidates.csv", index=False)

    finalists = _choose_finalists(
        validation_candidates,
        include_topology_placebo=args.include_topology_placebo,
    )
    finalists.to_csv(args.output_dir / "validation_finalists.csv", index=False)

    frozen_test = _build_frozen_test_table(
        forecasts,
        finalists,
        target_exceedance=args.target_var_exceedance,
    )
    frozen_test.to_csv(args.output_dir / "test_finalists.csv", index=False)

    print(f"Validation finalist selection completed. Outputs written to: {args.output_dir}")
    print("\nSelected finalists:")
    print(finalists.to_string(index=False))
    print("\nFrozen test comparison:")
    print(frozen_test.to_string(index=False))


if __name__ == "__main__":
    main()