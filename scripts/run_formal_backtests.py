from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal backtests (Kupiec, Christoffersen, FZ-style, MCS-style)."
    )
    parser.add_argument(
        "--forecasts",
        type=Path,
        default=Path("results/topology_pipeline/forecasts.csv"),
        help="Path to forecasts.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final_results_package/formal_tests"),
        help="Directory for formal backtest outputs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Forecast split to evaluate (default: test).",
    )
    parser.add_argument(
        "--alpha-mcs",
        type=float,
        default=0.10,
        help="Significance level for MCS-style elimination.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=1000,
        help="Bootstrap repetitions for MCS-style p-values.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260426,
        help="Random seed for bootstrap procedures.",
    )
    return parser.parse_args()


def kupiec_uc_test(hits: np.ndarray, alpha_exceed: float) -> tuple[float, float]:
    n = hits.size
    x = int(hits.sum())
    if n == 0:
        return float("nan"), float("nan")
    phat = x / n
    if phat <= 0.0:
        ll_alt = (n - x) * np.log(np.maximum(1.0 - phat, 1e-12))
    elif phat >= 1.0:
        ll_alt = x * np.log(np.maximum(phat, 1e-12))
    else:
        ll_alt = x * np.log(phat) + (n - x) * np.log(1.0 - phat)
    ll_null = x * np.log(alpha_exceed) + (n - x) * np.log(1.0 - alpha_exceed)
    lr_uc = -2.0 * (ll_null - ll_alt)
    p_value = 1.0 - chi2.cdf(lr_uc, df=1)
    return float(lr_uc), float(p_value)


def christoffersen_independence_test(hits: np.ndarray) -> tuple[float, float]:
    if hits.size < 2:
        return float("nan"), float("nan")

    prev = hits[:-1].astype(int)
    curr = hits[1:].astype(int)
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())

    n0 = n00 + n01
    n1 = n10 + n11
    p01 = n01 / n0 if n0 > 0 else 0.0
    p11 = n11 / n1 if n1 > 0 else 0.0
    p = (n01 + n11) / max(n0 + n1, 1)

    ll_indep = (
        n00 * np.log(np.maximum(1.0 - p, 1e-12))
        + n01 * np.log(np.maximum(p, 1e-12))
        + n10 * np.log(np.maximum(1.0 - p, 1e-12))
        + n11 * np.log(np.maximum(p, 1e-12))
    )
    ll_markov = (
        n00 * np.log(np.maximum(1.0 - p01, 1e-12))
        + n01 * np.log(np.maximum(p01, 1e-12))
        + n10 * np.log(np.maximum(1.0 - p11, 1e-12))
        + n11 * np.log(np.maximum(p11, 1e-12))
    )
    lr_ind = -2.0 * (ll_indep - ll_markov)
    p_value = 1.0 - chi2.cdf(lr_ind, df=1)
    return float(lr_ind), float(p_value)


def christoffersen_cc_test(hits: np.ndarray, alpha_exceed: float) -> tuple[float, float]:
    lr_uc, _ = kupiec_uc_test(hits, alpha_exceed)
    lr_ind, _ = christoffersen_independence_test(hits)
    if np.isnan(lr_uc) or np.isnan(lr_ind):
        return float("nan"), float("nan")
    lr_cc = lr_uc + lr_ind
    p_value = 1.0 - chi2.cdf(lr_cc, df=2)
    return float(lr_cc), float(p_value)


def fz_style_score(
    losses: np.ndarray,
    var_values: np.ndarray,
    es_values: np.ndarray,
    alpha_var: float = 0.99,
    alpha_es: float = 0.975,
) -> np.ndarray:
    # Project convention score:
    # - Quantile pinball at VaR(99%)
    # - Tail shortfall loss at ES(97.5%) on exceedance region
    # This is an FZ-style joint score surrogate for ranking with mixed alpha levels.
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


def dm_test_against_reference(
    score_matrix: pd.DataFrame,
    reference_method: str,
) -> pd.DataFrame:
    if reference_method not in score_matrix.columns:
        raise ValueError(f"Reference method not found: {reference_method}")

    ref = score_matrix[reference_method].to_numpy(dtype=float)
    out_rows: list[dict[str, float | str]] = []
    n = ref.size
    for method in score_matrix.columns:
        if method == reference_method:
            continue
        diff = score_matrix[method].to_numpy(dtype=float) - ref
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        if std_diff <= 1e-12:
            z_stat = 0.0
            p_value = 1.0
        else:
            z_stat = mean_diff / (std_diff / np.sqrt(n))
            p_value = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
        out_rows.append(
            {
                "reference_method": reference_method,
                "method": method,
                "n_obs": n,
                "mean_score_diff_vs_ref": mean_diff,
                "z_stat": z_stat,
                "p_value_two_sided": p_value,
            }
        )
    columns = [
        "reference_method",
        "method",
        "n_obs",
        "mean_score_diff_vs_ref",
        "z_stat",
        "p_value_two_sided",
    ]
    if not out_rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(out_rows).sort_values("p_value_two_sided")


def mcs_style_survivor_set(
    score_matrix: pd.DataFrame,
    alpha: float,
    reps: int,
    seed: int,
) -> tuple[list[str], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    methods = list(score_matrix.columns)
    current = methods.copy()
    trace_rows: list[dict[str, object]] = []

    while len(current) > 1:
        sub = score_matrix[current]
        means = sub.mean(axis=0)
        best_method = str(means.idxmin())
        worst_method = str(means.idxmax())

        d = sub[worst_method].to_numpy(dtype=float) - sub[best_method].to_numpy(dtype=float)
        d_bar = float(np.mean(d))
        # One-sided bootstrap p-value for H0: E[d] <= 0 vs H1: E[d] > 0
        centered = d - d_bar
        n = centered.size
        boot_means = np.empty(reps, dtype=float)
        for b in range(reps):
            idx = rng.integers(0, n, size=n)
            boot_means[b] = centered[idx].mean()
        p_value = float(np.mean(boot_means >= d_bar))

        trace_rows.append(
            {
                "best_method": best_method,
                "worst_method": worst_method,
                "mean_diff_worst_minus_best": d_bar,
                "bootstrap_p_value": p_value,
                "n_methods_before_step": len(current),
            }
        )

        if p_value < alpha:
            current.remove(worst_method)
        else:
            break

    return current, pd.DataFrame(trace_rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    forecasts = pd.read_csv(args.forecasts)
    data = forecasts[forecasts["split"] == args.split].copy()
    if data.empty:
        raise RuntimeError(f"No rows found for split={args.split}.")

    method_rows: list[dict[str, float | str | int]] = []
    score_rows: list[pd.DataFrame] = []

    for method, group in data.groupby("method", sort=True):
        group = group.sort_values("forecast_date")
        losses = group["realized_loss"].to_numpy(dtype=float)
        var_vals = group["VaR_99"].to_numpy(dtype=float)
        es_vals = group["ES_975"].to_numpy(dtype=float)
        hits = (losses > var_vals).astype(int)

        lr_uc, p_uc = kupiec_uc_test(hits, alpha_exceed=0.01)
        lr_ind, p_ind = christoffersen_independence_test(hits)
        lr_cc, p_cc = christoffersen_cc_test(hits, alpha_exceed=0.01)

        scores = fz_style_score(losses, var_vals, es_vals)
        score_rows.append(
            pd.DataFrame(
                {
                    "realized_date": group["realized_date"].to_numpy(),
                    "method": method,
                    "score": scores,
                }
            )
        )

        method_rows.append(
            {
                "split": args.split,
                "method": method,
                "n_obs": int(group.shape[0]),
                "var_exceedance_rate": float(hits.mean()),
                "kupiec_lr_uc": lr_uc,
                "kupiec_p_value": p_uc,
                "christoffersen_lr_ind": lr_ind,
                "christoffersen_p_value_ind": p_ind,
                "christoffersen_lr_cc": lr_cc,
                "christoffersen_p_value_cc": p_cc,
                "fz_style_mean_score": float(np.mean(scores)),
            }
        )

    formal_table = pd.DataFrame(method_rows).sort_values("fz_style_mean_score")
    formal_table.to_csv(args.output_dir / "formal_backtests_by_method.csv", index=False)

    score_long = pd.concat(score_rows, ignore_index=True)
    score_matrix = (
        score_long.pivot_table(
            index="realized_date",
            columns="method",
            values="score",
            aggfunc="mean",
        )
        .dropna(axis=0, how="any")
        .sort_index()
    )
    fz_rank = (
        score_matrix.mean(axis=0).rename("fz_style_mean_score").sort_values().reset_index()
    )
    fz_rank.columns = ["method", "fz_style_mean_score"]
    fz_rank.to_csv(args.output_dir / "fz_style_ranking.csv", index=False)

    reference_method = "topology_knn" if "topology_knn" in score_matrix.columns else score_matrix.columns[0]
    dm_table = dm_test_against_reference(score_matrix, reference_method=reference_method)
    dm_table.to_csv(args.output_dir / "dm_vs_reference.csv", index=False)

    survivors, mcs_trace = mcs_style_survivor_set(
        score_matrix,
        alpha=args.alpha_mcs,
        reps=args.bootstrap_reps,
        seed=args.seed,
    )
    pd.DataFrame({"method": survivors}).to_csv(
        args.output_dir / "mcs_style_survivors.csv",
        index=False,
    )
    mcs_trace.to_csv(args.output_dir / "mcs_style_trace.csv", index=False)

    notes = f"""# Formal Backtest Notes

Inputs:
- forecasts file: `{args.forecasts}`
- split: `{args.split}`

Outputs:
- `formal_backtests_by_method.csv`
- `fz_style_ranking.csv`
- `dm_vs_reference.csv`
- `mcs_style_survivors.csv`
- `mcs_style_trace.csv`

Important:
- Kupiec and Christoffersen are run on VaR(99%) exceedances.
- The joint score is an FZ-style surrogate for project convention
  VaR(99%) + ES(97.5%) (mixed alpha levels).
- `mcs_style_survivors.csv` is a bootstrap elimination set, intended as an
  MCS-style ranking aid for this stage.
"""
    (args.output_dir / "README.md").write_text(notes, encoding="utf-8")

    print(f"Formal backtests written to: {args.output_dir}")


if __name__ == "__main__":
    main()
