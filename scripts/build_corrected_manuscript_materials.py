"""Generate revised-manuscript tables from frozen final-analysis outputs.

This script only reads existing validation, evaluation, null, diagnostic, and
inference outputs.  It does not run a scientific analysis or change submitted
source files.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from tda_risk.parquet_fallback import MiniParquet


OUT = ROOT / "manuscript" / "revised_source"

LOCKED = [
    "fhs_ewma_l094",
    "garch_t_w1500_r20",
    "regime_hs_bins4",
    "rolling_hs_w500",
    "s_euclidean_k250",
    "s_mahalanobis_k500",
    "topology_l250_k100",
    "w_dtw_l250_k750",
    "w_euclidean_l250_k500",
    "w_fpca_l60_k500",
    "wz_dtw_l250_k250",
    "wz_euclidean_l250_k100",
    "wz_fpca_l125_k100",
]
DISPLAY = {
    "fhs_ewma_l094": "FHS--EWMA",
    "garch_t_w1500_r20": "GARCH-$t$",
    "regime_hs_bins4": "Regime-HS",
    "rolling_hs_w500": "Rolling-HS",
    "s_euclidean_k250": "State Euclidean",
    "s_mahalanobis_k500": "State Mahalanobis",
    "topology_l250_k100": "Topology $H_1$",
    "w_dtw_l250_k750": "Raw DTW",
    "w_euclidean_l250_k500": "Raw Euclidean",
    "w_fpca_l60_k500": "Raw FPCA",
    "wz_dtw_l250_k250": "Std. DTW",
    "wz_euclidean_l250_k100": "Std. Euclidean",
    "wz_fpca_l125_k100": "Std. FPCA",
}
PARAM_DISPLAY = {
    "fhs_ewma_l094": r"FHS--EWMA ($\lambda=0.94$)",
    "garch_t_w1500_r20": r"GARCH-$t$ (rolling $W=1{,}500$, refit 20 days)",
    "regime_hs_bins4": r"Regime-HS (4 bins)",
    "rolling_hs_w500": r"Rolling-HS ($W=500$)",
    "s_euclidean_k250": r"State Euclidean ($k=250$)",
    "s_mahalanobis_k500": r"State Mahalanobis ($k=500$)",
    "topology_l250_k100": r"Topology $H_1$ ($L=250$, $k=100$)",
    "w_dtw_l250_k750": r"Raw DTW ($L=250$, $k=750$)",
    "w_euclidean_l250_k500": r"Raw Euclidean ($L=250$, $k=500$)",
    "w_fpca_l60_k500": r"Raw FPCA ($L=60$, $k=500$)",
    "wz_dtw_l250_k250": r"Std. DTW ($L=250$, $k=250$)",
    "wz_euclidean_l250_k100": r"Std. Euclidean ($L=250$, $k=100$)",
    "wz_fpca_l125_k100": r"Std. FPCA ($L=125$, $k=100$)",
    "w_euclidean_l60_k100": r"Raw Euclidean ($L=60$, $k=100$)",
    "w_euclidean_l60_k250": r"Raw Euclidean ($L=60$, $k=250$)",
}
ROW_END = r" \\"


def tex_config(value: str) -> str:
    if value.startswith("gap_hs_"):
        return "Gap-HS"
    return PARAM_DISPLAY.get(value, DISPLAY.get(value, value.replace("_", " ")))


def fmt(value: float, digits: int = 6) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def fmt_loss(value: float, digits: int = 4) -> str:
    """Display a loss in units of 10^{-3}, with the scale in the header."""
    if not np.isfinite(value):
        return "--"
    return f"{value * 1000:.{digits}f}"


def fmt_sci(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    return f"{value:.2e}".replace("e-0", "e-").replace("e+0", "e+")


def table(lines: list[str]) -> str:
    return "\n".join(line.rstrip() for line in lines) + "\n"


def wrap_star_tables(lines: list[str]) -> list[str]:
    """Fit generated double-column tables to the available text width."""
    out: list[str] = []
    in_star = False
    wrapped = False
    for line in lines:
        if line == r"\begin{table*}[!t]":
            in_star = True
        if in_star and line.startswith(r"\begin{tabular}"):
            out.append(r"\begin{adjustbox}{max width=\textwidth}")
            wrapped = True
        out.append(line)
        if in_star and wrapped and line == r"\end{tabular}":
            out.append(r"\end{adjustbox}")
            wrapped = False
        if line == r"\end{table*}":
            in_star = False
    return out


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a flat Parquet file with pandas or the package fallback reader."""
    try:
        return pd.read_parquet(path, columns=columns)
    except (ImportError, ModuleNotFoundError):
        parquet = MiniParquet(path)
        selected = columns or parquet.columns()
        return pd.DataFrame({name: parquet.read_column(name) for name in selected})


def load_null_draws() -> pd.DataFrame:
    main = read_parquet(ROOT / "results" / "final_nulls" / "main_null_draws.parquet")
    if set(main["exclusion_radius"].astype(int)) == {0, 125, 250}:
        return main
    # Compatibility path for a pre-correction checkpoint; the canonical
    # package builder and verifier require the merged all-radius output.
    radius250 = read_parquet(ROOT / "results" / "final_nulls" / "radius250_null_draws.parquet")
    return pd.concat([main, radius250], ignore_index=True)


def real_topology(var: pd.DataFrame, fz: pd.DataFrame) -> dict[int, tuple[float, float]]:
    topo_var = var[var.configuration_id.eq("topology_l250_k100")].set_index("exclusion_radius")
    topo_fz = fz[fz.configuration_id.eq("topology_l250_k100")].set_index("exclusion_radius")
    return {
        int(radius): (float(topo_var.loc[radius, "mean_pinball_99"]), float(topo_fz.loc[radius, "mean_fz0_975"]))
        for radius in (0, 125, 250)
    }


def make_main_tables() -> str:
    inf = ROOT / "results" / "final_inference"
    var = pd.read_csv(inf / "var99_backtests.csv")
    fz = pd.read_csv(inf / "var_es975_backtests.csv")
    es = pd.read_csv(inf / "es_block_bootstrap.csv")
    primary_mcs = pd.read_csv(inf / "mcs_primary_radius_specific.csv")
    paired = pd.read_csv(inf / "paired_primary.csv")
    nulls = load_null_draws()
    real = real_topology(var, fz)
    posthoc = ROOT / "results" / "posthoc_diagnostics"
    recency = pd.read_csv(posthoc / "recency_matched_summary.csv").iloc[0]
    shuffle = pd.read_csv(posthoc / "temporal_shuffle_summary.csv").iloc[0]

    lines: list[str] = ["% Machine-generated table fragment for the revised manuscript.", ""]

    lines += [r"\begin{table*}[!t]", r"\centering", r"\tablefont",
              r"\caption{Topology results on the fixed evaluation panel. Radius denotes the temporal exclusion in trading days. UC, Ind., and CC denote unconditional coverage, exceedance independence, and conditional coverage; Mean FZ is the same-level Fissler--Ziegel score. The $Z_2$ columns report the mean moment contribution $\bar z$ and its studentized statistic $T_{Z_2}$; ES $p$-values use the centered circular block bootstrap. Lower pinball and FZ scores are better.}",
              r"\label{tab:topology_results}",
              r"\begin{tabular}{r r r r r r r r r r r}", r"\toprule",
              r"Radius & Exceed. & Exceedance rate (\%) & Pinball ($\times 10^{-3}$) & UC $p$ & Ind. $p$ & CC $p$ & Mean FZ & $\bar z$ & $T_{Z_2}$ & ES $p$ \\ ", r"\midrule"]
    for radius in (0, 125, 250):
        row = var[(var.exclusion_radius == radius) & var.configuration_id.eq("topology_l250_k100")].iloc[0]
        fzrow = fz[(fz.exclusion_radius == radius) & fz.configuration_id.eq("topology_l250_k100")].iloc[0]
        esrow = es[(es.exclusion_radius == radius) & es.configuration_id.eq("topology_l250_k100")].iloc[0]
        lines.append(
            f"{radius} & {int(row.exceedances)} & {row.exceedance_rate * 100:.2f}\\% & {fmt_loss(row.mean_pinball_99)} & "
            f"{fmt_sci(row.kupiec_uc_pvalue)} & {fmt_sci(row.christoffersen_ind_pvalue)} & "
            f"{fmt_sci(row.christoffersen_cc_pvalue)} & {fzrow.mean_fz0_975:.4f} & "
            f"{esrow.mean_z2_contribution:.3f} & {esrow.es_statistic:.3f} & {esrow.es_pvalue:.3f} " + ROW_END
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    lines += [r"\begin{table*}[!t]", r"\centering", r"\tablefont",
              r"\caption{History-only permutation benchmark for primary VaR(0.99) loss. Each row uses 500 draws; the uniform-subset rows are an analytically equivalent implementation check. $\Delta$ is the topology mean minus the benchmark mean and $p_{\mathrm{FS}}=(1+B)/(D+1)$ is the finite-sample lower-tail randomization $p$-value.}",
              r"\label{tab:null_primary}",
              r"\begin{tabular}{r l r r r r r r r}", r"\toprule",
              r"Radius & Implementation & $D$ & Null mean ($\times 10^{-3}$) & Null 5--95\% ($\times 10^{-3}$) & Null exceedance rate (\%) & Exceedance rate 5--95\% & $\Delta$ ($\times 10^{-3}$) & $p_{\mathrm{FS}}$ \\ ", r"\midrule"]
    for radius in (0, 125, 250):
        for control, label in (("topology_history_permutation", "History permutation"), ("uniform_random_retrieval", "Uniform-subset check")):
            sub = nulls[(nulls.exclusion_radius == radius) & nulls.control.eq(control)]
            pmean = float(sub.mean_pinball_99.mean())
            p05, p95 = np.quantile(sub.mean_pinball_99, [0.05, 0.95])
            rmean = float(sub.exceedance_rate_99.mean())
            r05, r95 = np.quantile(sub.exceedance_rate_99, [0.05, 0.95])
            real_value = real[radius][0]
            tail = int((sub.mean_pinball_99 <= real_value).sum())
            pfs = (1 + tail) / (len(sub) + 1)
            lines.append(
                f"{radius} & {label} & {len(sub)} & {fmt_loss(pmean)} & [{fmt_loss(p05)},{fmt_loss(p95)}] & "
                f"{rmean * 100:.2f}\\% & [{r05 * 100:.2f}\\%,{r95 * 100:.2f}\\%] & {fmt_loss(real_value-pmean, digits=4)} & {pfs:.3f} " + ROW_END
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    lines += [r"\begin{table*}[!t]", r"\centering", r"\tablefont",
              r"\caption{History-only permutation benchmark for the secondary same-level VaR--ES score. Radius denotes the temporal exclusion in trading days. The uniform-subset rows target the same theoretical benchmark and serve as a numerical implementation check, not independent evidence. Lower scores are better.}",
              r"\label{tab:null_secondary}",
              r"\begin{tabular}{r l r r r r r}", r"\toprule",
              r"Radius & Implementation & Null mean & Null 5--95\% & $\Delta$ & $p_{\mathrm{FS}}$ & Real FZ \\ ", r"\midrule"]
    for radius in (0, 125, 250):
        for control, label in (("topology_history_permutation", "History permutation"), ("uniform_random_retrieval", "Uniform-subset check")):
            sub = nulls[(nulls.exclusion_radius == radius) & nulls.control.eq(control)]
            mean = float(sub.mean_fz0_975.mean())
            p05, p95 = np.quantile(sub.mean_fz0_975, [0.05, 0.95])
            real_value = real[radius][1]
            tail = int((sub.mean_fz0_975 <= real_value).sum())
            pfs = (1 + tail) / (len(sub) + 1)
            lines.append(
                f"{radius} & {label} & {mean:.4f} & [{p05:.4f},{p95:.4f}] & {real_value-mean:+.4f} & {pfs:.3f} & {real_value:.4f} " + ROW_END
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    # Validation-only k sensitivity, using the precomputed 250-draw files.
    validation = read_parquet(ROOT / "results" / "final_validation" / "all_configurations.parquet")
    validation = validation[validation.configuration_id.str.match(r"topology_l250_k(?:100|250|500|750|1000)$")]
    kdraws = read_parquet(ROOT / "results" / "final_nulls" / "k_sensitivity_draws.parquet")
    kdraws = kdraws[(kdraws.exclusion_radius == 0) & kdraws.control.isin(["topology_history_permutation", "uniform_random_retrieval"])]
    lines += [r"\begin{table*}[!t]", r"\centering", r"\tablefont",
              r"\caption{Post hoc mechanism diagnostics. Panel A compares the no-exclusion topology forecast with a 500-draw benchmark that exactly matches the topology neighbor-age counts within seven prespecified age bands. Panel B compares each original $H_1$ landscape with landscapes from within-window order shuffles; the 100-window diagnostic uses 25 shuffles per window and a deterministic 30-point subsample of each delay cloud. The two panels address different questions and are not confirmatory model comparisons.}",
              r"\label{tab:posthoc_diagnostics}",
              r"\begin{tabular}{l r r r l r}", r"\toprule",
              r"Diagnostic & Observed & Reference mean & Difference & Reference interval & $p$ \\ ", r"\midrule",
              r"\multicolumn{6}{l}{\textit{Panel A: VaR(0.99) pinball loss, $\times 10^{-3}$}} \\ "]
    lines.append(
        f"Age-matched retrieval & {fmt_loss(recency.topology_mean_pinball_99)} & "
        f"{fmt_loss(recency.null_mean_pinball_99)} & "
        f"{fmt_loss(recency.topology_minus_null_mean_pinball_99)} & "
        f"[{fmt_loss(recency.null_p05_pinball_99)},{fmt_loss(recency.null_p95_pinball_99)}] & "
        f"{recency.lower_tail_randomization_pvalue:.3f} " + ROW_END
    )
    lines += [r"\midrule",
              r"\multicolumn{6}{l}{\textit{Panel B: $L^2$ distance between weighted $H_1$ landscapes}} \\ "]
    lines.append(
        f"Original versus order-shuffled & {shuffle.mean_original_to_shuffle_distance:.4f} & "
        f"{shuffle.mean_shuffle_to_shuffle_distance:.4f} & "
        f"{shuffle.mean_distance_difference:.4f} & "
        f"[{shuffle.bootstrap_ci_low:.4f},{shuffle.bootstrap_ci_high:.4f}] & "
        f"{'<0.001' if shuffle.two_sided_sign_flip_pvalue < 0.001 else f'{shuffle.two_sided_sign_flip_pvalue:.3f}'} " + ROW_END
    )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    lines += [r"\begin{table*}[!t]", r"\centering", r"\tablefont",
              r"\caption{Validation $k$-sensitivity of topology retrieval. Deterministic loss averages the three blocked validation folds. Null summaries use the precomputed 250-draw files at radius zero; $B$ is the randomization numerator, the number of null losses no larger than the deterministic loss. Eligible history is the observed min--max candidate count. Lower pinball loss is better.}",
              r"\label{tab:k_sensitivity}",
              r"\begin{tabular}{r r r r r r r}", r"\toprule",
              r"$k$ & \makecell{Deterministic loss\\($\times 10^{-3}$)} & \makecell{Perm. mean\\($\times 10^{-3}$)} & \makecell{Perm. 5--95\%\\($\times 10^{-3}$)} & \makecell{Uniform-subset mean\\($\times 10^{-3}$)} & $B$ / $p_{\mathrm{FS}}$ & Eligible history \\ ", r"\midrule"]
    for k in (100, 250, 500, 750, 1000):
        det_rows = validation[validation.configuration_id.eq(f"topology_l250_k{k}")]
        real_loss = float(det_rows.groupby("fold").pinball_99.mean().mean())
        perm = kdraws[(kdraws.k == k) & kdraws.control.eq("topology_history_permutation")]
        uniform = kdraws[(kdraws.k == k) & kdraws.control.eq("uniform_random_retrieval")]
        pmean = float(perm.mean_pinball_99.mean())
        p05, p95 = np.quantile(perm.mean_pinball_99, [0.05, 0.95])
        umean = float(uniform.mean_pinball_99.mean())
        tail = int((perm.mean_pinball_99 <= real_loss).sum())
        pfs = (1 + tail) / (len(perm) + 1)
        lo = int(perm.candidate_count_min.min())
        hi = int(perm.candidate_count_max.max())
        lines.append(f"{k} & {fmt_loss(real_loss)} & {fmt_loss(pmean)} & [{fmt_loss(p05)},{fmt_loss(p95)}] & {fmt_loss(umean)} & {tail} / {pfs:.3f} & {lo}--{hi} " + ROW_END)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    paired = paired[paired.comparator.isin([x for x in LOCKED if x != "topology_l250_k100"])]
    lines += [r"\begin{table*}[!t]", r"\centering", r"\tablefont",
              r"\caption{Paired topology-minus-comparator differences in VaR(0.99) pinball loss. Positive values mean higher, hence worse, topology loss. The interval is a 95\% circular block-bootstrap confidence interval; $p$ is the date-aligned Diebold--Mariano test and $p_H$ its Holm-adjusted value.}",
              r"\label{tab:paired_primary}",
              r"\begin{tabular}{r l r r r r}", r"\toprule",
              r"Radius & Comparator & \makecell{Difference\\($\times 10^{-3}$)} & \makecell{95\% block CI\\($\times 10^{-3}$)} & $p$ & $p_H$ \\ ", r"\midrule"]
    for _, row in paired.sort_values(["exclusion_radius", "comparator"]).iterrows():
        lines.append(
            f"{int(row.exclusion_radius)} & {DISPLAY.get(row.comparator, row.comparator)} & {fmt_loss(row.mean_difference_topology_minus_comparator)} & "
            f"[{fmt_loss(row.block_ci_low)},{fmt_loss(row.block_ci_high)}] & {fmt_sci(row.dm_pvalue)} & {fmt_sci(row.holm_pvalue)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    lines += [r"\begin{table*}[!t]", r"\centering", r"\tablefont",
              r"\caption{Primary model confidence set (MCS) membership at $\alpha=0.10$. Each column reports a separate Hansen--Lunde--Nason $R$-statistic analysis using only the stated temporal-exclusion radius and the 13 model-class specifications selected on validation.}",
              r"\label{tab:mcs_primary}",
              r"\begin{tabular}{l c c c}", r"\toprule",
              r"Model & Radius 0 & Radius 125 & Radius 250 \\ ", r"\midrule"]
    for config in LOCKED:
        cells = []
        for radius in (0, 125, 250):
            row = primary_mcs[(primary_mcs.exclusion_radius == radius) & primary_mcs.configuration_id.eq(config)].iloc[0]
            cells.append("Yes" if bool(row.included_in_mcs) else "No")
        lines.append(f"{DISPLAY[config]} & " + " & ".join(cells) + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    return table(wrap_star_tables(lines))


def make_appendix_tables() -> str:
    inf = ROOT / "results" / "final_inference"
    val = ROOT / "results" / "final_validation"
    diag = ROOT / "results" / "final_diagnostics"
    fz = pd.read_csv(inf / "var_es975_backtests.csv")
    es = pd.read_csv(inf / "es_block_bootstrap.csv")
    paired = pd.read_csv(inf / "paired_secondary.csv")
    secondary_mcs = pd.read_csv(inf / "mcs_secondary_radius_specific.csv")
    stability = pd.read_csv(val / "selection_stability.csv")
    fold_summary = pd.read_csv(val / "validation_fold_summary.csv")
    homology = pd.read_csv(val / "topology_homology_ablation.csv")
    exclusion = pd.read_csv(inf / "var99_backtests.csv")
    overlap = pd.read_csv(diag / "neighbor_overlap.csv")
    posthoc = ROOT / "results" / "posthoc_diagnostics"
    recency_profile = pd.read_csv(posthoc / "topology_neighbor_recency.csv")

    lines: list[str] = ["% Machine-generated table fragment for the revised manuscript.", ""]
    lines += [r"\begin{table*}[!t]", r"\centering", r"\tablefont",
              r"\caption{Model specifications selected on the blocked validation sample and held fixed during evaluation.}", r"\label{tab:locked}",
              r"\begin{tabular}{ll}", r"\toprule", r"Role & Configuration \\ ", r"\midrule"]
    roles = ["FHS", "GARCH-$t$", "Regime-HS", "Rolling-HS", "State Euclidean", "State Mahalanobis", "Topology $H_1$", "Raw DTW", "Raw Euclidean", "Raw FPCA", "Std. DTW", "Std. Euclidean", "Std. FPCA"]
    for role, config in zip(roles, LOCKED):
        lines.append(f"{role} & {tex_config(config)} " + ROW_END)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    lines += [r"\begin{table*}[!t]", r"\centering", r"\tablefont",
              r"\caption{Overall winners by blocked validation fold and the corresponding topology-family loss. The final column indicates whether topology is the best model across all candidate families in that fold.}", r"\label{tab:validation}",
              r"\begin{tabularx}{\textwidth}{@{}l X r c@{}}", r"\toprule",
              r"Fold & Winner & \makecell{Topology loss\\($\times 10^{-3}$)} & Topology was overall fold winner? \\ ", r"\midrule"]
    for fold in sorted(stability.loc[stability.analysis.eq("fold_winner"), "fold"].unique()):
        row = stability[(stability.analysis == "fold_winner") & (stability.fold == fold)].iloc[0]
        topo = fold_summary[(fold_summary.fold == int(fold)) & fold_summary.configuration_id.eq("topology_l250_k100")].iloc[0]
        lines.append(f"{int(fold)} & {tex_config(row.configuration_id)} & {fmt_loss(topo.mean_pinball_loss)} & {'Yes' if row.configuration_id == 'topology_l250_k100' else 'No'} " + ROW_END)
    loo = stability[stability.analysis.eq("leave_one_fold_out")].sort_values("fold")
    winner_text = "; ".join(tex_config(x) for x in loo.configuration_id.tolist())
    lines += [r"\midrule", f"Leave-one-fold-out & {winner_text} & -- & No " + ROW_END, r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""]

    lines += [r"\begin{table}[!t]", r"\centering", r"\tablefont", r"\caption{Validation-only homology diagnostics at $L=125$ and $k=100$.}", r"\label{tab:homology}", r"\begin{tabular}{l r}", r"\toprule", r"Representation & Mean pinball ($\times 10^{-3}$) \\ ", r"\midrule"]
    for label, prefix in (("$H_0$ only", "topology_h0_only"), ("$H_1$ primary", "topology_h1"), ("$H_0+H_1$", "topology_h0h1")):
        rows = homology[homology.configuration_id.str.startswith(prefix)]
        lines.append(f"{label} & {fmt_loss(rows.mean_pinball_loss.mean())} " + ROW_END)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    lines += [r"\begin{table*}[!t]", r"\centering", r"\tablefont", r"\caption{Exclusion-radius and gap historical-simulation diagnostics. Lower pinball loss is better; rates are percentages.}", r"\label{tab:exclusion}", r"\begin{tabular}{r l r r}", r"\toprule", r"Radius & Method & Pinball ($\times 10^{-3}$) & Exceedance rate (\%) \\ ", r"\midrule"]
    chosen = exclusion[exclusion.configuration_id.str.contains("gap_hs|topology_l250_k100|rolling_hs_w500", regex=True)]
    for radius in (125, 250):
        for config in [f"gap_hs_w250_r{radius}", "topology_l250_k100", "rolling_hs_w500"]:
            row = chosen[(chosen.exclusion_radius == radius) & chosen.configuration_id.eq(config)].iloc[0]
            lines.append(f"{radius} & {tex_config(config)} & {fmt_loss(row.mean_pinball_99)} & {row.exceedance_rate * 100:.2f}\\% " + ROW_END)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    lines += [r"\begin{table}[!t]", r"\centering", r"\caption{Radius-250 representation diagnostics: mean Jaccard overlap between the topology neighbor set and each preprocessing-matched comparator.}", r"\label{tab:overlap}", r"\begin{tabular}{l r}", r"\toprule", r"Comparator & Mean Jaccard \\ ", r"\midrule"]
    for config in ["wz_euclidean_l250_k100", "wz_dtw_l250_k250", "wz_fpca_l125_k100"]:
        row = overlap[(overlap.exclusion_radius == 250) & overlap.comparator.eq(config)].iloc[0]
        lines.append(f"{DISPLAY[config]} & {row.mean_jaccard_overlap:.4f} " + ROW_END)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    profile = dict(zip(recency_profile["measure"], recency_profile["value"]))
    lines += [r"\begin{table}[!t]", r"\centering", r"\tablefont",
              r"\caption{Post hoc age profile of topology neighbors under no temporal exclusion. Ages are measured in trading-day index differences. Jaccard overlap compares the selected 100-neighbor set with the most recent eligible 100 or 250 candidate dates.}",
              r"\label{tab:recency_profile}",
              r"\begin{tabular}{l r}", r"\toprule", r"Measure & Value \\ ", r"\midrule",
              f"Mean age & {profile['mean_age']:.1f} " + ROW_END,
              f"Age 5th percentile & {profile['age_q05']:.0f} " + ROW_END,
              f"Median age & {profile['age_q50']:.0f} " + ROW_END,
              f"Age 95th percentile & {profile['age_q95']:.0f} " + ROW_END,
              rf"Share aged $\leq 20$ days & {profile['share_age_le_20']*100:.2f}\% " + ROW_END,
              rf"Share aged $\leq 60$ days & {profile['share_age_le_60']*100:.2f}\% " + ROW_END,
              rf"Share aged $\leq 125$ days & {profile['share_age_le_125']*100:.2f}\% " + ROW_END,
              rf"Share aged $\leq 250$ days & {profile['share_age_le_250']*100:.2f}\% " + ROW_END,
              f"Mean Jaccard, recent 100 & {profile['mean_jaccard_recent_100']:.4f} " + ROW_END,
              f"Mean Jaccard, recent 250 & {profile['mean_jaccard_recent_250']:.4f} " + ROW_END,
              r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    lines += [r"\begin{table*}[!t]", r"\centering", r"\tablefont", r"\caption{Mean same-level Value-at-Risk--Expected-Shortfall scores for all model-class specifications selected on validation. FZ denotes the Fissler--Ziegel score; lower values are preferred.}", r"\label{tab:fz_all_finalists}", r"\begin{tabular}{l r r r}", r"\toprule", r"Model & Radius 0 & Radius 125 & Radius 250 \\ ", r"\midrule"]
    for config in LOCKED:
        values = []
        for radius in (0, 125, 250):
            row = fz[(fz.exclusion_radius == radius) & fz.configuration_id.eq(config)].iloc[0]
            values.append(f"{row.mean_fz0_975:.4f}")
        lines.append(f"{DISPLAY[config]} & " + " & ".join(values) + ROW_END)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    for radius in (0, 125, 250):
        lines += [r"\begin{table*}[!t]", r"\centering", r"\tablefont", f"\\caption{{ES backtest results for validation-selected model-class representatives at radius {radius}. $\\bar z$ is the mean $Z_2$ moment contribution, $T_{{Z_2}}$ is its studentized statistic, and $p$ is the centered circular block-bootstrap $p$-value.}}", f"\\label{{tab:es_backtest_all_finalists_r{radius}}}", r"\begin{tabular}{l r r r r}", r"\toprule", r"Model & Mean FZ & $\bar z$ & $T_{Z_2}$ & $p$ \\ ", r"\midrule"]
        for config in LOCKED:
            fzrow = fz[(fz.exclusion_radius == radius) & fz.configuration_id.eq(config)].iloc[0]
            esrow = es[(es.exclusion_radius == radius) & es.configuration_id.eq(config)].iloc[0]
            lines.append(f"{DISPLAY[config]} & {fzrow.mean_fz0_975:.4f} & {esrow.mean_z2_contribution:.3f} & {esrow.es_statistic:.3f} & {esrow.es_pvalue:.4f} " + ROW_END)
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    paired = paired[paired.comparator.isin(LOCKED + ["gap_hs_w250_r125", "gap_hs_w250_r250"])]
    lines += [r"\begin{table*}[!t]", r"\centering", r"\tablefont", r"\caption{Secondary paired topology-minus-comparator differences under the same-level Value-at-Risk--Expected-Shortfall score. Positive differences indicate worse topology performance.}", r"\label{tab:paired_secondary}", r"\begin{tabular}{r l r r r r}", r"\toprule", r"Radius & Comparator & Difference & 95\% block CI & $p$ & $p_H$ \\ ", r"\midrule"]
    for _, row in paired.sort_values(["exclusion_radius", "comparator"]).iterrows():
        label = tex_config(str(row.comparator))
        lines.append(f"{int(row.exclusion_radius)} & {label} & {row.mean_difference_topology_minus_comparator:.4f} & [{row.block_ci_low:.4f},{row.block_ci_high:.4f}] & {fmt_sci(row.dm_pvalue)} & {fmt_sci(row.holm_pvalue)} " + ROW_END)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    lines += [r"\begin{table*}[!t]", r"\centering", r"\tablefont", r"\caption{Corrected secondary same-level VaR--ES MCS membership at $\alpha=0.10$. Lower FZ score is better.}", r"\label{tab:mcs_secondary}", r"\begin{tabular}{l c c c}", r"\toprule", r"Model & Radius 0 & Radius 125 & Radius 250 \\ ", r"\midrule"]
    for config in LOCKED:
        cells = []
        for radius in (0, 125, 250):
            row = secondary_mcs[(secondary_mcs.exclusion_radius == radius) & secondary_mcs.configuration_id.eq(config)].iloc[0]
            cells.append("Yes" if bool(row.included_in_mcs) else "No")
        lines.append(f"{DISPLAY[config]} & " + " & ".join(cells) + ROW_END)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]

    sensitivity_rows: list[dict[str, object]] = []
    for window_length in (60, 125, 250):
        candidates = fold_summary[
            fold_summary.configuration_id.str.match(
                rf"topology_l{window_length}_k(?:100|250|500|750|1000)$"
            )
        ].copy()
        grouped = (
            candidates.groupby("configuration_id", as_index=False)
            .agg(
                mean_loss=("mean_pinball_loss", "mean"),
                worst_fold_loss=("mean_pinball_loss", "max"),
            )
        )
        best_id = grouped.sort_values(
            ["mean_loss", "worst_fold_loss", "configuration_id"]
        ).iloc[0]["configuration_id"]
        best = candidates[candidates.configuration_id.eq(best_id)].sort_values("fold")
        folds = {str(row.fold): float(row.mean_pinball_loss) for _, row in best.iterrows()}
        sensitivity_rows.append(
            {
                "window_length": window_length,
                "best_k": int(str(best_id).rsplit("_k", 1)[1]),
                "mean_pinball_loss": float(best.mean_pinball_loss.mean()),
                "loss_2012": folds["2012"],
                "loss_2013": folds["2013"],
                "loss_2014": folds["2014"],
                "worst_fold_loss": float(best.mean_pinball_loss.max()),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(val / "topology_window_sensitivity.csv", index=False)
    lines += [r"\begin{table*}[!t]", r"\centering", r"\tablefont",
              r"\caption{Topology window-length sensitivity on the blocked validation sample. For each $L$, the table reports the best $k$ under mean VaR(0.99) pinball loss. Neither $L$ nor $k$ is selected on the fixed evaluation panel.}",
              r"\label{tab:window_sensitivity}",
              r"\begin{tabular}{r r r r r r r}", r"\toprule",
              r"$L$ & Best $k$ & Mean loss ($\times 10^{-3}$) & 2012 & 2013 & 2014 & Worst fold ($\times 10^{-3}$) \\ ", r"\midrule"]
    for _, row in sensitivity.iterrows():
        lines.append(
            f"{int(row.window_length)} & {int(row.best_k)} & {fmt_loss(row.mean_pinball_loss)} & "
            f"{fmt_loss(row.loss_2012)} & {fmt_loss(row.loss_2013)} & {fmt_loss(row.loss_2014)} & "
            f"{fmt_loss(row.worst_fold_loss)} " + ROW_END
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return table(wrap_star_tables(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "generated_main_tables.tex").write_text(make_main_tables(), encoding="utf-8")
    (OUT / "generated_appendix_tables.tex").write_text(make_appendix_tables(), encoding="utf-8")
    (OUT / "generated_tables.tex").write_text(
        "% Compatibility stub. Tables are split by manuscript location.\n"
        "% Machine-generated table fragments for the revised manuscript.\n",
        encoding="utf-8",
    )
    print("wrote corrected manuscript table fragments")


if __name__ == "__main__":
    main()
