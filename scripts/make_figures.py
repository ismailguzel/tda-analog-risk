"""Generate manuscript figures from pre-computed result files.

Active manuscript figures produced here
--------------------------------------
fig3_landscape.pdf            : H1 persistence landscape for three market regimes
fig4_var_timeseries.pdf       : Realized loss vs VaR forecast bands (test window)
fig5_monthly_exceedance.pdf   : Main-text monthly exceedance panel for selected models
fig6_covid_exceedance_bar.pdf : Main-text COVID-window daily comparison for selected models
fig7_w_sensitivity.pdf        : Scenario-mixture weight sensitivity (Appendix)

Legacy optional figures retained in code but excluded from the current
manuscript build: cumulative exceedance, stress heatmap, and hybrid matched bar.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── global style ──────────────────────────────────────────────────────────────
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "lines.linewidth": 1.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": "0.92",
        "grid.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# colourblind-friendly palette
C = {
    "topology":  "#2166ac",   # blue
    "mixture":   "#1a9850",   # green
    "fhs":       "#d6604d",   # red-orange
    "rolling":   "#8073ac",   # purple
    "garch":     "#4d4d4d",   # dark gray
    "euclidean": "#b2abd2",   # light purple
    "nominal":   "#000000",   # black
    "loss":      "#999999",   # mid gray
    "shade":     "#fef0d9",   # stress band fill
}

STRESS_BANDS = [
    ("2020-02-15", "2020-04-30", "COVID crash"),
    ("2022-01-01", "2022-12-31", "2022 rates shock"),
]


def _package_dir(results_dir: Path) -> Path:
    return results_dir / "final_results_package"


def _shade_stress(ax: plt.Axes, df_dates: pd.DatetimeIndex) -> None:
    for start, end, _ in STRESS_BANDS:
        ax.axvspan(
            pd.Timestamp(start),
            pd.Timestamp(end),
            alpha=0.12,
            color="#fc8d59",
            linewidth=0,
            zorder=0,
        )


def _annotate_stress_bands(ax: plt.Axes, y_frac: float = 0.97) -> None:
    import matplotlib.transforms as mtransforms
    for start, end, label in STRESS_BANDS:
        center = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        # x in data coords, y in axes fraction — text stays just inside top edge
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(center, y_frac, label, fontsize=7, color="#e34a33",
                ha="center", va="top", transform=trans)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path,
                        default=REPO_ROOT / "results")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "figures")
    return parser.parse_args()


# ── Figure 1: VaR time series ─────────────────────────────────────────────────

def fig1_var_timeseries(results_dir: Path, out: Path) -> None:
    fc = pd.read_csv(results_dir / "topology_pipeline" / "forecasts.csv",
                     parse_dates=["forecast_date"])
    test = fc[fc["split"] == "test"].copy()

    topo = test[test["method"] == "topology_knn"].set_index("forecast_date")
    fhs  = test[test["method"] == "fhs_ewma"].set_index("forecast_date")

    mix_fc = pd.read_csv(
        results_dir / "topology_fhs_scenario_mixture" / "hybrid_forecasts.csv",
        parse_dates=["forecast_date"],
    )
    mix = mix_fc[mix_fc["split"] == "test"].set_index("forecast_date")

    dates = topo.index
    loss  = topo["realized_loss"].values * 100   # pct

    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    # realized loss as gray fill
    ax.fill_between(dates, 0, loss, where=loss > 0,
                    color=C["loss"], alpha=0.75, linewidth=0, zorder=1)
    ax.fill_between(dates, loss, 0, where=loss < 0,
                    color="#d1e5f0", linewidth=0, alpha=0.5, zorder=1)

    # VaR lines
    l_topo, = ax.plot(dates, topo["VaR_99"].values * 100,
                      color=C["topology"], lw=1.5, label="$w$-Topology", zorder=3)
    l_fhs,  = ax.plot(fhs.index, fhs["VaR_99"].values * 100,
                      color=C["fhs"], lw=1.2, linestyle="--", label="FHS-EWMA", zorder=3)
    l_mix,  = ax.plot(mix.index, mix["VaR_99"].values * 100,
                      color=C["mixture"], lw=1.2, linestyle=":", label="Scenario mixture", zorder=3)

    # exceedance markers
    exc_topo = topo[topo["var_exceedance_99"] == 1]
    ax.scatter(exc_topo.index, exc_topo["realized_loss"].values * 100,
               color=C["topology"], s=14, zorder=5, marker="x", linewidths=1.0)

    # stress bands
    _shade_stress(ax, dates)

    ax.axhline(0, color="0.35", linewidth=0.9, zorder=0)
    ax.set_ylabel("Daily portfolio loss (%, log-return scale)")
    ax.set_xlabel("")
    ax.grid(False)

    # legend: solid gray line for fill, lines for VaR
    from matplotlib.lines import Line2D
    loss_line = Line2D([0], [0], color=C["loss"], lw=4, solid_capstyle="butt", label="Realized loss")
    ax.legend(handles=[loss_line, l_topo, l_fhs, l_mix],
              loc="lower right", ncol=1, handlelength=2.0)
    ax.set_xlim(dates.min(), dates.max())

    # annotate stress bands
    _annotate_stress_bands(ax)

    fig.tight_layout()
    fig.savefig(out / "fig4_var_timeseries.pdf")
    plt.close(fig)
    print("  fig4_var_timeseries.pdf")


# ── Legacy Figure: Cumulative exceedance ─────────────────────────────────────

def fig2_cumulative_exc(results_dir: Path, out: Path) -> None:
    fc = pd.read_csv(results_dir / "topology_pipeline" / "forecasts.csv",
                     parse_dates=["forecast_date"])
    test = fc[fc["split"] == "test"].copy()

    mix_fc = pd.read_csv(
        results_dir / "topology_fhs_scenario_mixture" / "hybrid_forecasts.csv",
        parse_dates=["forecast_date"],
    )
    mix_test = mix_fc[mix_fc["split"] == "test"].copy()

    focus = {
        "topology_knn":       ("$w$-Topology",      C["topology"],  "-",   1.6),
        "fhs_ewma":           ("FHS-EWMA",           C["fhs"],       "--",  1.3),
        "rolling_hs":         ("Rolling HS",         C["rolling"],   "-.",  1.1),
        "garch_t":            ("GARCH(1,1)-$t$",     C["garch"],     ":",   1.1),
        "euclidean_knn":      ("$s$-Euclidean",    C["euclidean"], "--",  1.0),
    }

    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    all_dates = (
        test[test["method"] == "topology_knn"]
        .sort_values("forecast_date")["forecast_date"]
    )
    n_total = len(all_dates)
    nominal_x = [all_dates.iloc[0], all_dates.iloc[-1]]
    nominal_y = [0, 0.01 * n_total]
    ax.plot(nominal_x, nominal_y, color=C["nominal"], lw=1.0, linestyle="--",
            label="Nominal (1%)", zorder=2)

    for method, (label, color, ls, lw) in focus.items():
        sub = test[test["method"] == method].sort_values("forecast_date")
        if sub.empty:
            continue
        cum = sub["var_exceedance_99"].cumsum().values
        ax.plot(sub["forecast_date"].values, cum,
                color=color, ls=ls, lw=lw, label=label, zorder=3)

    # scenario mixture
    mix_sorted = mix_test.sort_values("forecast_date")
    cum_mix = mix_sorted["var_exceedance_99"].cumsum().values
    ax.plot(mix_sorted["forecast_date"].values, cum_mix,
            color=C["mixture"], ls="-", lw=1.6, label="Scenario mixture", zorder=4)

    for start, end, _ in STRESS_BANDS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.10, color="#fc8d59", linewidth=0, zorder=0)

    ax.set_ylabel("Cumulative VaR(99%) exceedances")
    ax.set_xlabel("")
    ax.legend(loc="upper right", ncol=1)
    ax.set_xlim(all_dates.min(), all_dates.max())
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out / "fig2_cumulative_exc.pdf")
    plt.close(fig)
    print("  fig2_cumulative_exc.pdf")


# ── Legacy Figure: Stress-period heatmap ─────────────────────────────────────

def fig3_stress_heatmap(results_dir: Path, out: Path) -> None:
    stress_path = results_dir / "extended_stress" / "table_stress_extended.csv"
    if stress_path.exists():
        stress = pd.read_csv(stress_path)
        stress_period_col = "stress_period_key"
    else:
        stress = pd.read_csv(_package_dir(results_dir) / "table_stress_period_summary.csv")
        stress_period_col = "stress_period"
    mix_fc = pd.read_csv(
        results_dir / "topology_fhs_scenario_mixture" / "hybrid_forecasts.csv",
        parse_dates=["forecast_date"],
    )
    mix_fc = mix_fc[mix_fc["split"] == "test"].copy()

    windows = {
        "covid_crash":         ("2020-02-15", "2020-04-30", "COVID crash\n(Feb–Apr 2020)"),
        "inflation_rates_shock": ("2022-01-01", "2022-12-31", "Rates shock\n(2022)"),
        "2018_q4_selloff":     ("2018-10-01", "2018-12-31", "2018 Q4\nselloff"),
        "2023_banking_stress": ("2023-03-08", "2023-05-05", "2023 SVB\nstress"),
        "2020_rates_crash":    ("2020-03-09", "2020-03-31", "2020 rates\nflash crash"),
    }

    col_methods = [
        ("topology_knn",             "Topology\n$k$-NN"),
        ("hybrid_topology_fhs_scenario", "Scenario\nmixture"),
        ("fhs_ewma",                 "FHS-EWMA"),
        ("garch_t",                  "GARCH"),
        ("regime_hs",                "Regime-HS"),
        ("euclidean_knn",            "Euclidean\n$k$-NN"),
    ]

    row_labels = [v[2] for v in windows.values()]
    col_labels = [v[1] for v in col_methods]
    n_rows, n_cols = len(windows), len(col_methods)
    matrix = np.full((n_rows, n_cols), np.nan)

    for i, (wkey, (wstart, wend, _)) in enumerate(windows.items()):
        for j, (mname, _) in enumerate(col_methods):
            if mname == "hybrid_topology_fhs_scenario":
                sub = mix_fc[
                    (mix_fc["forecast_date"] >= wstart) &
                    (mix_fc["forecast_date"] <= wend)
                ]
                if len(sub):
                    matrix[i, j] = sub["var_exceedance_99"].mean() * 100
            else:
                row = stress[
                    (stress[stress_period_col] == wkey) &
                    (stress["method"] == mname)
                ]
                if len(row):
                    matrix[i, j] = float(row["exceedance_rate_99"].iloc[0]) * 100

    fig, ax = plt.subplots(figsize=(7.5, 3.4))

    # custom diverging colormap anchored at nominal 1%
    from matplotlib.colors import TwoSlopeNorm
    vmax = max(float(np.nanmax(matrix)), 5.0)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)
    cmap = plt.cm.RdYlGn_r

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, ha="center")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, va="center")
    ax.tick_params(length=0)

    # annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            v = matrix[i, j]
            if not np.isnan(v):
                txt = f"{v:.1f}"
                brightness = norm(v)
                color = "white" if brightness > 0.7 or brightness < 0.25 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color=color, fontweight="normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("VaR(99%) exc. rate (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.axhline(y=norm(1.0), color="black", linewidth=1.0, linestyle="--")

    ax.set_xlabel("")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "fig3_stress_heatmap.pdf")
    plt.close(fig)
    print("  fig3_stress_heatmap.pdf")


# ── Figure 3: Persistence landscape ──────────────────────────────────────────

def fig4_landscape(results_dir: Path, out: Path) -> None:
    try:
        from dataclasses import replace
        from tda_risk.config import default_pipeline_config
        from tda_risk.data import build_research_panel
        from tda_risk.features import attach_state_zscores, build_state_panel
        from tda_risk.topology import (
            _select_embedding_parameters,
            _get_topology_window_matrix,
            _window_to_topology_vector,
        )
    except ImportError as exc:
        print(f"  fig3_landscape.pdf  SKIPPED ({exc})")
        return

    config = default_pipeline_config()
    from dataclasses import replace
    topology = replace(
        config.topology,
        window_lengths=(125,),
        alphas=(0.0,),
        feature_modes=("landscape_h1_top3_weighted",),
        input_modes=("portfolio_only",),
        landscape_num_steps=200,
        landscape_topk_weights=(1.0, 0.5, 0.25),
        n_jobs=1,
    )

    print("  fig4: loading panel…", end="", flush=True)
    panel_result = build_research_panel(config.data)
    panel = panel_result.panel
    data = build_state_panel(panel)
    data = attach_state_zscores(data, state_columns=config.state_columns,
                                min_history=config.min_zscore_history)
    print(" done")

    tau, m = _select_embedding_parameters(
        data, topology, config.splits.warmup_end, "portfolio_only"
    )
    windows = _get_topology_window_matrix(data, 125, "portfolio_only")

    dates_of_interest = {
        "Calm (Jun 2019)":       "2019-06-03",
        "Pre-crisis (Jan 2020)": "2020-01-20",
        "Crisis (Mar 2020)":     "2020-03-16",
    }
    colors_dates = ["#2166ac", "#f4a582", "#d6604d"]
    linestyles   = ["-", "--", "-"]

    x_grid = np.linspace(
        topology.landscape_range[0],
        topology.landscape_range[1],
        topology.landscape_num_steps,
    )

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    for (label, date_str), color, ls in zip(
        dates_of_interest.items(), colors_dates, linestyles
    ):
        ts = pd.Timestamp(date_str)
        if ts not in data.index:
            ts = data.index[data.index.get_indexer([ts], method="nearest")[0]]
        pos = data.index.get_loc(ts)
        window = windows[pos]
        if not np.isfinite(window).all():
            print(f"  fig4: skipping {date_str} (window not available)")
            continue
        try:
            vec = _window_to_topology_vector(
                window, tau=tau, embedding_dimension=m,
                feature_mode="landscape_h1_top3_weighted",
                topology=topology,
            )
        except Exception as exc:
            print(f"  fig4: skipping {date_str} ({exc})")
            continue
        ax.plot(x_grid, vec, color=color, ls=ls, lw=1.6, label=label)

    ax.set_xlabel("Filtration parameter")
    ax.set_ylabel("Weighted $H_1$ landscape $\\Lambda_t$")
    ax.legend(loc="upper right")
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out / "fig3_landscape.pdf")
    plt.close(fig)
    print("  fig3_landscape.pdf")


# ── Figure 7: Scenario-mixture weight sensitivity (Appendix) ─────────────────

def fig5_w_sensitivity(results_dir: Path, out: Path) -> None:
    scan_path = (
        results_dir / "topology_fhs_scenario_mixture" / "validation_weight_scan.csv"
    )
    if not scan_path.exists():
        print("  fig7_w_sensitivity.pdf  SKIPPED (validation_weight_scan.csv not found)")
        return

    scan = pd.read_csv(scan_path)

    fig, axes = plt.subplots(2, 1, figsize=(3.6, 5.8), sharex=True, sharey=False)

    lam_styles = {0.94: ("-", "o", "#2166ac"), 0.97: ("--", "s", "#d6604d")}

    for lam, (ls, mk, color) in lam_styles.items():
        sub = scan[scan["lambda"] == lam].sort_values("hybrid_weight_topology")
        w   = sub["hybrid_weight_topology"].values
        exc = sub["validation_exceedance_rate_99"].values * 100
        fz  = sub["validation_fz_style_mean"].values

        axes[0].plot(w, exc,  color=color, ls=ls, marker=mk, ms=4, lw=1.4,
                     label=f"$\\lambda={lam}$")
        axes[1].plot(w, fz, color=color, ls=ls, marker=mk, ms=4, lw=1.4,
                     label=f"$\\lambda={lam}$")

    # nominal 1% line and selected w
    axes[0].axhline(1.0,  color="0.5", lw=0.8, ls=":", label="Nominal 1%")
    for ax in axes:
        ax.axvline(0.10, color="#1a9850", lw=1.2, ls="--", label="Selected $w=0.10$")

    axes[0].set_ylabel("Exceedance rate (%)")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Topology weight $w$")
    axes[1].set_ylabel("FZ score (lower = better)")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.set_xlim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(out / "fig7_w_sensitivity.pdf")
    plt.close(fig)
    print("  fig7_w_sensitivity.pdf")


# ── Main-text figure set from final_results_package ──────────────────────────

def fig_monthly_exceedance(results_dir: Path, out: Path) -> None:
    forecasts = pd.read_csv(
        results_dir / "topology_pipeline" / "forecasts.csv",
        parse_dates=["forecast_date", "realized_date"],
    )
    forecasts = forecasts[forecasts["split"] == "test"].copy()
    selected = pd.concat(
        [
            forecasts[(forecasts["method"] == "topology_knn") & (forecasts["k"] == 1000)],
            forecasts[(forecasts["method"] == "fhs_ewma") & (forecasts["lambda"] == 0.94)],
            forecasts[(forecasts["method"] == "rolling_hs") & (forecasts["window"] == 250)],
        ],
        ignore_index=True,
    )
    selected["month"] = selected["forecast_date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        selected.groupby(["month", "method"], dropna=False)
        .agg(exceedance_rate_99=("var_exceedance_99", "mean"))
        .reset_index()
    )
    monthly["model_label"] = monthly["method"].map(
        {
            "topology_knn": "$w$-Topology",
            "fhs_ewma": "FHS-EWMA",
            "rolling_hs": "Rolling HS",
        }
    )

    styles = {
        "$w$-Topology": (C["topology"], "-", 1.7),
        "FHS-EWMA": (C["fhs"], "--", 1.3),
        "Rolling HS": (C["rolling"], "-.", 1.2),
    }

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    for label, (color, linestyle, linewidth) in styles.items():
        sub = monthly[monthly["model_label"] == label].sort_values("month")
        if sub.empty:
            continue
        ax.plot(
            sub["month"],
            sub["exceedance_rate_99"] * 100,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )

    ax.axhline(1.0, color=C["nominal"], lw=1.0, ls=":", label="Nominal 1%")
    _shade_stress(ax, monthly["month"])

    ax.set_ylabel("Monthly VaR(99%) exceedance rate (%)")
    ax.set_xlabel("")
    ax.set_xlim(monthly["month"].min(), monthly["month"].max())
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", ncol=1)

    _annotate_stress_bands(ax)

    fig.tight_layout()
    fig.savefig(out / "fig5_monthly_exceedance.pdf")
    plt.close(fig)
    print("  fig5_monthly_exceedance.pdf")


def fig_covid_exceedance_bar(results_dir: Path, out: Path) -> None:
    covid = pd.read_csv(
        _package_dir(results_dir) / "figure_main_covid_selected_models.csv",
        parse_dates=["forecast_date", "realized_date"],
    )
    selected = covid[
        ((covid["method"] == "topology_knn") & (covid["k"] == 1000))
        | ((covid["method"] == "fhs_ewma") & (covid["lambda"] == 0.94))
        | ((covid["method"] == "rolling_hs") & (covid["window"] == 250))
    ].copy()

    label_map = {
        "topology_knn": "$w$-Topology",
        "fhs_ewma": "FHS-EWMA",
        "rolling_hs": "Rolling HS",
    }
    color_map = {
        "$w$-Topology": C["topology"],
        "FHS-EWMA": C["fhs"],
        "Rolling HS": C["rolling"],
    }
    linestyle_map = {
        "$w$-Topology": "-",
        "FHS-EWMA": "--",
        "Rolling HS": "-.",
    }

    realized = (
        selected[["realized_date", "realized_loss"]]
        .drop_duplicates()
        .sort_values("realized_date")
    )

    fig, ax = plt.subplots(figsize=(7.3, 3.6))
    loss = realized["realized_loss"].to_numpy() * 100
    dates = realized["realized_date"]
    ax.bar(dates, loss, width=1.0, color=C["loss"], edgecolor="none", alpha=0.8, label="Realized loss", zorder=1)

    for method, label in label_map.items():
        sub = selected[selected["method"] == method].sort_values("realized_date")
        ax.plot(
            sub["realized_date"],
            sub["VaR_99"] * 100,
            color=color_map[label],
            linestyle=linestyle_map[label],
            linewidth=1.5 if method == "topology_knn" else 1.25,
            label=label,
            zorder=3,
        )
        exc = sub[sub["var_exceedance_99"] == 1]
        if not exc.empty:
            ax.scatter(
                exc["realized_date"],
                exc["realized_loss"] * 100,
                color=color_map[label],
                s=16,
                marker="x",
                linewidths=0.9,
                zorder=4,
            )

    ax.axhline(0, color="0.55", linewidth=0.6, zorder=0)
    ax.set_ylabel("Daily loss / VaR (%)")
    ax.set_xlabel("")
    ax.set_xlim(dates.min(), dates.max())
    ax.legend(loc="lower right", ncol=1)

    fig.tight_layout()
    fig.savefig(out / "fig6_covid_exceedance_bar.pdf")
    plt.close(fig)
    print("  fig6_covid_exceedance_bar.pdf")


def fig_hybrid_matched_bar(results_dir: Path, out: Path) -> None:
    hybrid = pd.read_csv(_package_dir(results_dir) / "table_main_hybrid_comparison.csv")

    colors = [C["fhs"], C["topology"], C["mixture"]]
    labels = hybrid["display_label"].tolist()
    values = hybrid["exceedance_rate_99"].to_numpy() * 100

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.axhline(1.0, color=C["nominal"], lw=1.0, ls=":", label="Nominal 1%")
    ax.set_ylabel("VaR(99%) exceedance rate (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, max(values.max() * 1.25, 1.6))
    ax.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "fig_hybrid_matched_bar.pdf")
    plt.close(fig)
    print("  fig_hybrid_matched_bar.pdf")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to: {args.output_dir.resolve()}")
    fig1_var_timeseries(args.results_dir, args.output_dir)
    # Legacy manuscript-external figures kept available on demand:
    # fig2_cumulative_exc(args.results_dir, args.output_dir)
    # fig3_stress_heatmap(args.results_dir, args.output_dir)
    fig4_landscape(args.results_dir, args.output_dir)
    fig5_w_sensitivity(args.results_dir, args.output_dir)
    fig_monthly_exceedance(args.results_dir, args.output_dir)
    fig_covid_exceedance_bar(args.results_dir, args.output_dir)
    # fig_hybrid_matched_bar(args.results_dir, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
