"""Generate a synthetic mechanism figure comparing DTW and topology retrieval.

The figure uses a single common candidate pool and selects one query window
for which DTW and topology retrieve different analogue regions under the same
search set. The goal is to visualize that the disagreement comes from the
distance definition rather than from method-specific candidate filtering.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

from dtaidistance import dtw

try:
    from ripser import ripser
    from persim import PersLandscapeApprox
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        "Topology dependencies missing. Install project requirements first."
    ) from exc


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.92",
        "grid.linewidth": 0.6,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

WINDOW_COLORS = {
    "query": "#2166ac",
    "dtw": "#d95f0e",
    "topology": "#1b9e77",
}


@dataclass(frozen=True)
class DemoConfig:
    n_total: int = 1100
    window: int = 125
    tau: int = 2
    embed_dim: int = 3
    landscape_steps: int = 180
    landscape_start: float = 0.0
    landscape_stop: float = 0.09
    landscape_topk: int = 3
    landscape_weights: tuple[float, ...] = (1.0, 0.5, 0.25)
    dtw_band: int = 10
    exclusion_radius: int = 90
    train_end: int = 839
    query_start_min: int = 860
    query_start_max: int = 975
    disagreement_percentile_cutoff: float = 10.0


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "figures",
    )
    return parser.parse_args()


def _resample_signal(values: np.ndarray, n_out: int) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, values.size)
    x_new = np.linspace(0.0, 1.0, n_out)
    return np.interp(x_new, x_old, values)


def _base_loop_template(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    envelope = 1.0 + 0.18 * np.sin(2.0 * np.pi * t - 0.35)
    return (
        envelope
        * (
            0.0125 * np.sin(2.0 * np.pi * 2.9 * t + 0.20 * np.sin(2.0 * np.pi * t))
            + 0.0062 * np.sin(2.0 * np.pi * 5.8 * t + 0.45)
        )
        + 0.0022 * np.sin(2.0 * np.pi * 8.7 * t + 1.05)
        + 0.0018 * np.sin(2.0 * np.pi * 1.05 * t + 0.8)
    )


def _localized_pulses(t: np.ndarray, centers: tuple[float, ...], amps: tuple[float, ...], width: float) -> np.ndarray:
    out = np.zeros_like(t)
    for center, amp in zip(centers, amps):
        out += amp * np.exp(-0.5 * ((t - center) / width) ** 2)
    return out


def _elastic_variant(base: np.ndarray) -> np.ndarray:
    n = base.size
    t = np.linspace(0.0, 1.0, n)
    warped = np.clip(t + 0.12 * np.sin(2.0 * np.pi * t) - 0.06 * np.sin(4.0 * np.pi * t), 0.0, 1.0)
    low_component = 0.0105 * np.sin(2.0 * np.pi * 2.9 * t + 0.25 * np.sin(2.0 * np.pi * t))
    signal = 0.78 * np.interp(warped, t, low_component)
    signal += 0.0018 * np.sin(2.0 * np.pi * 1.1 * t + 0.6)
    signal += 0.0038 * np.maximum(t - 0.38, 0.0)
    signal -= 0.0030 * np.maximum(t - 0.74, 0.0)
    signal += 0.0016 * np.sign(np.sin(2.0 * np.pi * 1.4 * t + 0.4))
    signal += _localized_pulses(t, (0.28, 0.67), (0.0034, -0.0036), width=0.038)
    return signal


def _topology_variant(base: np.ndarray) -> np.ndarray:
    n = base.size
    t = np.linspace(0.0, 1.0, n)
    mod = 1.0 + 0.38 * np.sin(2.0 * np.pi * t + 0.30)
    warped = np.clip(t + 0.20 * np.sin(2.0 * np.pi * t + 0.55), 0.0, 1.0)
    signal = np.interp(warped, t, base) * mod
    signal += 0.0036 * np.sign(np.sin(2.0 * np.pi * 2.9 * t + 0.2))
    signal += 0.0041 * np.sin(2.0 * np.pi * 5.8 * t + 1.0)
    signal += 0.0028 * np.sin(2.0 * np.pi * 8.7 * t + 0.6)
    signal += 0.0024 * (t - 0.5)
    signal += _localized_pulses(t, (0.18, 0.81), (0.0018, -0.0016), width=0.018)
    return signal


def _query_variant(base: np.ndarray) -> np.ndarray:
    n = base.size
    t = np.linspace(0.0, 1.0, n)
    warped = np.clip(t + 0.06 * np.sin(2.0 * np.pi * t + 0.18), 0.0, 1.0)
    signal = np.interp(warped, t, base)
    signal *= 1.0 + 0.20 * np.sin(2.0 * np.pi * t + 1.1)
    signal += 0.0041 * np.sin(2.0 * np.pi * 5.8 * t + 0.9)
    signal += 0.0030 * np.sin(2.0 * np.pi * 8.7 * t + 0.35)
    signal += 0.0021 * np.sign(np.sin(2.0 * np.pi * 2.9 * t + 0.5))
    signal += _localized_pulses(t, (0.22, 0.76), (0.0018, -0.0016), width=0.014)
    return signal


def _filler_regime(n: int, rng: np.random.Generator, slope: float, seasonal_freq: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return (
        slope * (t - 0.5)
        + 0.0045 * np.sin(2.0 * np.pi * seasonal_freq * np.arange(n) + 0.7)
        + rng.normal(scale=0.0025, size=n)
    )


def build_synthetic_returns(cfg: DemoConfig, seed: int) -> tuple[pd.Series, dict[str, tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    base = _base_loop_template(220)

    loop_topology = _topology_variant(base) + rng.normal(scale=0.0012, size=base.size)
    filler_a = _filler_regime(170, rng, slope=0.020, seasonal_freq=0.040)
    loop_dtw = _elastic_variant(base) + rng.normal(scale=0.0010, size=base.size)
    filler_b = _filler_regime(230, rng, slope=-0.015, seasonal_freq=0.025)
    loop_query = _query_variant(base) + rng.normal(scale=0.0010, size=base.size)
    filler_tail = _filler_regime(40, rng, slope=0.010, seasonal_freq=0.060)

    values = np.concatenate([loop_topology, filler_a, loop_dtw, filler_b, loop_query, filler_tail])
    if values.size != cfg.n_total:
        raise ValueError(f"Synthetic series length mismatch: got {values.size}, expected {cfg.n_total}.")

    regions = {
        "topology_loop": (0, 219),
        "filler_a": (220, 389),
        "dtw_loop": (390, 609),
        "filler_b": (610, 839),
        "query_loop": (840, 1059),
        "tail": (1060, 1099),
    }
    idx = pd.RangeIndex(start=0, stop=cfg.n_total, step=1, name="t")
    return pd.Series(values, index=idx, name="return"), regions


def delay_embed(window: np.ndarray, tau: int, embed_dim: int) -> np.ndarray:
    usable = len(window) - tau * (embed_dim - 1)
    if usable < 5:
        raise ValueError("Window too short for embedding parameters.")
    cols = [window[i * tau : i * tau + usable] for i in range(embed_dim)]
    return np.column_stack(cols)


def h1_weighted_topk_landscape(window: np.ndarray, cfg: DemoConfig) -> np.ndarray:
    cloud = delay_embed(window, tau=cfg.tau, embed_dim=cfg.embed_dim)
    dgms = ripser(cloud, maxdim=1)["dgms"]
    d1 = dgms[1]
    finite = d1[np.isfinite(d1[:, 1])]
    if finite.size == 0:
        return np.zeros(cfg.landscape_steps, dtype=float)

    landscape = PersLandscapeApprox(
        start=cfg.landscape_start,
        stop=cfg.landscape_stop,
        num_steps=cfg.landscape_steps,
        dgms=[np.empty((0, 2), dtype=float), finite],
        hom_deg=1,
    )
    values = np.asarray(landscape.values, dtype=float)
    if values.size == 0:
        return np.zeros(cfg.landscape_steps, dtype=float)

    topk = values[: min(cfg.landscape_topk, values.shape[0])]
    if topk.shape[0] < cfg.landscape_topk:
        pad = np.zeros((cfg.landscape_topk - topk.shape[0], cfg.landscape_steps), dtype=float)
        topk = np.vstack([topk, pad])

    weights = np.asarray(cfg.landscape_weights[: cfg.landscape_topk], dtype=float)
    weights = weights / weights.sum()
    return np.sum(topk * weights[:, None], axis=0)


def h1_diagram(window: np.ndarray, cfg: DemoConfig) -> np.ndarray:
    cloud = delay_embed(window, tau=cfg.tau, embed_dim=cfg.embed_dim)
    dgms = ripser(cloud, maxdim=1)["dgms"]
    d1 = dgms[1]
    finite = d1[np.isfinite(d1[:, 1])]
    if finite.size == 0:
        return np.empty((0, 2), dtype=float)
    return finite


def _standardize_rows(history: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = history.mean(axis=0)
    std = history.std(axis=0, ddof=0)
    std = np.where(std == 0.0, 1.0, std)
    return (history - mean) / std, (current - mean) / std


def _region_name(start_idx: int, regions: dict[str, tuple[int, int]]) -> str:
    for name, (lo, hi) in regions.items():
        if lo <= start_idx <= hi:
            return name
    return "unknown"


def pick_query_and_neighbours(
    series: pd.Series,
    cfg: DemoConfig,
    regions: dict[str, tuple[int, int]],
) -> dict[str, object]:
    values = series.to_numpy(dtype=float)
    starts = np.arange(0, len(values) - cfg.window + 1)
    query_starts = starts[(starts >= cfg.query_start_min) & (starts <= cfg.query_start_max)]
    train_starts = starts[starts + cfg.window - 1 <= cfg.train_end]
    relevant_starts = np.unique(np.concatenate([train_starts, query_starts]))
    window_map = {
        int(start): values[start : start + cfg.window] for start in relevant_starts
    }
    landscape_map = {
        int(start): h1_weighted_topk_landscape(window_map[int(start)], cfg) for start in relevant_starts
    }

    if query_starts.size == 0:
        raise ValueError("No query windows in the requested range.")

    def evaluate_query(q_start: int) -> dict[str, object]:
        query_window = window_map[q_start]
        candidate_starts = train_starts[np.abs(train_starts - q_start) > cfg.exclusion_radius]
        history_windows = np.vstack([window_map[int(s)] for s in candidate_starts])

        dtw_distances = np.asarray(
            dtw.distance_matrix_fast(
                np.vstack([query_window, history_windows]).astype(np.double, copy=False),
                block=((0, 1), (1, history_windows.shape[0] + 1)),
                compact=True,
                window=cfg.dtw_band,
                parallel=False,
                use_pruning=True,
            ),
            dtype=float,
        )

        query_land = landscape_map[q_start]
        history_lands = np.vstack([landscape_map[int(s)] for s in candidate_starts])
        history_lands_std, query_land_std = _standardize_rows(history_lands, query_land)
        topo_distances = np.linalg.norm(history_lands_std - query_land_std, axis=1)

        i_dtw = int(np.argmin(dtw_distances))
        i_topo = int(np.argmin(topo_distances))
        dtw_start = int(candidate_starts[i_dtw])
        topo_start = int(candidate_starts[i_topo])

        return {
            "query_start": q_start,
            "candidate_starts": candidate_starts,
            "query_window": query_window,
            "query_land": query_land,
            "history_windows": history_windows,
            "history_lands": history_lands,
            "dtw_distances": dtw_distances,
            "topology_distances": topo_distances,
            "dtw_start": dtw_start,
            "topology_start": topo_start,
            "dtw_window": window_map[dtw_start],
            "topology_window": window_map[topo_start],
            "dtw_land": landscape_map[dtw_start],
            "topology_land": landscape_map[topo_start],
            "dtw_distance": float(dtw_distances[i_dtw]),
            "topology_distance": float(topo_distances[i_topo]),
            "dtw_percentile": float(np.mean(dtw_distances <= dtw_distances[i_dtw]) * 100.0),
            "topology_percentile": float(np.mean(topo_distances <= topo_distances[i_topo]) * 100.0),
            "dtw_region": _region_name(dtw_start, regions),
            "topology_region": _region_name(topo_start, regions),
        }

    evaluations = [evaluate_query(int(q_start)) for q_start in query_starts]
    preferred = [
        ev
        for ev in evaluations
        if ev["dtw_region"] == "dtw_loop"
        and ev["topology_region"] == "topology_loop"
        and ev["dtw_percentile"] <= cfg.disagreement_percentile_cutoff
        and ev["topology_percentile"] <= cfg.disagreement_percentile_cutoff
    ]
    if preferred:
        return min(
            preferred,
            key=lambda ev: float(ev["dtw_percentile"]) + float(ev["topology_percentile"]),
        )

    differing = [ev for ev in evaluations if ev["dtw_region"] != ev["topology_region"]]
    if differing:
        return min(
            differing,
            key=lambda ev: float(ev["dtw_percentile"]) + float(ev["topology_percentile"]),
        )

    return min(
        evaluations,
        key=lambda ev: float(ev["dtw_percentile"]) + float(ev["topology_percentile"]),
    )


def plot_figure(
    series: pd.Series,
    regions: dict[str, tuple[int, int]],
    res: dict[str, object],
    cfg: DemoConfig,
    output_path: Path,
) -> None:
    q0 = int(res["query_start"])
    d0 = int(res["dtw_start"])
    t0 = int(res["topology_start"])
    q = np.asarray(res["query_window"])
    dw = np.asarray(res["dtw_window"])
    tw = np.asarray(res["topology_window"])
    ql = np.asarray(res["query_land"])
    dl = np.asarray(res["dtw_land"])
    tl = np.asarray(res["topology_land"])
    x_land = np.linspace(cfg.landscape_start, cfg.landscape_stop, cfg.landscape_steps)
    candidate_starts = np.asarray(res["candidate_starts"])
    dtw_distances = np.asarray(res["dtw_distances"])
    topo_distances = np.asarray(res["topology_distances"])

    fig = plt.figure(figsize=(11.8, 7.6))
    grid = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.30)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[1, 1])

    ax0.set_title("(A) Synthetic series with selected windows")
    ax0.plot(series.index, series.values, color="#3a3a3a", lw=1.0)
    ax0.axvspan(q0, q0 + cfg.window, color=WINDOW_COLORS["query"], alpha=0.18, linewidth=0)
    ax0.axvspan(d0, d0 + cfg.window, color=WINDOW_COLORS["dtw"], alpha=0.18, linewidth=0)
    ax0.axvspan(t0, t0 + cfg.window, color=WINDOW_COLORS["topology"], alpha=0.18, linewidth=0)
    ymin, ymax = ax0.get_ylim()
    ytxt = ymax - 0.12 * (ymax - ymin)
    ax0.text(q0 + cfg.window / 2, ytxt, "Q", color=WINDOW_COLORS["query"], ha="center", va="center", fontweight="bold")
    ax0.text(d0 + cfg.window / 2, ytxt, "D", color=WINDOW_COLORS["dtw"], ha="center", va="center", fontweight="bold")
    ax0.text(t0 + cfg.window / 2, ytxt, "T", color=WINDOW_COLORS["topology"], ha="center", va="center", fontweight="bold")
    ax0.set_xlabel("Time index")
    ax0.set_ylabel("Return")
    ax0.grid(False)

    xw = np.arange(cfg.window)
    ax1.set_title("(B) Query vs DTW and topology analogues")
    ax1.plot(xw, q, color=WINDOW_COLORS["query"], lw=1.8, label="Q (Query)")
    ax1.plot(xw, dw, color=WINDOW_COLORS["dtw"], lw=1.3, ls="--", label="D (DTW)")
    ax1.plot(xw, tw, color=WINDOW_COLORS["topology"], lw=1.3, ls="-.", label="T (Topology)")
    ax1.set_xlabel("Within-window time")
    ax1.set_ylabel("Return")
    ax1.grid(False)
    ax1.legend(loc="upper right")

    dtw_scaled = dtw_distances / np.quantile(dtw_distances, 0.9)
    topo_scaled = topo_distances / np.quantile(topo_distances, 0.9)
    # skip the burn-in region at the start where distances are unstable / spiking
    _skip = int(np.searchsorted(candidate_starts, candidate_starts[0] + cfg.exclusion_radius))
    cs_plot  = candidate_starts[_skip:]
    dtw_plot  = dtw_scaled[_skip:]
    topo_plot = topo_scaled[_skip:]
    ax2.set_title("(C) Distance profiles over candidate pool")
    ax2.plot(cs_plot, dtw_plot,  color=WINDOW_COLORS["dtw"],      lw=1.0, label="D distance")
    ax2.plot(cs_plot, topo_plot, color=WINDOW_COLORS["topology"],  lw=1.0, label="T distance")
    ax2.axvline(d0, color=WINDOW_COLORS["dtw"],      lw=1.0, ls="--")
    ax2.axvline(t0, color=WINDOW_COLORS["topology"], lw=1.0, ls="--")
    ax2.set_xlim(cs_plot[0], candidate_starts[-1])
    ax2.set_ylim(0, np.quantile(np.concatenate([dtw_plot, topo_plot]), 0.995) * 1.08)
    ax2.set_xlabel("Candidate window start")
    ax2.set_ylabel("Distance / 90th percentile")
    ax2.legend(loc="lower right")

    ax3.set_title("(D) Weighted $H_1$ persistence landscapes")
    ax3.plot(x_land, ql, color=WINDOW_COLORS["query"], lw=1.8, label="Q")
    ax3.plot(x_land, dl, color=WINDOW_COLORS["dtw"], lw=1.3, ls="--", label="D")
    ax3.plot(x_land, tl, color=WINDOW_COLORS["topology"], lw=1.3, ls="-.", label="T")
    active_mask = (ql > 1e-5) | (dl > 1e-5) | (tl > 1e-5)
    if np.any(active_mask):
        active_x = x_land[active_mask]
        x_pad = 0.05 * max(active_x[-1] - active_x[0], 1e-3)
        ax3.set_xlim(max(cfg.landscape_start, active_x[0] - x_pad), min(cfg.landscape_stop, active_x[-1] + x_pad))
    ax3.set_xlabel("Filtration")
    ax3.set_ylabel("Landscape value")
    ax3.grid(False)
    ax3.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax3.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax3.legend(loc="upper right")

    fig.savefig(output_path)
    plt.close(fig)


def save_selection_table(res: dict[str, object], output_dir: Path) -> None:
    candidate_starts = np.asarray(res["candidate_starts"])
    dtw_distances = np.asarray(res["dtw_distances"])
    topo_distances = np.asarray(res["topology_distances"])
    dtw_idx = int(np.flatnonzero(candidate_starts == int(res["dtw_start"]))[0])
    topo_idx = int(np.flatnonzero(candidate_starts == int(res["topology_start"]))[0])

    rows = [
        {
            "selection": "query",
            "window_start": int(res["query_start"]),
            "region": "query_loop",
            "dtw_distance": 0.0,
            "topology_distance": 0.0,
            "dtw_percentile": 0.0,
            "topology_percentile": 0.0,
        },
        {
            "selection": "dtw_nn",
            "window_start": int(res["dtw_start"]),
            "region": str(res["dtw_region"]),
            "dtw_distance": float(dtw_distances[dtw_idx]),
            "topology_distance": float(topo_distances[dtw_idx]),
            "dtw_percentile": float(res["dtw_percentile"]),
            "topology_percentile": float(np.mean(topo_distances <= topo_distances[dtw_idx]) * 100.0),
        },
        {
            "selection": "topology_nn",
            "window_start": int(res["topology_start"]),
            "region": str(res["topology_region"]),
            "dtw_distance": float(dtw_distances[topo_idx]),
            "topology_distance": float(topo_distances[topo_idx]),
            "dtw_percentile": float(np.mean(dtw_distances <= dtw_distances[topo_idx]) * 100.0),
            "topology_percentile": float(res["topology_percentile"]),
        },
    ]
    pd.DataFrame(rows).to_csv(output_dir / "table_mechanism_dtw_topology.csv", index=False)


def main() -> None:
    args = parse_args()
    cfg = DemoConfig()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    series, regions = build_synthetic_returns(cfg, seed=args.seed)
    res = pick_query_and_neighbours(series, cfg, regions)

    fig_path = args.output_dir / "fig2_mechanism_dtw_topology.pdf"
    plot_figure(series, regions, res, cfg, fig_path)
    save_selection_table(res, args.output_dir)

    print(f"Wrote: {fig_path}")
    print(f"Wrote: {args.output_dir / 'table_mechanism_dtw_topology.csv'}")
    print(
        f"query={res['query_start']}, dtw_nn={res['dtw_start']} ({res['dtw_region']}), "
        f"topology_nn={res['topology_start']} ({res['topology_region']})"
    )


if __name__ == "__main__":
    main()