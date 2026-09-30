"""Generate a concept figure for the persistence-landscape pipeline.

The figure shows one stylized return window, its delay-embedded point cloud,
the corresponding H1 persistence diagram, and a persistence landscape panel
with the first three layers.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter, ScalarFormatter
import numpy as np

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
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
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
# Color-blind safe palette with a purple middle layer.
LAYER_COLORS = ("#0072B2", "#7B2CBF", "#009E73")


@dataclass(frozen=True)
class DemoConfig:
    window: int = 125
    tau: int = 2
    embed_dim: int = 3
    landscape_steps: int = 220
    landscape_start: float = 0.0
    landscape_stop: float = 0.085
    weights: tuple[float, ...] = (1.0, 0.5, 0.25)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "figures",
    )
    return parser.parse_args()


def build_stylized_window(cfg: DemoConfig) -> np.ndarray:
    t = np.linspace(0.0, 1.0, cfg.window)
    envelope = 1.0 + 0.26 * np.sin(2.0 * np.pi * t - 0.22) + 0.08 * np.sin(4.0 * np.pi * t + 0.3)
    carrier1 = 0.0145 * np.sin(2.0 * np.pi * 2.2 * t + 0.30 * np.sin(2.0 * np.pi * t))
    carrier2 = 0.0130 * np.sin(2.0 * np.pi * 4.4 * t + 0.55 * np.sin(2.0 * np.pi * t + 0.4))
    carrier3 = 0.0060 * np.sin(2.0 * np.pi * 6.6 * t + 0.95)
    base = envelope * (carrier1 + carrier2) + carrier3
    base += 0.0024 * np.exp(-0.5 * ((t - 0.23) / 0.028) ** 2)
    base -= 0.0020 * np.exp(-0.5 * ((t - 0.70) / 0.022) ** 2)
    return base


def delay_embed(window: np.ndarray, tau: int, embed_dim: int) -> np.ndarray:
    usable = len(window) - tau * (embed_dim - 1)
    if usable < 5:
        raise ValueError("Window too short for embedding parameters.")
    cols = [window[i * tau : i * tau + usable] for i in range(embed_dim)]
    return np.column_stack(cols)


def persistence_diagram(cloud: np.ndarray) -> np.ndarray:
    dgms = ripser(cloud, maxdim=1)["dgms"]
    d1 = dgms[1]
    finite = d1[np.isfinite(d1[:, 1])]
    if finite.size == 0:
        return np.empty((0, 2), dtype=float)
    return finite


def landscape_layers(diagram: np.ndarray, cfg: DemoConfig) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(cfg.landscape_start, cfg.landscape_stop, cfg.landscape_steps)
    if diagram.size == 0:
        zeros = np.zeros((3, cfg.landscape_steps), dtype=float)
        return x, zeros

    landscape = PersLandscapeApprox(
        start=cfg.landscape_start,
        stop=cfg.landscape_stop,
        num_steps=cfg.landscape_steps,
        dgms=[np.empty((0, 2), dtype=float), diagram],
        hom_deg=1,
    )
    values = np.asarray(landscape.values, dtype=float)
    layers = values[: min(3, values.shape[0])] if values.size else np.zeros((0, cfg.landscape_steps), dtype=float)
    if layers.shape[0] < 3:
        pad = np.zeros((3 - layers.shape[0], cfg.landscape_steps), dtype=float)
        layers = np.vstack([layers, pad])
    return x, layers


def plot_pipeline(
    window: np.ndarray,
    cloud: np.ndarray,
    diagram: np.ndarray,
    x: np.ndarray,
    layers: np.ndarray,
    cfg: DemoConfig,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.6))
    fig.subplots_adjust(hspace=0.36, wspace=0.30, left=0.09, right=0.97, top=0.94, bottom=0.10)
    ax0, ax1, ax2, ax3 = axes.flat

    # ── Panel A: Return window — points colored by time (same viridis as B) ──
    # Scale to % for readable tick labels
    t_win = np.arange(window.size)
    t_color = np.linspace(0.0, 1.0, window.size)
    win_pct = window * 100.0
    ax0.set_title("(A) Return window $w_t$", pad=5)
    ax0.plot(t_win, win_pct, color="0.78", lw=0.8, zorder=1)
    ax0.scatter(t_win, win_pct, c=t_color, cmap="viridis",
                s=10, edgecolors="none", alpha=0.85, zorder=2)
    ax0.axhline(0, color="0.70", lw=0.5, ls="--")
    ax0.set_xlabel("Within-window day $s$")
    ax0.set_ylabel("Return (%)")
    ax0.xaxis.set_major_locator(MaxNLocator(3))
    ax0.yaxis.set_major_locator(MaxNLocator(4))
    ax0.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax0.margins(y=0.04)
    ax0.grid(False)

    # ── Panel B: 2-D phase-plane delay embedding (x_t, x_{t-τ}) ────────────
    # Same ×100 scaling → directly comparable colours to Panel A
    t_color_cloud = np.linspace(0.0, 1.0, cloud.shape[0])
    cloud_pct = cloud * 100.0
    ax1.set_title(r"(B) Phase-plane embedding $(x_t,\,x_{t-\tau})$", pad=5)
    ax1.plot(cloud_pct[:, 0], cloud_pct[:, 1], color="0.78", lw=0.7, alpha=0.40, zorder=1)
    ax1.scatter(
        cloud_pct[:, 0], cloud_pct[:, 1],
        c=t_color_cloud, cmap="viridis",
        s=18, edgecolors="none", alpha=0.90, zorder=2,
    )
    ax1.set_xlabel(r"$x_t$ (%)")
    ax1.set_ylabel(r"$x_{t-\tau}$ (%)")
    ax1.xaxis.set_major_locator(MaxNLocator(3))
    ax1.yaxis.set_major_locator(MaxNLocator(3))
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax1.margins(y=0.04)
    ax1.grid(False)

    # ── Panel C: H1 persistence diagram ─────────────────────────────────────
    ax2.set_title("(C) $H_1$ persistence diagram", pad=5)
    # Panel C data scaled ×10³ so ticks read like 2.5, 4.1 … instead of 0.0025
    _s2 = 1e3  # display scale factor for C
    if diagram.size:
        births_s  = diagram[:, 0] * _s2
        deaths_s  = diagram[:, 1] * _s2
        pad_x = 0.18 * max(float(np.ptp(births_s)), 1e-2)
        pad_y = 0.18 * max(float(np.ptp(deaths_s)), 1e-2)
        lo = max(0.0, float(births_s.min()) - pad_x)
        hi = max(float(deaths_s.max()) + pad_y, float(births_s.max()) + pad_x)
        ax2.fill_between([lo, hi], [lo, hi], [hi, hi],
                         color="#f0f0f0", linewidth=0, zorder=0)
        ax2.plot([lo, hi], [lo, hi], color="0.68", lw=0.9, dashes=(3, 2), zorder=1)
        ax2.scatter(births_s, deaths_s,
                    color="#E69F00", s=48, alpha=0.92,
                    edgecolors="white", linewidths=0.5, zorder=3)
        i_max = int(np.argmax(deaths_s - births_s))
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(lo, hi)
    else:
        ax2.set_xlim(0.0, 30.0)
        ax2.set_ylim(0.0, 30.0)
        ax2.plot([0.0, 30.0], [0.0, 30.0], color="0.68", lw=0.9, dashes=(3, 2))
    ax2.set_xlabel(r"Birth $\;(\times10^{-3})$")
    ax2.set_ylabel(r"Death $\;(\times10^{-3})$")
    ax2.xaxis.set_major_locator(MaxNLocator(3, min_n_ticks=3))
    ax2.yaxis.set_major_locator(MaxNLocator(3, min_n_ticks=3))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax2.grid(False)

    # ── Panel D: Persistence landscape layers ────────────────────────────────
    ax3.set_title("(D) Persistence landscape $\\lambda_k$", pad=5)
    active_mask = np.any(layers > 1e-5, axis=0)
    for idx, color in enumerate(LAYER_COLORS, start=1):
        lw = 2.2 - 0.35 * (idx - 1)
        ax3.plot(x, layers[idx - 1], color=color, lw=lw, alpha=0.95, label=rf"$\lambda_{idx}$")
    if np.any(active_mask):
        active_x = x[active_mask]
        xpad = 0.05 * max(active_x[-1] - active_x[0], 1e-4)
        ax3.set_xlim(max(cfg.landscape_start, active_x[0] - xpad),
                     min(cfg.landscape_stop,   active_x[-1] + xpad))
        ymax = float(np.max(layers[:, active_mask]))
        ax3.set_ylim(0.0, 1.15 * ymax if ymax > 0 else 0.02)
    # Panel D: scale x (filtration) ×10³, y (landscape) ×10⁴
    _sx = 1e3
    _sy = 1e4
    ax3.set_xlabel(r"Filtration $\varepsilon\;(\times10^{-3})$")
    ax3.set_ylabel(r"Landscape $\;(\times10^{-4})$")
    ax3.xaxis.set_major_locator(MaxNLocator(3))
    ax3.yaxis.set_major_locator(MaxNLocator(4))
    ax3.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*_sx:.1f}"))
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*_sy:.1f}"))
    ax3.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="0.80")
    ax3.grid(False)

    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = DemoConfig()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    window = build_stylized_window(cfg)
    cloud = delay_embed(window, tau=cfg.tau, embed_dim=cfg.embed_dim)
    diagram = persistence_diagram(cloud)
    x, layers = landscape_layers(diagram, cfg)

    output_path = args.output_dir / "fig1_pipeline_concept.pdf"
    plot_pipeline(window, cloud, diagram, x, layers, cfg, output_path)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
