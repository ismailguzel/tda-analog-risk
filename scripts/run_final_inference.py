"""Run the locked Phase B primary and secondary inference suite."""

from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tda_risk.inference_var99 import circular_block_ci, dm_pinball, holm_adjust, pinball_loss
from tda_risk.scoring import (
    acerbi_szekely_z2,
    christoffersen_independence,
    conditional_coverage,
    duration_geometric_test,
    exceedance_durations,
    fz0_score,
    kupiec_uc,
)
from scripts.run_corrected_mcs import generate_mcs_outputs


OUT = ROOT / "results" / "final_inference"
LEVEL = 0.99
ES_LEVEL = 0.975
BOOTSTRAPS = 2000
BLOCK_LENGTH = 10
SEED = 2026092705
RADIUS_VALUES = (0, 125, 250)


def _circular_resample(values: np.ndarray, block_length: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    block = max(1, min(int(block_length), n))
    pieces = []
    total = 0
    while total < n:
        start = int(rng.integers(0, n))
        indices = (start + np.arange(block)) % n
        pieces.append(values[indices])
        total += block
    return np.concatenate(pieces)[:n]


def _block_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.array(
        [float(_circular_resample(x, BLOCK_LENGTH, rng).mean()) for _ in range(BOOTSTRAPS)]
    )
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _es_block_bootstrap(y: np.ndarray, q: np.ndarray, e: np.ndarray, seed: int) -> dict[str, object]:
    valid = np.isfinite(y) & np.isfinite(q) & np.isfinite(e) & (e > 0.0)
    y, q, e = y[valid], q[valid], e[valid]
    if y.size < 2:
        return {"n_valid": int(y.size), "es_statistic": math.nan, "es_pvalue": math.nan}
    contributions = y * (y > q) / ((1.0 - ES_LEVEL) * e) - 1.0
    centered = contributions - contributions.mean()
    observed_sd = float(np.std(contributions, ddof=1))
    observed = 0.0 if observed_sd <= 0 else float(np.sqrt(len(contributions)) * contributions.mean() / observed_sd)
    rng = np.random.default_rng(seed)
    bootstrap_stats = np.empty(BOOTSTRAPS)
    for idx in range(BOOTSTRAPS):
        sample = _circular_resample(centered, BLOCK_LENGTH, rng)
        sd = float(np.std(sample, ddof=1))
        bootstrap_stats[idx] = 0.0 if sd <= 0 else float(np.sqrt(len(sample)) * sample.mean() / sd)
    pvalue = float((1.0 + np.sum(np.abs(bootstrap_stats) >= abs(observed))) / (BOOTSTRAPS + 1.0))
    return {
        "n_valid": int(y.size),
        "es_statistic": observed,
        "es_pvalue": pvalue,
        "bootstrap_reps": BOOTSTRAPS,
        "block_length": BLOCK_LENGTH,
        "mean_z2_contribution": float(contributions.mean()),
    }


def _backtests(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    var_rows = []
    secondary_rows = []
    es_rows = []
    for (radius, config_id, method), group in frame.groupby(
        ["exclusion_radius", "configuration_id", "method"], sort=True
    ):
        group = group.sort_values("forecast_date")
        y = group.realized_loss.to_numpy(float)
        q99 = group.VaR_99.to_numpy(float)
        q975 = group.VaR_975.to_numpy(float)
        e975 = group.ES_975.to_numpy(float)
        exceed = np.isfinite(y) & np.isfinite(q99) & (y > q99)
        uc_stat, uc_p = kupiec_uc(exceed, LEVEL)
        ind_stat, ind_p = christoffersen_independence(exceed)
        cc_stat, cc_p = conditional_coverage(exceed, LEVEL)
        durations = exceedance_durations(exceed)
        duration_stat, duration_p = duration_geometric_test(exceed, LEVEL)
        pin = pinball_loss(y, q99, LEVEL)
        pin_mean, pin_lo, pin_hi = _block_mean_ci(pin, SEED + int(radius) + len(var_rows))
        var_rows.append(
            {
                "exclusion_radius": int(radius), "configuration_id": config_id, "method": method,
                "n_forecasts": len(group), "mean_pinball_99": pin_mean, "pinball_ci_low": pin_lo,
                "pinball_ci_high": pin_hi, "exceedances": int(exceed.sum()),
                "exceedance_rate": float(exceed.mean()), "kupiec_uc_stat": uc_stat,
                "kupiec_uc_pvalue": uc_p, "christoffersen_ind_stat": ind_stat,
                "christoffersen_ind_pvalue": ind_p, "christoffersen_cc_stat": cc_stat,
                "christoffersen_cc_pvalue": cc_p, "duration_count": int(len(durations)),
                "duration_mean": float(durations.mean()) if len(durations) else math.nan,
                "duration_geometric_stat": duration_stat, "duration_geometric_pvalue": duration_p,
            }
        )
        fz = fz0_score(y, q975, e975, ES_LEVEL)
        fz_mean, fz_lo, fz_hi = _block_mean_ci(fz, SEED + 1000 + int(radius) + len(secondary_rows))
        secondary_rows.append(
            {
                "exclusion_radius": int(radius), "configuration_id": config_id, "method": method,
                "n_forecasts": len(group), "n_valid_fz": int(np.isfinite(fz).sum()),
                "mean_fz0_975": fz_mean, "fz_ci_low": fz_lo, "fz_ci_high": fz_hi,
                "mean_var_975": float(np.nanmean(q975)), "mean_es_975": float(np.nanmean(e975)),
            }
        )
        es = _es_block_bootstrap(y, q975, e975, SEED + 2000 + int(radius) + len(es_rows))
        es.update({"exclusion_radius": int(radius), "configuration_id": config_id, "method": method})
        es_rows.append(es)
    return pd.DataFrame(var_rows), pd.DataFrame(secondary_rows), pd.DataFrame(es_rows)


def _paired(frame: pd.DataFrame, secondary: bool) -> pd.DataFrame:
    rows = []
    for radius, radius_frame in frame.groupby("exclusion_radius"):
        pivot_cols = ["realized_loss", "VaR_975", "ES_975"] if secondary else ["realized_loss", "VaR_99"]
        wide = radius_frame.pivot(index="forecast_date", columns="configuration_id", values=pivot_cols)
        configs = sorted(radius_frame.configuration_id.unique())
        topology = "topology_l250_k100"
        topo = radius_frame[radius_frame.configuration_id.eq(topology)].set_index("forecast_date")
        topo_loss = fz0_score(topo.realized_loss, topo.VaR_975, topo.ES_975, ES_LEVEL) if secondary else pinball_loss(topo.realized_loss, topo.VaR_99, LEVEL)
        pvalues = []
        for config_id in configs:
            if config_id == topology:
                continue
            comp = radius_frame[radius_frame.configuration_id.eq(config_id)].set_index("forecast_date")
            comp_loss = fz0_score(comp.realized_loss, comp.VaR_975, comp.ES_975, ES_LEVEL) if secondary else pinball_loss(comp.realized_loss, comp.VaR_99, LEVEL)
            joined = pd.concat([pd.Series(topo_loss, index=topo.index, name="topology"), pd.Series(comp_loss, index=comp.index, name="comparator")], axis=1).dropna()
            difference = joined.topology.to_numpy() - joined.comparator.to_numpy()
            stat, pvalue = dm_pinball(difference, bandwidth=BLOCK_LENGTH)
            lo, hi = circular_block_ci(difference, BLOCK_LENGTH, BOOTSTRAPS, SEED + len(rows))
            rows.append(
                {
                    "exclusion_radius": int(radius), "topology": topology, "comparator": config_id,
                    "loss_family": "fz0_975" if secondary else "pinball_99", "n_pairs": len(difference),
                    "mean_difference_topology_minus_comparator": float(difference.mean()),
                    "dm_statistic": stat, "dm_pvalue": pvalue, "block_ci_low": lo,
                    "block_ci_high": hi, "bootstrap_reps": BOOTSTRAPS, "block_length": BLOCK_LENGTH,
                }
            )
            pvalues.append(pvalue)
        if pvalues:
            adjusted = holm_adjust(np.asarray(pvalues))
            for offset, value in enumerate(adjusted):
                rows[-len(pvalues) + offset]["holm_pvalue"] = value
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_parquet(ROOT / "results" / "final_evaluation" / f"forecasts_radius{radius}.parquet") for radius in RADIUS_VALUES]
    frame = pd.concat(frames, ignore_index=True)
    var, secondary, es = _backtests(frame)
    var.to_csv(OUT / "var99_backtests.csv", index=False)
    secondary.to_csv(OUT / "var_es975_backtests.csv", index=False)
    es.to_csv(OUT / "es_block_bootstrap.csv", index=False)
    primary_paired = _paired(frame, secondary=False)
    secondary_paired = _paired(frame, secondary=True)
    primary_paired.to_csv(OUT / "paired_primary.csv", index=False)
    secondary_paired.to_csv(OUT / "paired_secondary.csv", index=False)
    corrected_primary, corrected_secondary = generate_mcs_outputs(frame, OUT)
    duration = var[["exclusion_radius", "configuration_id", "method", "duration_count", "duration_mean", "duration_geometric_stat", "duration_geometric_pvalue"]].copy()
    duration["interpretation"] = "exploratory diagnostic; not a primary acceptance criterion"
    duration.to_csv(OUT / "duration_diagnostic.csv", index=False)
    print(
        f"wrote final inference rows: var={len(var)} "
        f"paired_primary={len(primary_paired)} "
        f"mcs_primary={len(corrected_primary)} "
        f"mcs_secondary={len(corrected_secondary)}"
    )


if __name__ == "__main__":
    main()
