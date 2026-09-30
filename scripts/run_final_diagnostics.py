"""Build final locked neighbor, exclusion, and Gap-HS diagnostics."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final_diagnostics"


def _indices(value: object) -> np.ndarray:
    if not isinstance(value, str) or not value:
        return np.array([], dtype=int)
    return np.asarray([int(part) for part in value.split(";") if part], dtype=int)


def neighbor_age(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(pd.read_csv(ROOT / "data" / "panel_published.csv.gz", index_col=0).index)
    positions = {date: pos for pos, date in enumerate(dates)}
    for (radius, config_id), group in frame.groupby(["exclusion_radius", "configuration_id"]):
        ages = []
        for _, row in group.iterrows():
            forecast_pos = positions[pd.Timestamp(row.forecast_date)]
            neighbors = _indices(row.neighbor_indices)
            ages.extend((forecast_pos - neighbors).tolist())
        if ages:
            values = np.asarray(ages, dtype=float)
            rows.append(
                {
                    "exclusion_radius": int(radius), "configuration_id": config_id,
                    "n_neighbors": len(values), "mean_age": float(values.mean()),
                    "median_age": float(np.median(values)), "p05_age": float(np.quantile(values, 0.05)),
                    "p95_age": float(np.quantile(values, 0.95)), "min_age": int(values.min()),
                }
            )
    return pd.DataFrame(rows)


def neighbor_overlap(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    compare = ["topology_l250_k100", "wz_euclidean_l250_k100", "wz_dtw_l250_k250", "wz_fpca_l125_k100"]
    for radius, radius_frame in frame.groupby("exclusion_radius"):
        pivot = {name: group.set_index("forecast_date") for name, group in radius_frame.groupby("configuration_id")}
        if "topology_l250_k100" not in pivot:
            continue
        dates = set(pivot["topology_l250_k100"].index)
        for comparator in compare[1:]:
            if comparator not in pivot:
                continue
            values = []
            for date in sorted(dates & set(pivot[comparator].index)):
                left = set(_indices(pivot["topology_l250_k100"].loc[date, "neighbor_indices"]))
                right = set(_indices(pivot[comparator].loc[date, "neighbor_indices"]))
                union = left | right
                values.append(len(left & right) / len(union) if union else np.nan)
            values = np.asarray(values, dtype=float)
            rows.append(
                {
                    "exclusion_radius": int(radius), "reference": "topology_l250_k100",
                    "comparator": comparator, "n_dates": int(np.isfinite(values).sum()),
                    "mean_jaccard_overlap": float(np.nanmean(values)),
                    "median_jaccard_overlap": float(np.nanmedian(values)),
                }
            )
    return pd.DataFrame(rows)


def exclusion_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    zero = frame[frame.exclusion_radius.eq(0)].set_index(["forecast_date", "configuration_id"])
    wide = frame[frame.exclusion_radius.eq(125)].set_index(["forecast_date", "configuration_id"])
    joined = zero.join(wide, lsuffix="_r0", rsuffix="_r125", how="inner")
    rows = []
    for config_id, group in joined.groupby(level="configuration_id"):
        rows.append(
            {
                "configuration_id": config_id, "n_dates": len(group),
                "mean_pinball_r0": float(group.pinball_99_r0.mean()),
                "mean_pinball_r125": float(group.pinball_99_r125.mean()),
                "difference_r125_minus_r0": float(group.pinball_99_r125.mean() - group.pinball_99_r0.mean()),
                "mean_var99_r0": float(group.VaR_99_r0.mean()), "mean_var99_r125": float(group.VaR_99_r125.mean()),
                "mean_es975_r0": float(group.ES_975_r0.mean()), "mean_es975_r125": float(group.ES_975_r125.mean()),
            }
        )
    return pd.DataFrame(rows)


def gap_hs_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exclusion_radius in (125, 250):
        radius = frame[frame.exclusion_radius.eq(exclusion_radius)]
        ids = [f"gap_hs_w250_r{exclusion_radius}", "topology_l250_k100", "rolling_hs_w500"]
        for config_id in ids:
            group = radius[radius.configuration_id.eq(config_id)]
            if group.empty:
                continue
            rows.append(
                {
                    "exclusion_radius": exclusion_radius, "configuration_id": config_id, "n_dates": len(group),
                    "mean_pinball_99": float(group.pinball_99.mean()),
                    "exceedance_rate_99": float(group.var_exceedance_99.mean()),
                    "mean_var99": float(group.VaR_99.mean()), "mean_es975": float(group.ES_975.mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_parquet(ROOT / "results" / "final_evaluation" / f"forecasts_radius{radius}.parquet") for radius in (0, 125, 250)]
    frame = pd.concat(frames, ignore_index=True)
    neighbor_age(frame).to_csv(OUT / "neighbor_age.csv", index=False)
    neighbor_overlap(frame).to_csv(OUT / "neighbor_overlap.csv", index=False)
    exclusion_comparison(frame).to_csv(OUT / "exclusion_comparison.csv", index=False)
    gap_hs_comparison(frame).to_csv(OUT / "gap_hs_comparison.csv", index=False)
    print("wrote final diagnostics")


if __name__ == "__main__":
    main()
