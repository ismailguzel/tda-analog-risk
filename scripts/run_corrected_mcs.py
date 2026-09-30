"""Compute radius-specific Hansen--Lunde--Nason model confidence sets."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from arch.bootstrap import MCS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tda_risk.inference_var99 import pinball_loss
from tda_risk.scoring import fz0_score


OUT = ROOT / "results" / "final_inference"
RADIUS_VALUES = (0, 125, 250)
MCS_ALPHA = 0.10
MCS_REPS = 2000
MCS_BLOCK_LENGTH = 10
MCS_METHOD = "R"
MCS_BOOTSTRAP = "circular"
MCS_SEED = 2026092705
PRIMARY_LOSS = "pinball_99"
SECONDARY_LOSS = "fz0_975"


def locked_configuration_ids() -> list[str]:
    registry = pd.read_csv(ROOT / "revision_config" / "finalists.csv")
    labels = registry["configuration_id"].astype(str).tolist()
    if len(labels) != 13 or len(set(labels)) != 13:
        raise ValueError("finalists.csv must contain exactly 13 unique locked configurations")
    return labels


def radius_loss_matrices(
    frame: pd.DataFrame, radius: int, labels: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return separate primary and secondary loss matrices for one radius."""

    labels = labels or locked_configuration_ids()
    frame_radius = frame.loc[
        frame["exclusion_radius"].eq(radius)
        & frame["configuration_id"].astype(str).isin(labels)
    ].copy()
    observed = set(frame_radius["configuration_id"].astype(str).unique())
    if observed != set(labels):
        raise ValueError(
            f"radius {radius} does not contain exactly the locked finalists; "
            f"missing={sorted(set(labels) - observed)}, extra={sorted(observed - set(labels))}"
        )
    if frame_radius.duplicated(["forecast_date", "configuration_id"]).any():
        raise ValueError(f"radius {radius} has duplicate forecast-date/configuration rows")

    frame_radius["primary_loss"] = pinball_loss(
        frame_radius["realized_loss"].to_numpy(float),
        frame_radius["VaR_99"].to_numpy(float),
        0.99,
    )
    frame_radius["secondary_loss"] = fz0_score(
        frame_radius["realized_loss"].to_numpy(float),
        frame_radius["VaR_975"].to_numpy(float),
        frame_radius["ES_975"].to_numpy(float),
        0.975,
    )
    primary = frame_radius.pivot(
        index="forecast_date", columns="configuration_id", values="primary_loss"
    ).reindex(columns=labels).sort_index()
    secondary = frame_radius.pivot(
        index="forecast_date", columns="configuration_id", values="secondary_loss"
    ).reindex(columns=labels).sort_index()
    if primary.isna().any().any() or secondary.isna().any().any():
        raise ValueError(f"radius {radius} has incomplete loss matrix")
    return primary, secondary


def compute_official_mcs(
    losses: pd.DataFrame,
    *,
    radius: int,
    loss_family: str,
    seed: int,
    reps: int = MCS_REPS,
) -> pd.DataFrame:
    """Run the documented arch MCS API and return labelled elimination rows."""

    if losses.shape[1] < 2:
        raise ValueError(f"MCS requires at least two loss columns, got {losses.shape[1]}")
    if losses.isna().any().any():
        raise ValueError("MCS loss matrix contains missing values")
    model_confidence_set = MCS(
        losses,
        size=MCS_ALPHA,
        reps=reps,
        block_size=MCS_BLOCK_LENGTH,
        method=MCS_METHOD,
        bootstrap=MCS_BOOTSTRAP,
        seed=seed,
    )
    model_confidence_set.compute()
    pvalues = model_confidence_set.pvalues.reset_index()
    pvalues = pvalues.rename(columns={"Model name": "configuration_id", "Pvalue": "mcs_pvalue"})
    if "configuration_id" not in pvalues:
        pvalues = pvalues.rename(columns={pvalues.columns[0]: "configuration_id"})
    pvalues["configuration_id"] = pvalues["configuration_id"].astype(str)
    pvalues["elimination_order"] = np.arange(1, len(pvalues) + 1)
    pvalues["included_in_mcs"] = pvalues["configuration_id"].isin(model_confidence_set.included)
    pvalues["mean_loss"] = [float(losses[label].mean()) for label in pvalues["configuration_id"]]
    pvalues["loss_family"] = loss_family
    pvalues["exclusion_radius"] = int(radius)
    pvalues["mcs_alpha"] = MCS_ALPHA
    pvalues["bootstrap_reps"] = int(reps)
    pvalues["block_length"] = MCS_BLOCK_LENGTH
    pvalues["mcs_method"] = MCS_METHOD
    pvalues["bootstrap_method"] = MCS_BOOTSTRAP
    pvalues["mcs_seed"] = int(seed)
    pvalues["n_dates"] = int(losses.shape[0])
    pvalues["n_models"] = int(losses.shape[1])
    return pvalues[
        [
            "exclusion_radius",
            "loss_family",
            "configuration_id",
            "elimination_order",
            "included_in_mcs",
            "mean_loss",
            "mcs_pvalue",
            "mcs_alpha",
            "bootstrap_reps",
            "block_length",
            "mcs_method",
            "bootstrap_method",
            "mcs_seed",
            "n_dates",
            "n_models",
        ]
    ]


def _latex_label(label: str) -> str:
    return label.replace("_", r"\_")


def _write_latex_fragment(result: pd.DataFrame, path: Path, caption: str) -> None:
    lines = [
        "% Generated by scripts/run_corrected_mcs.py; do not edit by hand.",
        r"\begin{table}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\begin{tabular}{l c c c}",
        r"\toprule",
        r"Model & Radius 0 & Radius 125 & Radius 250 \\",
        r"\midrule",
    ]
    labels = locked_configuration_ids()
    for label in labels:
        cells = []
        for radius in RADIUS_VALUES:
            row = result[
                result["configuration_id"].eq(label) & result["exclusion_radius"].eq(radius)
            ]
            if len(row) != 1:
                raise ValueError(f"missing unique LaTeX row for {label}, radius {radius}")
            cells.append("Yes" if bool(row.iloc[0]["included_in_mcs"]) else "No")
        lines.append(f"{_latex_label(label)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_mcs_outputs(
    frame: pd.DataFrame, out: Path = OUT, *, reps: int = MCS_REPS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate only the corrected MCS CSV and LaTeX outputs."""

    out.mkdir(parents=True, exist_ok=True)
    labels = locked_configuration_ids()
    primary_rows: list[pd.DataFrame] = []
    secondary_rows: list[pd.DataFrame] = []
    for radius in RADIUS_VALUES:
        primary, secondary = radius_loss_matrices(frame, radius, labels)
        primary_rows.append(
            compute_official_mcs(
                primary,
                radius=radius,
                loss_family=PRIMARY_LOSS,
                seed=MCS_SEED + radius,
                reps=reps,
            )
        )
        secondary_rows.append(
            compute_official_mcs(
                secondary,
                radius=radius,
                loss_family=SECONDARY_LOSS,
                seed=MCS_SEED + 100 + radius,
                reps=reps,
            )
        )
    primary_result = pd.concat(primary_rows, ignore_index=True)
    secondary_result = pd.concat(secondary_rows, ignore_index=True)
    primary_result.to_csv(out / "mcs_primary_radius_specific.csv", index=False)
    secondary_result.to_csv(out / "mcs_secondary_radius_specific.csv", index=False)
    elimination = pd.concat([primary_result, secondary_result], ignore_index=True)
    elimination.to_csv(out / "mcs_elimination_order_pvalues.csv", index=False)

    summary_rows = []
    for result in (primary_result, secondary_result):
        for radius in RADIUS_VALUES:
            subset = result[result["exclusion_radius"].eq(radius)].sort_values("configuration_id")
            retained = subset.loc[subset["included_in_mcs"], "configuration_id"].tolist()
            summary_rows.append(
                {
                    "exclusion_radius": radius,
                    "loss_family": subset.iloc[0]["loss_family"],
                    "mcs_alpha": subset.iloc[0]["mcs_alpha"],
                    "bootstrap_reps": subset.iloc[0]["bootstrap_reps"],
                    "block_length": subset.iloc[0]["block_length"],
                    "mcs_method": subset.iloc[0]["mcs_method"],
                    "bootstrap_method": subset.iloc[0]["bootstrap_method"],
                    "n_models": subset.iloc[0]["n_models"],
                    "n_dates": subset.iloc[0]["n_dates"],
                    "retained_count": len(retained),
                    "retained_models": ";".join(retained),
                    "topology_included": "topology_l250_k100" in retained,
                }
            )
    pd.DataFrame(summary_rows).to_csv(out / "mcs_membership_summary.csv", index=False)
    _write_latex_fragment(
        primary_result,
        out / "mcs_primary_tables.tex",
        "Primary Hansen--Lunde--Nason MCS membership at $\\alpha=0.10$.",
    )
    _write_latex_fragment(
        secondary_result,
        out / "mcs_secondary_tables.tex",
        "Secondary same-level VaR--ES Hansen--Lunde--Nason MCS membership at $\\alpha=0.10$.",
    )
    return primary_result, secondary_result


def main() -> None:
    frames = [
        pd.read_parquet(ROOT / "results" / "final_evaluation" / f"forecasts_radius{radius}.parquet")
        for radius in RADIUS_VALUES
    ]
    frame = pd.concat(frames, ignore_index=True)
    primary, secondary = generate_mcs_outputs(frame)
    print(
        "wrote corrected radius-specific MCS outputs "
        f"primary={len(primary)} secondary={len(secondary)} "
        f"method={MCS_METHOD} bootstrap={MCS_BOOTSTRAP} reps={MCS_REPS}"
    )


if __name__ == "__main__":
    main()
