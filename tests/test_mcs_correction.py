from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from scripts.run_corrected_mcs import (
    MCS_METHOD,
    MCS_BOOTSTRAP,
    compute_official_mcs,
    locked_configuration_ids,
    radius_loss_matrices,
)


ROOT = Path(__file__).resolve().parents[1]
RADIUS_EXPECTATIONS = {
    0: (0.00029130326754479087, -3.8080589232787823),
    125: (0.00032520768953628044, -3.6758011888932054),
    250: (0.0003261005559705759, -3.6662490012951343),
}


def evaluation_frame() -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_parquet(
                ROOT / "results" / "final_evaluation" / f"forecasts_radius{radius}.parquet"
            )
            for radius in (0, 125, 250)
        ],
        ignore_index=True,
    )


def test_radius_specific_source_means_are_not_mixed() -> None:
    frame = evaluation_frame()
    primary_output = pd.read_csv(ROOT / "results" / "final_inference" / "mcs_primary_radius_specific.csv")
    secondary_output = pd.read_csv(ROOT / "results" / "final_inference" / "mcs_secondary_radius_specific.csv")
    primary_means = []
    secondary_means = []
    for radius, (expected_primary, expected_secondary) in RADIUS_EXPECTATIONS.items():
        primary, secondary = radius_loss_matrices(frame, radius)
        primary_mean = float(primary["topology_l250_k100"].mean())
        secondary_mean = float(secondary["topology_l250_k100"].mean())
        primary_means.append(primary_mean)
        secondary_means.append(secondary_mean)
        assert primary_mean == pytest.approx(expected_primary, abs=1e-12)
        assert secondary_mean == pytest.approx(expected_secondary, abs=1e-12)
        primary_row = primary_output[
            primary_output["configuration_id"].eq("topology_l250_k100")
            & primary_output["exclusion_radius"].eq(radius)
        ]
        secondary_row = secondary_output[
            secondary_output["configuration_id"].eq("topology_l250_k100")
            & secondary_output["exclusion_radius"].eq(radius)
        ]
        assert len(primary_row) == len(secondary_row) == 1
        assert primary_row.iloc[0]["mean_loss"] == pytest.approx(expected_primary, abs=1e-12)
        assert secondary_row.iloc[0]["mean_loss"] == pytest.approx(expected_secondary, abs=1e-12)

    assert len({round(value, 12) for value in primary_means}) == 3
    assert len({round(value, 9) for value in secondary_means}) == 3
    mixed_primary = float(np.mean(primary_means))
    mixed_secondary = float(np.mean(secondary_means))
    assert all(abs(value - mixed_primary) > 1e-8 for value in primary_means)
    assert all(abs(value - mixed_secondary) > 1e-4 for value in secondary_means)


def test_corrected_outputs_have_one_radius_and_thirteen_locked_labels() -> None:
    labels = set(locked_configuration_ids())
    for name in ("mcs_primary_radius_specific.csv", "mcs_secondary_radius_specific.csv"):
        result = pd.read_csv(ROOT / "results" / "final_inference" / name)
        assert set(result["exclusion_radius"]) == {0, 125, 250}
        assert set(result["configuration_id"]) == labels
        assert result.groupby("exclusion_radius").size().to_dict() == {0: 13, 125: 13, 250: 13}
        assert set(result["n_models"]) == {13}
        assert set(result["mcs_method"]) == {MCS_METHOD}
        assert set(result["bootstrap_method"]) == {MCS_BOOTSTRAP}
        assert "gap_hs_w250_r125" not in set(result["configuration_id"])
        assert "gap_hs_w250_r250" not in set(result["configuration_id"])


def test_membership_summary_retains_correct_label_columns() -> None:
    summary = pd.read_csv(ROOT / "results" / "final_inference" / "mcs_membership_summary.csv")
    assert set(summary["loss_family"]) == {"pinball_99", "fz0_975"}
    assert summary.groupby(["loss_family", "exclusion_radius"]).size().eq(1).all()
    assert summary["n_models"].eq(13).all()
    assert summary["retained_models"].map(lambda value: "gap_hs" not in value).all()


def test_official_mcs_is_deterministic_for_same_seed() -> None:
    losses = pd.DataFrame(
        np.arange(96, dtype=float).reshape(24, 4) / 100.0,
        columns=["model_a", "model_b", "model_c", "model_d"],
    )
    first = compute_official_mcs(
        losses, radius=0, loss_family="pinball_99", seed=987654, reps=40
    )
    second = compute_official_mcs(
        losses, radius=0, loss_family="pinball_99", seed=987654, reps=40
    )
    assert_frame_equal(first, second)
