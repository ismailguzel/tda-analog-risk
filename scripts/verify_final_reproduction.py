"""Verify the shipped frozen inputs, registries, outputs, and figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


LOCKED_EVALUATION_NULL_RADII = {0, 125, 250}
VALIDATION_K_SENSITIVITY_RADII = {0, 125}
VALIDATION_K_VALUES = {100, 250, 500, 750, 1000}
NULL_CONTROLS = {"topology_history_permutation", "uniform_random_retrieval"}


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def require(path: Path) -> Path:
    if not path.exists():
        fail(f"Missing required path: {path.relative_to(ROOT)}")
    return path



RESULT_MANIFEST = ROOT / "provenance" / "result_hash_manifest.sha256"


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file with the pinned pyarrow engine."""
    return pd.read_parquet(path, engine="pyarrow")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help=(
            "Check schemas and designs only.  Use after a full rerun: recomputed "
            "outputs carry fresh timing fields and are not byte-identical to the "
            "recorded hashes."
        ),
    )
    args = parser.parse_args()

    protocol_path = require(ROOT / "revision_config" / "revision_protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    panel_info = protocol["canonical_panel"]
    panel_path = require(ROOT / panel_info["path"])
    actual_hash = digest(panel_path)
    if actual_hash != panel_info["sha256"]:
        fail(f"Frozen panel hash mismatch: {actual_hash}")
    panel = pd.read_csv(panel_path, index_col=0, parse_dates=True)
    dates = pd.to_datetime(panel.index).tz_localize(None)
    if len(panel) != panel_info["expected_rows"]:
        fail(f"Frozen panel row count is {len(panel)}, expected {panel_info['expected_rows']}")
    if dates.min().date().isoformat() != panel_info["expected_first_date"]:
        fail("Frozen panel first date does not match the protocol")
    if dates.max().date().isoformat() != panel_info["expected_last_date"]:
        fail("Frozen panel last date does not match the protocol")

    registry_paths = [
        ROOT / "revision_config" / "finalists.csv",
        ROOT / "revision_config" / "recovery_candidate_registry.csv",
        ROOT / "revision_config" / "diagnostic_candidate_registry.csv",
    ]
    for path in registry_paths:
        frame = pd.read_csv(require(path))
        if frame.empty:
            fail(f"Registry is empty: {path.relative_to(ROOT)}")
        if "configuration_id" not in frame.columns:
            fail(f"Registry lacks configuration_id: {path.relative_to(ROOT)}")
    if len(pd.read_csv(ROOT / "revision_config" / "finalists.csv")) != 13:
        fail("Locked finalist registry does not contain 13 families")
    json.loads(require(ROOT / "revision_config" / "random_seed_registry.json").read_text())

    evaluation_required = {
        "forecast_date", "realized_date", "realized_loss", "VaR_99", "VaR_975",
        "ES_975", "configuration_id", "exclusion_radius",
    }
    for radius in (0, 125, 250):
        frame = read_parquet(require(ROOT / "results" / "final_evaluation" / f"forecasts_radius{radius}.parquet"))
        missing = evaluation_required.difference(frame.columns)
        if missing:
            fail(f"Radius {radius} forecasts lack columns: {sorted(missing)}")
        if set(frame["exclusion_radius"].dropna().astype(int)) != {radius}:
            fail(f"Radius {radius} forecast file contains another radius")

    null_root = ROOT / "results" / "final_nulls"
    null_config = protocol["null_controls"]
    expected_radii = set(int(radius) for radius in null_config["radii"])
    if expected_radii != LOCKED_EVALUATION_NULL_RADII:
        fail(f"Protocol locked-null radii are not {sorted(LOCKED_EVALUATION_NULL_RADII)}")
    declared_draws = int(null_config["minimum_draws_final"])
    metadata = json.loads(require(null_root / "null_run_metadata.json").read_text(encoding="utf-8"))
    if set(int(radius) for radius in metadata["radii"]) != expected_radii:
        fail("Null metadata does not declare every final radius")
    if int(metadata["draws_per_control_radius"]) != declared_draws:
        fail("Null metadata draw count does not match the protocol")
    if metadata["locked_configuration"] != "topology_l250_k100":
        fail("Null metadata does not identify the locked topology configuration")
    frame = read_parquet(require(null_root / "main_null_draws.parquet"))
    if set(frame["control"].astype(str)) != NULL_CONTROLS:
        fail("Canonical null output controls do not match the locked implementations")
    if set(frame["split"].astype(str)) != {"evaluation"}:
        fail("Canonical locked null output is not evaluation-only")
    if set(frame["window_length"].astype(int)) != {250} or set(frame["k"].astype(int)) != {100}:
        fail("Canonical locked null metadata does not match topology L=250, k=100")
    if set(frame["history_only"].astype(bool)) != {True}:
        fail("Canonical locked null output is not history-only")
    observed_radii = set(frame["exclusion_radius"].astype(int))
    if observed_radii != expected_radii:
        fail(f"Canonical null output is missing radii: {sorted(expected_radii - observed_radii)}")
    groups = frame.groupby(["exclusion_radius", "control"], sort=False).size()
    required_groups = {(radius, control) for radius in expected_radii for control in NULL_CONTROLS}
    if set(groups.index) != required_groups:
        fail(f"Locked null groups are inconsistent: observed={sorted(groups.index)}")
    if set(groups.astype(int)) != {declared_draws}:
        fail(f"Declared null draw count is not {declared_draws}: {groups.to_dict()}")
    if set(metadata["controls"]) != set(frame["control"].unique()):
        fail("Null metadata controls do not match the canonical output")

    sensitivity = read_parquet(require(null_root / "k_sensitivity_draws.parquet"))
    if set(sensitivity["control"].astype(str)) != NULL_CONTROLS:
        fail("Validation k-sensitivity controls do not match the locked implementations")
    if set(sensitivity["split"].astype(str)) != {"validation"}:
        fail("Validation k-sensitivity output is not validation-only")
    observed_sensitivity_radii = set(sensitivity["exclusion_radius"].astype(int))
    if observed_sensitivity_radii != VALIDATION_K_SENSITIVITY_RADII:
        fail(
            "Validation k-sensitivity radii are inconsistent: "
            f"observed={sorted(observed_sensitivity_radii)}, "
            f"expected={sorted(VALIDATION_K_SENSITIVITY_RADII)}"
        )
    observed_k = set(sensitivity["k"].astype(int))
    if observed_k != VALIDATION_K_VALUES:
        fail(f"Validation k-sensitivity k values are inconsistent: {sorted(observed_k)}")
    if set(sensitivity["window_length"].astype(int)) != {250}:
        fail("Validation k-sensitivity window length does not match the frozen design")
    sensitivity_groups = sensitivity.groupby(["control", "exclusion_radius", "k"], sort=False).size()
    required_sensitivity_groups = {
        (control, radius, k)
        for control in NULL_CONTROLS
        for radius in VALIDATION_K_SENSITIVITY_RADII
        for k in VALIDATION_K_VALUES
    }
    if set(sensitivity_groups.index) != required_sensitivity_groups:
        fail("Validation k-sensitivity does not contain exactly the 20 required groups")
    if set(sensitivity_groups.astype(int)) != {250}:
        fail(f"Validation k-sensitivity draw counts are inconsistent: {sensitivity_groups.to_dict()}")
    if 250 in observed_sensitivity_radii:
        fail("Unexpected radius-250 validation k-sensitivity output is present")

    inference_root = ROOT / "results" / "final_inference"
    for name in ("mcs_primary_radius_specific.csv", "mcs_secondary_radius_specific.csv"):
        frame = pd.read_csv(require(inference_root / name))
        required = {"exclusion_radius", "configuration_id", "included_in_mcs", "mcs_alpha", "bootstrap_reps", "block_length", "mcs_method", "bootstrap_method"}
        missing = required.difference(frame.columns)
        if missing:
            fail(f"MCS file lacks columns: {sorted(missing)}")
        if set(frame["exclusion_radius"].astype(int)) != {0, 125, 250}:
            fail(f"MCS file is not radius-specific: {name}")
        if set(frame["bootstrap_reps"].astype(int)) != {2000} or set(frame["block_length"].astype(int)) != {10}:
            fail(f"MCS bootstrap declaration is inconsistent: {name}")
        if set(frame["mcs_method"].astype(str)) != {"R"}:
            fail(f"MCS method is not the Hansen--Lunde--Nason R statistic: {name}")
    for name in ("mcs_membership_summary.csv", "mcs_elimination_order_pvalues.csv"):
        require(inference_root / name)

    window_sensitivity = pd.read_csv(require(ROOT / "results" / "final_validation" / "topology_window_sensitivity.csv"))
    if set(window_sensitivity["window_length"].astype(int)) != {60, 125, 250}:
        fail("Topology window-sensitivity output does not cover L=60, 125, and 250")
    expected_best_k = {60: 1000, 125: 1000, 250: 100}
    observed_best_k = dict(zip(window_sensitivity.window_length.astype(int), window_sensitivity.best_k.astype(int)))
    if observed_best_k != expected_best_k:
        fail(f"Topology window-sensitivity winners are inconsistent: {observed_best_k}")

    tie_path = ROOT / "results" / "final_diagnostics" / "cutoff_tie_diagnostic.csv"
    ties = pd.read_csv(require(tie_path))
    tie_columns = {
        "sample_type", "exclusion_radius", "window_length", "k", "number_queries",
        "exact_tie_count", "near_tie_count", "exact_tie_rate", "near_tie_rate", "min_cutoff_gap",
    }
    missing_tie_columns = tie_columns.difference(ties.columns)
    if missing_tie_columns:
        fail(f"Cutoff-tie diagnostic lacks columns: {sorted(missing_tie_columns)}")
    if ties.empty or (ties.number_queries <= 0).any():
        fail("Cutoff-tie diagnostic has no valid query catalog")
    if (ties.exact_tie_count < 0).any() or (ties.near_tie_count < 0).any():
        fail("Cutoff-tie diagnostic contains negative tie counts")

    posthoc_root = ROOT / "results" / "posthoc_diagnostics"
    recency_draws = pd.read_csv(require(posthoc_root / "recency_matched_draws.csv"))
    recency_summary = pd.read_csv(require(posthoc_root / "recency_matched_summary.csv"))
    recency_profile = pd.read_csv(require(posthoc_root / "topology_neighbor_recency.csv"))
    shuffle_detail = pd.read_csv(require(posthoc_root / "temporal_shuffle_window_detail.csv"))
    shuffle_summary = pd.read_csv(require(posthoc_root / "temporal_shuffle_summary.csv"))
    if len(recency_draws) != 500 or set(recency_summary["draws"].astype(int)) != {500}:
        fail("Recency-matched benchmark does not contain the declared 500 draws")
    required_recency = {
        "mean_pinball_99", "exceedance_rate_99", "mean_var_99", "seed"
    }
    if required_recency.difference(recency_draws.columns):
        fail("Recency-matched draw file lacks required columns")
    if recency_profile.empty or set(recency_profile.columns) != {"measure", "value"}:
        fail("Topology-neighbor recency profile is missing or malformed")
    if len(shuffle_detail) != 100:
        fail(f"Temporal-shuffle diagnostic has {len(shuffle_detail)} windows, expected 100")
    shuffle_row = shuffle_summary.iloc[0]
    if int(shuffle_row["sampled_windows"]) != 100 or int(shuffle_row["shuffles_per_window"]) != 25:
        fail("Temporal-shuffle summary does not match the declared 100-window, 25-shuffle design")
    if int(shuffle_row["subsampled_delay_points"]) != 30:
        fail("Temporal-shuffle diagnostic does not use the declared 30-point cloud subsample")

    for path in (
        ROOT / "figures" / "fig1_pipeline_concept.pdf",
        ROOT / "figures" / "fig2_mechanism_dtw_topology.pdf",
        ROOT / "figures" / "table_mechanism_dtw_topology.csv",
    ):
        require(path)

    # Every frozen input and shipped result must match its recorded hash.
    entries = [line.split(maxsplit=1) for line in require(RESULT_MANIFEST).read_text().splitlines() if line.strip()]
    if not args.skip_hash_check:
        for expected, name in entries:
            name = name.strip().lstrip("*")
            actual = digest(require(ROOT / name))
            if actual != expected:
                fail(f"Hash mismatch for {name}: {actual} (manifest {expected})")

    print("[OK] Frozen panel verified")
    print(f"[OK] Panel SHA-256: {actual_hash}")
    print("[OK] Active CSV/JSON registries parse")
    print("[OK] Final forecast schemas and radius isolation verified")
    print(f"[OK] Final null draw counts verified at {declared_draws} per control/radius for {sorted(expected_radii)}")
    print("[OK] Validation k-sensitivity verified at radii [0, 125], five k values, 20 groups, 250 draws each")
    print("[OK] Standard radius-specific MCS metadata verified")
    print("[OK] Topology window-length sensitivity winners verified")
    print("[OK] Post hoc recency-matched and temporal-order diagnostics verified")
    print(
        "[OK] Complete-IID cutoff-tie diagnostic verified: "
        f"exact={int(ties.exact_tie_count.sum())}, near={int(ties.near_tie_count.sum())}"
    )
    if args.skip_hash_check:
        print("[SKIP] Hash check skipped (--skip-hash-check)")
    else:
        print(f"[OK] {len(entries)} frozen inputs and results match {RESULT_MANIFEST.relative_to(ROOT)}")
    print("[OK] Explanatory figures present")


if __name__ == "__main__":
    main()
