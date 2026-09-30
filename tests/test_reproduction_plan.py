from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_declares_all_final_null_radii() -> None:
    protocol = json.loads(
        (ROOT / "revision_config" / "revision_protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["null_controls"]["radii"] == [0, 125, 250]
    assert protocol["null_controls"]["minimum_draws_final"] == 500
    assert protocol["null_controls"]["types"] == [
        "topology_history_permutation",
        "uniform_random_retrieval",
    ]


def test_full_plan_is_lightweight_and_covers_both_nulls_and_all_radii() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "reproduce_final.sh"), "--plan"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = result.stdout
    assert "topology history-feature permutation" in plan
    assert "uniform-subset retrieval" in plan
    assert "A. locked evaluation null analysis" in plan
    assert "radii 0, 125, and 250; 500 draws" in plan
    assert "B. validation k-sensitivity" in plan
    assert "radii 0 and 125" in plan
    assert "1000}; 250 draws" in plan
    assert "radii 250" not in plan.split("B. validation k-sensitivity", 1)[1]


def test_final_null_script_exposes_separate_frozen_radius_sets() -> None:
    import scripts.run_final_nulls as final_nulls

    assert final_nulls.LOCKED_EVALUATION_NULL_RADII == (0, 125, 250)
    assert final_nulls.VALIDATION_K_SENSITIVITY_RADII == (0, 125)
