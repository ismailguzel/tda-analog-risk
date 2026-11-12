from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_seed_registry_contains_the_null_control_roots() -> None:
    registry = json.loads(
        (ROOT / "analysis_config" / "random_seed_registry.json").read_text(encoding="utf-8")
    )
    assert set(registry) == {"root_seed", "null_controls"}
    assert set(registry["null_controls"]) == {
        "topology_history_permutation",
        "uniform_random_retrieval",
    }

