from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_seed_registry_contains_only_active_final_roots() -> None:
    registry = json.loads(
        (ROOT / "revision_config" / "random_seed_registry.json").read_text(encoding="utf-8")
    )
    assert set(registry) == {"root_seed", "recovery"}
    assert set(registry["recovery"]) == {
        "topology_history_permutation",
        "uniform_random_retrieval",
    }

