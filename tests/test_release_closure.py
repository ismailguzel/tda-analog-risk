from __future__ import annotations

import json
from pathlib import Path
import subprocess


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


def test_response_page_line_verifier_passes() -> None:
    result = subprocess.run(
        ["python3", str(ROOT / "scripts" / "verify_response_page_lines.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "references verified" in result.stdout
