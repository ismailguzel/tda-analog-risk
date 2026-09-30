from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "revision_config" / "revision_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def main() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    panel_info = protocol["canonical_panel"]
    panel_path = REPO_ROOT / panel_info["path"]

    if not panel_path.exists():
        fail(f"Canonical panel is missing: {panel_path}")
    actual_hash = sha256(panel_path)
    if actual_hash != panel_info["sha256"]:
        fail(
            "Canonical panel hash mismatch. "
            f"Expected {panel_info['sha256']}, found {actual_hash}."
        )

    panel = pd.read_csv(panel_path, index_col=0, parse_dates=True)
    panel.index = pd.to_datetime(panel.index).tz_localize(None)
    if len(panel) != panel_info["expected_rows"]:
        fail(f"Expected {panel_info['expected_rows']} panel rows, found {len(panel)}.")
    if panel.index.min().date().isoformat() != panel_info["expected_first_date"]:
        fail(f"Unexpected first panel date: {panel.index.min().date()}")
    if panel.index.max().date().isoformat() != panel_info["expected_last_date"]:
        fail(f"Unexpected last panel date: {panel.index.max().date()}")

    required_modules = ["numpy", "pandas", "scipy", "sklearn"]
    optional_modules = ["arch", "ripser", "persim", "dtaidistance"]
    missing_required: list[str] = []
    missing_optional: list[str] = []
    for module in required_modules:
        try:
            importlib.import_module(module)
        except Exception:
            missing_required.append(module)
    for module in optional_modules:
        try:
            importlib.import_module(module)
        except Exception:
            missing_optional.append(module)

    if missing_required:
        fail("Missing required Python modules: " + ", ".join(missing_required))

    final_forecasts = REPO_ROOT / "results" / "final_evaluation" / "forecasts_radius0.parquet"
    if not final_forecasts.exists():
        fail("Final evaluation forecast snapshot is missing.")
    header = pd.read_parquet(final_forecasts, columns=None).head(5)
    expected_columns = {
        "forecast_date",
        "realized_date",
        "method",
        "VaR_99",
        "ES_975",
        "realized_loss",
    }
    missing_columns = sorted(expected_columns.difference(header.columns))
    if missing_columns:
        fail("Final forecast snapshot is missing columns: " + ", ".join(missing_columns))

    print("[OK] Revision workspace integrity checks passed.")
    print(f"[OK] Python: {sys.version.split()[0]}")
    print(f"[OK] Canonical panel: {panel_path.relative_to(REPO_ROOT)}")
    print(f"[OK] Panel rows/date range: {len(panel)} / {panel.index.min().date()} to {panel.index.max().date()}")
    print(f"[OK] SHA-256: {actual_hash}")
    if missing_optional:
        print("[WARN] Optional analysis modules not installed yet: " + ", ".join(missing_optional))
        print("       Install requirements.txt before running the full pipeline.")


if __name__ == "__main__":
    main()
