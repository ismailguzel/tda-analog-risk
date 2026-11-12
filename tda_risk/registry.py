"""Loader for the validation candidate registry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CANDIDATE_REGISTRY_COLUMNS = {
    "configuration_id",
    "method",
    "status",
}


def load_candidate_registry(path: Path) -> pd.DataFrame:
    """Load the complete, explicit validation candidate universe."""
    frame = pd.read_csv(path)
    missing = CANDIDATE_REGISTRY_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"candidate registry is missing columns: {sorted(missing)}")
    if frame["configuration_id"].isna().any() or frame["configuration_id"].duplicated().any():
        raise ValueError("candidate registry configuration_id values must be unique and non-null")
    if frame["method"].isna().any():
        raise ValueError("candidate registry method values must be non-null")
    return frame
