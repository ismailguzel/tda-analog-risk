"""Version-controlled finalist registry for submitted-specification reproduction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def load_finalist_registry(path: Path) -> pd.DataFrame:
    """Load and validate one explicit submitted finalist per method."""
    frame = pd.read_csv(path)
    required = {"method", "label", "status"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Finalist registry is missing columns: {sorted(missing)}")
    if frame["method"].isna().any() or frame["method"].duplicated().any():
        duplicates = frame.loc[frame["method"].duplicated(keep=False), "method"].tolist()
        raise ValueError(f"Finalist registry must contain exactly one row per method: {duplicates}")
    if frame["label"].duplicated().any():
        raise ValueError("Finalist registry labels must be unique.")
    return frame


def registry_row(registry: pd.DataFrame, method: str) -> pd.Series:
    matches = registry[registry["method"].astype(str) == method]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one finalist for {method!r}, found {len(matches)}.")
    return matches.iloc[0]


def _same_value(actual: Any, expected: Any) -> bool:
    if pd.isna(expected):
        return True
    if pd.isna(actual):
        return False
    try:
        return bool(float(actual) == float(expected))
    except (TypeError, ValueError):
        return str(actual) == str(expected)


def resolve_finalist(
    candidate_summary: pd.DataFrame,
    registry: pd.DataFrame,
    method: str,
    parameter_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Resolve a registry row against candidates without heuristic tie-breaking."""
    row = registry_row(registry, method)
    candidates = candidate_summary[candidate_summary["method"].astype(str) == method].copy()
    for column in parameter_columns:
        if column in candidates.columns and column in row.index and not pd.isna(row[column]):
            candidates = candidates[candidates[column].map(lambda value: _same_value(value, row[column]))]
    if len(candidates) != 1:
        raise ValueError(
            f"Registry finalist {method!r} must resolve to exactly one candidate; found {len(candidates)}."
        )
    resolved = candidates.copy()
    resolved.insert(0, "registry_label", row["label"])
    resolved.insert(1, "registry_status", row["status"])
    return resolved
