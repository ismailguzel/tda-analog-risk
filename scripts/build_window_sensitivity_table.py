"""Build the topology window-length sensitivity table from frozen validation outputs.

For each window length L, select the best k on the blocked validation folds by
mean VaR(0.99) pinball loss (ties broken by worst-fold loss, then by
configuration id) and report its per-fold losses.  Neither L nor k is chosen on
the fixed evaluation panel.  The script only reads
``results/final_validation/validation_fold_summary.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "results" / "final_validation"
WINDOW_LENGTHS = (60, 125, 250)
K_VALUES = (100, 250, 500, 750, 1000)


def build_sensitivity(fold_summary: pd.DataFrame) -> pd.DataFrame:
    k_pattern = "|".join(str(k) for k in K_VALUES)
    rows: list[dict[str, object]] = []
    for window_length in WINDOW_LENGTHS:
        candidates = fold_summary[
            fold_summary.configuration_id.str.match(rf"topology_l{window_length}_k(?:{k_pattern})$")
        ].copy()
        grouped = candidates.groupby("configuration_id", as_index=False).agg(
            mean_loss=("mean_pinball_loss", "mean"),
            worst_fold_loss=("mean_pinball_loss", "max"),
        )
        best_id = grouped.sort_values(
            ["mean_loss", "worst_fold_loss", "configuration_id"]
        ).iloc[0]["configuration_id"]
        best = candidates[candidates.configuration_id.eq(best_id)].sort_values("fold")
        folds = {str(row.fold): float(row.mean_pinball_loss) for _, row in best.iterrows()}
        rows.append(
            {
                "window_length": window_length,
                "best_k": int(str(best_id).rsplit("_k", 1)[1]),
                "mean_pinball_loss": float(best.mean_pinball_loss.mean()),
                "loss_2012": folds["2012"],
                "loss_2013": folds["2013"],
                "loss_2014": folds["2014"],
                "worst_fold_loss": float(best.mean_pinball_loss.max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=VALIDATION / "topology_window_sensitivity.csv")
    args = parser.parse_args()
    fold_summary = pd.read_csv(VALIDATION / "validation_fold_summary.csv")
    build_sensitivity(fold_summary).to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
