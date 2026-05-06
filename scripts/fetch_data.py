"""Cache-first data fetcher for the tda-analog-risk pipeline.

Default mode (`--use-cache`, the default) loads the aligned daily panel
from `data/panel_cached.csv.gz` and exits.  In refresh mode
(`--refresh`) the script re-pulls the underlying yfinance and FRED
series, rebuilds the aligned panel via `tda_risk.data.build_research_panel`,
and overwrites both the cached panel and the raw API dumps.

Refresh mode requires a working network connection.  Cached mode is
fully offline, which is the canonical path for reproducing the
manuscript.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tda_risk.config import default_pipeline_config
from tda_risk.data import build_research_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Load the aligned panel from the cached file (default).",
    )
    mode.add_argument(
        "--refresh",
        action="store_true",
        help="Re-pull from yfinance and FRED, then rebuild and overwrite the cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = REPO_ROOT / "data" / "panel_cached.csv.gz"

    if args.refresh:
        import pandas as pd
        import yfinance as yf
        from pandas_datareader import data as web

        config = default_pipeline_config()

        # Refresh raw dumps
        yfinance_dir = REPO_ROOT / "data" / "yfinance_raw"
        fred_dir = REPO_ROOT / "data" / "fred_raw"
        yfinance_dir.mkdir(parents=True, exist_ok=True)
        fred_dir.mkdir(parents=True, exist_ok=True)

        for ticker in config.data.yahoo_tickers:
            safe = ticker.replace("^", "").replace("=", "_")
            raw = yf.download(
                tickers=ticker,
                start=config.data.start_date,
                auto_adjust=False,
                progress=False,
            )
            raw.to_csv(yfinance_dir / f"{safe}.csv.gz", compression="gzip")
            print(f"  yfinance {ticker}: {raw.shape}")

        for series in config.data.fred_series:
            try:
                raw = web.DataReader(series, "fred", config.data.start_date)
                raw.to_csv(fred_dir / f"{series}.csv.gz", compression="gzip")
                print(f"  fred     {series}: {raw.shape}")
            except Exception as exc:  # pragma: no cover
                print(f"  fred     {series}: FAILED ({exc})")

        # Rebuild aligned panel and overwrite cache
        result = build_research_panel(config.data)
        panel = result.panel
        panel.to_csv(cache_path, compression="gzip")
        print(f"\nRefreshed panel: {panel.shape} -> {cache_path.relative_to(REPO_ROOT)}")

        meta = {
            "start_date": str(config.data.start_date),
            "end_date": str(panel.index.max().date()),
            "n_dates": int(len(panel)),
            "columns": list(panel.columns),
            "yahoo_tickers": list(config.data.yahoo_tickers),
            "fred_series": list(getattr(config.data, "fred_series", [])),
        }
        with open(REPO_ROOT / "data" / "panel_metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        return

    # Cached mode: just confirm the file exists and report its shape.
    if not cache_path.exists():
        sys.exit(
            f"Cached panel not found at {cache_path}.\n"
            "Run with --refresh to rebuild it from yfinance and FRED."
        )

    import pandas as pd

    panel = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    print(f"Loaded cached panel: {panel.shape} from {cache_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
