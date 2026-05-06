# Data

This directory holds the canonical input data for the pipeline.

## Files

| File | Description |
|---|---|
| `panel_cached.csv.gz` | Aligned daily panel (yfinance + FRED) used by every pipeline run. Loaded by default through `scripts/fetch_data.py --use-cache`. |
| `panel_metadata.json` | Provenance metadata (date range, tickers, series, column list). |
| `yfinance_raw/*.csv.gz` | Raw API responses from `yfinance` for each ticker, gzipped. |
| `fred_raw/*.csv.gz` | Raw API responses from FRED via `pandas-datareader`, gzipped. |

## Sources

- **Yahoo Finance** (`yfinance`): SPY, IEF, ^VIX, EURUSD=X, CL=F.
  Adjusted closing prices.
- **FRED** (`pandas-datareader`): DGS2, DGS10, T10Y2Y, DFF,
  BAMLH0A0HYM2, BAA10Y.

The aligned panel is built in `tda_risk/data.py::build_research_panel`,
which forward-fills macro series across non-publication days so that
downstream features are defined on every SPY trading day.

## Refreshing the cache

```bash
python scripts/fetch_data.py --refresh
```

This re-pulls every series, regenerates `panel_cached.csv.gz`, and
overwrites the raw dumps.  The cached version distributed with this
repository is what the manuscript reports.

## Date range

Effective analysis period: **2002-07-22** (IEF inception, the latest
starting series among the required features) through approximately
**2026** (depending on when the cache was last refreshed).
