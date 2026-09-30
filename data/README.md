# Data

This directory holds the canonical, frozen input data for the reviewer
revision. The revision uses `panel_published.csv.gz`; it does not refresh
external financial data.

## Files

| File | Description |
|---|---|
| `panel_published.csv.gz` | Frozen aligned daily panel used by every final validation and evaluation run. |
| `panel_metadata.json` | Provenance metadata (date range, tickers, series, column list). |

## Sources

- **Yahoo Finance** (`yfinance`): SPY, IEF, ^VIX, EURUSD=X, CL=F.
  Adjusted closing prices.
- **FRED** (`pandas-datareader`): DGS2, DGS10, T10Y2Y, DFF,
  BAMLH0A0HYM2, BAA10Y.

The aligned panel is built in `tda_risk/data.py::build_research_panel`,
which forward-fills macro series across non-publication days so that
downstream features are defined on every SPY trading day.

The live-download clients and historical refresh utilities are outside the
default reproduction path. Reproduction uses the frozen panel and does not
contact external data providers.

## Date range

Effective analysis period: **2002-07-30** through **2026-05-01** in the
frozen panel. Forecasts use the one-step-ahead target convention described in
`revision_config/revision_protocol.json`.
