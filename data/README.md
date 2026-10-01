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

The panel aligns every series to the SPY trading calendar and forward-fills
macro series across non-publication days, so that downstream features are
defined on every SPY trading day.

This repository does not contain a download path. Reproduction uses the frozen
panel and never contacts external data providers. The data module that
originally built the panel from live sources is kept in the Git history
(commit `1b824a7`).

## Date range

Effective analysis period: **2002-07-30** through **2026-05-01** in the
frozen panel. Forecasts use the one-step-ahead target convention described in
`revision_config/revision_protocol.json`.
