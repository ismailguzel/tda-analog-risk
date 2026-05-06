# Data sources

## Equity and bond returns (Yahoo Finance)

- **SPY** — SPDR S\&P 500 ETF Trust (US equity benchmark)
- **IEF** — iShares 7–10 Year Treasury Bond ETF (US fixed income benchmark)
- **^VIX** — CBOE Volatility Index
- **EURUSD=X** — EUR/USD spot rate (held as auxiliary state component)
- **CL=F** — WTI crude futures (held as auxiliary state component)

Adjusted closing prices are used throughout. IEF inception is
2002-07-22, which sets the start of the aligned panel.

## Macro and credit series (FRED)

- **DGS2** — 2-year Treasury constant-maturity yield
- **DGS10** — 10-year Treasury constant-maturity yield
- **T10Y2Y** — 10Y–2Y term spread
- **DFF** — Effective federal funds rate
- **BAMLH0A0HYM2** — ICE BofA US High-Yield Option-Adjusted Spread
- **BAA10Y** — Moody's Baa corporate yield minus 10Y Treasury (HYOAS fallback)

## Alignment rule

All series are aligned to the SPY trading calendar (US equity market
open days). Macro series, which are not always published on every
trading day, are forward-filled across non-publication gaps so that
distances and rolling features are well-defined on every forecast
date.

## Pinning

The reproducibility cache (`data/panel_cached.csv.gz`) is the canonical
input.  Live API behavior may change over time; for any reanalysis
downstream of the cache, the cached panel reproduces the manuscript
exactly.
