from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import DataConfig


BENCHMARK_COLUMNS = ("SPY_adj_close", "IEF_adj_close")
PRIMARY_HY_SPREAD_SERIES = "BAMLH0A0HYM2"
FALLBACK_CREDIT_SPREAD_SERIES = "BAA10Y"


@dataclass(frozen=True)
class PanelBuildResult:
    panel: pd.DataFrame
    metadata: dict[str, str]


def fetch_yahoo_panel(config: DataConfig) -> pd.DataFrame:
    import yfinance as yf

    tickers = list(config.yahoo_tickers)
    price_data: dict[str, pd.Series] = {}
    for ticker in tickers:
        raw = yf.download(
            tickers=ticker,
            start=config.start_date,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw.empty:
            raise RuntimeError(f"Yahoo Finance download returned an empty dataset for {ticker}.")
        if ticker in ("SPY", "IEF"):
            series = raw["Adj Close"].squeeze("columns").rename(f"{ticker}_adj_close")
        else:
            series = raw["Close"].squeeze("columns").rename(f"{ticker}_close")
        price_data[series.name] = series

    yahoo_panel = pd.DataFrame(price_data).sort_index()
    yahoo_panel.index = pd.to_datetime(yahoo_panel.index).tz_localize(None)
    return yahoo_panel


def fetch_fred_panel(config: DataConfig) -> pd.DataFrame:
    from pandas_datareader import data as web

    frames: list[pd.Series] = []
    for series_name in config.fred_series:
        series = web.DataReader(series_name, "fred", config.start_date).squeeze("columns")
        series.index = pd.to_datetime(series.index).tz_localize(None)
        frames.append(series.rename(series_name))
    return pd.concat(frames, axis=1).sort_index()


def build_research_panel(config: DataConfig) -> PanelBuildResult:
    yahoo_panel = fetch_yahoo_panel(config)
    fred_panel = fetch_fred_panel(config)

    benchmark_panel = yahoo_panel.loc[:, list(BENCHMARK_COLUMNS)].dropna()
    if benchmark_panel.empty:
        raise RuntimeError("Benchmark series are empty after download.")

    spy_calendar = benchmark_panel.index
    aligned = benchmark_panel.copy()

    extra_yahoo = yahoo_panel.drop(columns=list(BENCHMARK_COLUMNS), errors="ignore")
    if not extra_yahoo.empty:
        aligned = aligned.join(extra_yahoo.reindex(spy_calendar).ffill())

    aligned = aligned.join(fred_panel.reindex(spy_calendar).ffill())
    aligned = aligned.dropna(subset=list(BENCHMARK_COLUMNS))

    primary_first_valid = aligned[PRIMARY_HY_SPREAD_SERIES].dropna().index.min()
    fallback_first_valid = aligned[FALLBACK_CREDIT_SPREAD_SERIES].dropna().index.min()
    aligned["credit_spread_proxy"] = aligned[PRIMARY_HY_SPREAD_SERIES].combine_first(
        aligned[FALLBACK_CREDIT_SPREAD_SERIES]
    )

    aligned["SPY_return"] = aligned["SPY_adj_close"].pct_change()
    aligned["IEF_return"] = aligned["IEF_adj_close"].pct_change()
    aligned["portfolio_return"] = (
        config.spy_weight * aligned["SPY_return"] + config.ief_weight * aligned["IEF_return"]
    )
    aligned["portfolio_loss"] = -aligned["portfolio_return"]

    metadata = {
        "calendar": "SPY trading days",
        "fred_fill_rule": "forward-fill to the next SPY trading day",
        "yahoo_fill_rule": "forward-fill non-benchmark series to the SPY trading calendar",
        "benchmark": f"{config.spy_weight:.1f}*SPY + {config.ief_weight:.1f}*IEF",
        "credit_spread_feature": (
            f"{PRIMARY_HY_SPREAD_SERIES} with {FALLBACK_CREDIT_SPREAD_SERIES} fallback "
            "when the ICE series is unavailable in FRED"
        ),
        "credit_spread_primary_first_valid": (
            primary_first_valid.date().isoformat() if not pd.isna(primary_first_valid) else "missing"
        ),
        "credit_spread_fallback_first_valid": (
            fallback_first_valid.date().isoformat() if not pd.isna(fallback_first_valid) else "missing"
        ),
    }
    return PanelBuildResult(panel=aligned, metadata=metadata)
