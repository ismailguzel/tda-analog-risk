from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_compound_return(series: pd.Series, window: int) -> pd.Series:
    return (1.0 + series).rolling(window=window).apply(np.prod, raw=True) - 1.0


def build_state_panel(panel: pd.DataFrame) -> pd.DataFrame:
    state = panel.copy()
    state["rv20_p"] = state["portfolio_return"].rolling(window=20).std(ddof=0) * np.sqrt(252.0)
    state["ret20_spy"] = _rolling_compound_return(state["SPY_return"], window=20)
    state["VIX"] = state["^VIX_close"]
    state["Delta5_DGS10"] = state["DGS10"].diff(5)
    state["T10Y2Y"] = state["T10Y2Y"]
    # Keep the preregistered feature name while allowing a long-history fallback proxy.
    state["HYOAS"] = state["credit_spread_proxy"]
    return state


def expanding_zscore(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    min_history: int = 60,
) -> pd.DataFrame:
    history = frame.loc[:, columns].expanding(min_periods=min_history)
    mean = history.mean().shift(1)
    std = history.std(ddof=0).shift(1).replace(0.0, np.nan)
    zscores = (frame.loc[:, columns] - mean) / std
    return zscores.add_prefix("z_")


def attach_state_zscores(
    frame: pd.DataFrame,
    state_columns: list[str] | tuple[str, ...],
    min_history: int = 60,
) -> pd.DataFrame:
    zscores = expanding_zscore(frame, state_columns, min_history=min_history)
    return frame.join(zscores)
