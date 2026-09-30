from __future__ import annotations

import numpy as np
import pandas as pd


NEAR_ZERO_SCALE = 1e-8


def standardize_window(window: np.ndarray, near_zero_scale: float = NEAR_ZERO_SCALE) -> np.ndarray:
    """Standardize each return-window column using that window only.

    The topology representation uses population standard deviation (``ddof=0``)
    and maps constant or near-constant columns to unit scale.  Comparators use
    this helper so their preprocessing is exactly matched to topology.
    """
    values = np.asarray(window, dtype=float)
    was_1d = values.ndim == 1
    if was_1d:
        values = values[:, None]
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, ddof=0, keepdims=True)
    scale = np.where(scale > near_zero_scale, scale, 1.0)
    standardized = (values - mean) / scale
    return standardized[:, 0] if was_1d else standardized


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
