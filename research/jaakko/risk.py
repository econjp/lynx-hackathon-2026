"""
research/jaakko/risk.py - price difference sizing
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import polars as pl

DATE = "AsOfDate"

def compute_price_vol_blend(
    prices_df: pd.DataFrame,
    fast_span: int = 35,
    slow_window: int = 500,
    fast_weight: float = 0.70
) -> pd.DataFrame:
    # fast 35d vol + slow 500d anchor
    diff = prices_df.diff()
    fast_vol = np.sqrt((diff ** 2).ewm(span=fast_span, min_periods=fast_span // 2).mean())
    slow_vol = fast_vol.rolling(slow_window, min_periods=60).mean()
    blended = fast_weight * fast_vol + (1.0 - fast_weight) * slow_vol
    return blended.replace(0.0, np.nan)


def apply_inverse_vol_sizing(
    signals_df: pd.DataFrame,
    vol_df: pd.DataFrame
) -> pd.DataFrame:
    # X_i = Signal_i / vol(diff(P_i))
    positions = signals_df / vol_df
    return positions.replace([np.inf, -np.inf], 0.0).fillna(0.0)
