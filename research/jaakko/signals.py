"""
research/jaakko/signals.py - Multi-horizon EWMAC & Donchian breakout signals
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import polars as pl

DATE = "AsOfDate"

def compute_ewmac_signals(
    prices_df: pd.DataFrame,
    speeds: list[tuple[int, int]] = [(16, 64), (32, 128), (64, 256)],
    vol_series: pd.DataFrame | None = None
) -> pd.DataFrame:
    if vol_series is None:
        vol_series = prices_df.diff().rolling(60, min_periods=20).std().replace(0.0, np.nan)
        
    combined = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    
    for fast_span, slow_span in speeds:
        ema_fast = prices_df.ewm(span=fast_span, min_periods=fast_span // 2).mean()
        ema_slow = prices_df.ewm(span=slow_span, min_periods=slow_span // 2).mean()
        raw = (ema_fast - ema_slow) / vol_series
        
        scale = raw.abs().expanding(min_periods=100).mean()
        forecast = (raw / scale.replace(0.0, np.nan)).clip(-2.0, 2.0).fillna(0.0)
        combined += forecast / len(speeds)
        
    return combined


def compute_breakout_signals(
    prices_df: pd.DataFrame,
    windows: list[int] = [80, 160, 320]
) -> pd.DataFrame:
    combined = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    
    for win in windows:
        roll_max = prices_df.rolling(win, min_periods=win // 2).max()
        roll_min = prices_df.rolling(win, min_periods=win // 2).min()
        channel = roll_max - roll_min
        pos = 2.0 * ((prices_df - roll_min) / channel.replace(0.0, np.nan)) - 1.0
        combined += pos.clip(-2.0, 2.0).fillna(0.0) / len(windows)
        
    return combined


def build_signal_sleeve(
    prices: pl.DataFrame | pd.DataFrame,
    use_breakout: bool = True,
    w_breakout: float = 0.35
) -> pd.DataFrame:
    if isinstance(prices, pl.DataFrame):
        df_px = prices.to_pandas().set_index(DATE)
    else:
        df_px = prices.copy()
        if DATE in df_px.columns:
            df_px = df_px.set_index(DATE)
            
    ewmac = compute_ewmac_signals(df_px)
    
    if use_breakout:
        breakout = compute_breakout_signals(df_px)
        return ((1.0 - w_breakout) * ewmac + w_breakout * breakout).fillna(0.0)
        
    return ewmac.fillna(0.0)
