"""
JP base research model
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LYNX_DIR = ROOT_DIR / "lynx_materials"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(LYNX_DIR) not in sys.path:
    sys.path.insert(0, str(LYNX_DIR))

import numpy as np
import pandas as pd
import polars as pl

try:
    from research.jaakko.signals import build_signal_sleeve
    from research.jaakko.risk import compute_price_vol_blend, apply_inverse_vol_sizing
    from research.jaakko.portfolio import apply_ex_ante_covariance_scaling, apply_smooth_execution_glide
except ImportError:
    from signals import build_signal_sleeve
    from risk import compute_price_vol_blend, apply_inverse_vol_sizing
    from portfolio import apply_ex_ante_covariance_scaling, apply_smooth_execution_glide

DATE = "AsOfDate"

def generate_positions(
    prices: pl.DataFrame | pd.DataFrame,
    use_breakout: bool = True,
    w_breakout: float = 0.35,
    smooth_alpha: float = 0.30
) -> pl.DataFrame:
    if isinstance(prices, pl.DataFrame):
        df_px = prices.to_pandas()
    else:
        df_px = prices.copy()
        
    date_col = DATE if DATE in df_px.columns else df_px.columns[0]
    df_px = df_px.set_index(date_col)
    
    # 1. signals: ewmac + breakout
    signals = build_signal_sleeve(df_px, use_breakout=use_breakout, w_breakout=w_breakout)
    
    # 2. sizing: price difference sizing
    vol_blend = compute_price_vol_blend(df_px)
    raw_positions = apply_inverse_vol_sizing(signals, vol_blend)
    
    # 3. portfolio: cut correlated bets on day 1
    scaled_positions = apply_ex_ante_covariance_scaling(raw_positions, df_px)
    
    # 4. execution: 30% daily glide
    final_positions = apply_smooth_execution_glide(scaled_positions, alpha=smooth_alpha)
    
    out_df = final_positions.copy()
    out_df.index.name = DATE
    return pl.from_pandas(out_df.reset_index())
