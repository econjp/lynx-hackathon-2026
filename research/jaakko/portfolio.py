"""
research/jaakko/portfolio.py - cut correlated bets on day 1
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import polars as pl

DATE = "AsOfDate"

def apply_ex_ante_covariance_scaling(
    positions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    cov_window: int = 250,
    ridge_shrinkage: float = 1e-6
) -> pd.DataFrame:
    # cut correlated bets on day 1
    p_mat = positions_df.values
    ret_mat = prices_df.pct_change().fillna(0.0).values
    n_days, n_assets = p_mat.shape
    
    scaled = positions_df.copy()
    
    for i in range(n_days):
        if i < 60:
            scaled.iloc[i] = 0.0
            continue
            
        sub_rets = ret_mat[max(0, i - cov_window):i]
        cov = np.cov(sub_rets, rowvar=False) + np.eye(n_assets) * ridge_shrinkage
        pt = p_mat[i]
        port_var = pt @ cov @ pt
        
        if port_var > 1e-12:
            scaled.iloc[i] = pt / np.sqrt(port_var)
        else:
            scaled.iloc[i] = pt
            
    return scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def apply_smooth_execution_glide(
    target_positions: pd.DataFrame,
    alpha: float = 0.30
) -> pd.DataFrame:
    # 30% daily glide towards target
    if alpha >= 1.0:
        return target_positions
    smooth = target_positions.ewm(alpha=alpha, adjust=False).mean()
    return smooth.replace([np.inf, -np.inf], 0.0).fillna(0.0)
