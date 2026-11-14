"""
src/utils.py - Shared quantitative research helpers for Team NewNordics
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
LYNX_DIR = ROOT_DIR / "lynx_materials"

if str(LYNX_DIR) not in sys.path:
    sys.path.insert(0, str(LYNX_DIR))

import numpy as np
import polars as pl
import evaluation

DATE = "AsOfDate"

def load_prices(csv_path: str | Path | None = None) -> pl.DataFrame:
    """Loads and standardizes input prices CSV."""
    path = Path(csv_path) if csv_path else (LYNX_DIR / "example_prices.csv")
    df = pl.read_csv(path, try_parse_dates=True)
    if "dates" in df.columns:
        df = df.rename({"dates": DATE})
    elif "date" in df.columns:
        df = df.rename({"date": DATE})
    return df

def evaluate_model(positions: pl.DataFrame, prices: pl.DataFrame | None = None) -> dict[str, object]:
    """Evaluates strategy positions using Lynx's official scoring function."""
    if prices is None:
        prices = load_prices()
    return evaluation.calc_key_figures(positions, prices)

def plot_model(positions: pl.DataFrame, prices: pl.DataFrame | None = None):
    """Generates Lynx's official 4-panel diagnostic performance chart."""
    if prices is None:
        prices = load_prices()
    return evaluation.plot_key_figures(positions, prices)

def check_causality(strategy_fn, prices: pl.DataFrame | None = None, n_cuts: int = 15) -> tuple[bool, float]:
    """
    Verifies that the strategy has zero lookahead bias via truncation invariance.
    Returns (passed, max_diff).
    """
    if prices is None:
        prices = load_prices()
        
    instruments = [c for c in prices.columns if c != DATE]
    ret_full = prices.select(DATE, pl.col(instruments).diff())
    pos_full = strategy_fn(ret_full)
    
    n_bars = len(prices)
    cut_indices = np.linspace(int(n_bars * 0.5), n_bars - 2, n_cuts, dtype=int)
    max_diff = 0.0
    
    for cut in cut_indices:
        prices_cut = prices.slice(0, cut + 1)
        ret_cut = prices_cut.select(DATE, pl.col(instruments).diff())
        pos_cut = strategy_fn(ret_cut)
        
        row_full = pos_full.filter(pl.col(DATE) == prices_cut[DATE][-1]).select(instruments).to_numpy()
        row_cut = pos_cut.filter(pl.col(DATE) == prices_cut[DATE][-1]).select(instruments).to_numpy()
        diff = float(np.nanmax(np.abs(row_full - row_cut))) if len(row_full) > 0 and len(row_cut) > 0 else 0.0
        max_diff = max(max_diff, diff)
        
    return (max_diff < 1e-9), max_diff

def split_walk_forward(prices: pl.DataFrame | None = None, train_ratio: float = 0.7) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Splits dataset chronologically into In-Sample (train) and Out-of-Sample (validate)."""
    if prices is None:
        prices = load_prices()
    n_train = int(len(prices) * train_ratio)
    return prices.slice(0, n_train), prices.slice(n_train, len(prices) - n_train)
