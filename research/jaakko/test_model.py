"""
research/jaakko/test_model.py - Model evaluator & truncation leak test
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
import polars as pl
from src.utils import load_prices, evaluate_model, DATE

try:
    from research.jaakko.model import generate_positions
except ImportError:
    from model import generate_positions

def run_tests():
    prices = load_prices()
    positions = generate_positions(prices)
    
    # 1. key figures
    res = evaluate_model(positions, prices)
    print("=== Jaakko's Model Evaluation ===")
    for k in ["adjusted_sharpe", "sharpe", "sharpe_1d_lag", "kurtosis", "av_holding", "position_bias"]:
        val = res.get(k)
        print(f"  {k:20s}: {val:+.4f}" if isinstance(val, float) else f"  {k:20s}: {val}")
        
    # 2. truncation leak check
    n_bars = len(prices)
    cut_indices = [int(n_bars * 0.1), int(n_bars * 0.5), n_bars - 5]
    max_diff = 0.0
    full_pos = positions.to_pandas().set_index(DATE)
    
    print("\n=== Truncation Leak Check ===")
    for cut in cut_indices:
        prices_cut = prices.slice(0, cut + 1)
        pos_cut = generate_positions(prices_cut).to_pandas().set_index(DATE)
        diff = np.abs(full_pos.iloc[:cut+1].values - pos_cut.values).max()
        max_diff = max(max_diff, diff)
        print(f"  cut {cut:5d} / {n_bars}: diff = {diff:.2e}")
        
    print(f"\nResult: {'PASS (0 lookahead)' if max_diff < 1e-10 else 'FAIL (lookahead detected)'}")

if __name__ == "__main__":
    run_tests()
