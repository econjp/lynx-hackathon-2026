"""
src/baseline.py - Team NewNordics Joint Baseline Model
======================================================
Multi-horizon trend model (40d / 80d) with continuous tanh activation,
price volatility scaling, and portfolio model risk normalization.

Benchmark performance on example_prices.csv:
- Adjusted Sharpe: ~0.8600 (vs Lynx baseline 0.6621)
- 1-day lagged Sharpe: ~0.7917
- Causality max diff: 0.00e+00
"""
from __future__ import annotations
import sys
from pathlib import Path

# Add project root and lynx_materials to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import polars as pl
from src.utils import load_prices, evaluate_model, check_causality, DATE

def champion_model(
    ret: pl.DataFrame,
    w_fast: int = 40,
    w_med: int = 80,
    vol_win: int = 60,
    risk_win: int = 100
) -> pl.DataFrame:
    """
    Multi-horizon trend strategy:
    - Fast and medium momentum normalized by asset price volatility.
    - Saturated with tanh activation to prevent turnover whipsaws and excess kurtosis.
    - Portfolio model risk normalization to target steady variance over time.
    """
    cols = [c for c in ret.columns if c != DATE]
    vol = ret.select(DATE, pl.col(cols).pow(2).rolling_sum(vol_win, min_samples=vol_win // 2).sqrt())
    s_fast = ret.select(DATE, pl.col(cols).rolling_sum(w_fast, min_samples=w_fast // 2))
    s_med = ret.select(DATE, pl.col(cols).rolling_sum(w_med, min_samples=w_med // 2))
    
    exprs = []
    for c in cols:
        v = vol[c]
        sig = ((0.40 * s_fast[c] + 0.60 * s_med[c]) / v).tanh()
        exprs.append((sig / v).alias(c))
        
    positions = ret.select(DATE).with_columns(exprs)
    
    # Portfolio model risk normalization (with mandatory 2-day execution lag)
    gross = positions.select(DATE, *[(pl.col(c).shift(2) * ret[c]).alias(c) for c in cols])
    risk = gross.filter(~pl.all_horizontal(pl.col(cols).is_null())).select(
        DATE, pl.sum_horizontal(pl.col(cols)).rolling_std(risk_win, min_samples=20).alias("model_risk")
    )
    aligned = positions.join(risk, on=DATE, how="left", maintain_order="left")
    return aligned.select(DATE, *[(pl.col(c) / pl.col("model_risk")).fill_nan(None).alias(c) for c in cols])

def main():
    prices = load_prices()
    instruments = [c for c in prices.columns if c != DATE]
    print(f"Loaded {len(prices)} bars across {len(instruments)} instruments: {instruments}")
    
    print("\nRunning truncation causality test (zero lookahead check) ...")
    passed, max_diff = check_causality(champion_model, prices)
    print(f"Causality: {'OK (Zero Lookahead)' if passed else 'FAIL'} (max diff: {max_diff:.2e})")
    assert passed, "Failed causality test!"
    
    ret = prices.select(DATE, pl.col(instruments).diff())
    positions = champion_model(ret)
    
    print("\n--- Lynx Official Key Figures ---")
    res = evaluate_model(positions, prices)
    print(f"Adjusted Sharpe  : {res.get('adjusted_sharpe'):.4f}  (Lynx baseline was 0.6621)")
    print(f"Nominal Sharpe   : {res.get('sharpe'):.4f}")
    print(f"Sharpe (+1d lag) : {res.get('sharpe_1d_lag'):.4f}")
    print(f"Average Holding  : {res.get('av_holding'):.1f} days")
    print(f"Position Bias    : {res.get('position_bias'):.4f}")
    print(f"Excess Kurtosis  : {res.get('kurtosis') - 3.0:.4f}")
    
    out_csv = ROOT_DIR / "results.csv"
    positions.write_csv(out_csv)
    print(f"\nSaved positions to {out_csv.name}")

if __name__ == "__main__":
    main()
