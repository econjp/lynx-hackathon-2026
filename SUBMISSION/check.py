"""Leak test + score for the final model. Run before submitting:   python SUBMISSION/check.py [positions.csv]

1. Leak test, both methods from Lynx's instructions (prices and macro together):
   a) shorter history: positions on a cut history must equal the full run on every shared day
   b) changed future: replace everything after day T by random data, positions up to T must not move
2. Key figures from Lynx's evaluation.py (2-day lag, costs, adjusted Sharpe).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "lynx_materials")]

import numpy as np
import polars as pl

import evaluation as ev
from model import DATE, final_positions, load

DATA = HERE.parent / "data"
prices = load([DATA / "close_prices_insample.csv"])
macro = load([DATA / "macro_data_insample.csv"])
full = final_positions(prices, macro)
F = full.drop(DATE).to_numpy()
scale = np.abs(F).max()

worst_cut = 0.0
for c in (0.1, 0.3, 0.5, 0.7, 0.9, 0.97):
    n = int(len(prices) * c)
    p = prices.head(n)
    part = final_positions(p, macro.filter(pl.col(DATE) <= p[DATE][-1])).drop(DATE).to_numpy()
    worst_cut = max(worst_cut, np.abs(F[:n] - part).max() / scale)
print(f"leak test a) shorter history : {'PASS' if worst_cut < 1e-9 else 'FAIL'}  (largest difference {worst_cut:.1e})")

rng = np.random.default_rng(0)
worst_rand = 0.0
for c in (0.3, 0.5, 0.7, 0.9):
    n = int(len(prices) * c)
    px = prices.to_pandas().set_index(DATE)
    px.iloc[n:] = px.iloc[n:] * np.exp(rng.normal(0, 0.05, px.iloc[n:].shape)).cumprod(axis=0)
    mc = macro.to_pandas().set_index(DATE)
    k = int((mc.index <= px.index[n - 1]).sum())
    mc.iloc[k:] = mc.iloc[k:] * np.exp(rng.normal(0, 0.01, mc.iloc[k:].shape)).cumprod(axis=0)
    back = lambda d: pl.from_pandas(d.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))
    alt = final_positions(back(px), back(mc)).drop(DATE).to_numpy()
    worst_rand = max(worst_rand, np.abs(F[:n] - alt[:n]).max() / scale)
print(f"leak test b) changed future  : {'PASS' if worst_rand < 1e-9 else 'FAIL'}  (largest difference {worst_rand:.1e})")

pos = pl.read_csv(sys.argv[1], try_parse_dates=True).with_columns(pl.col(DATE).cast(pl.Date)) if len(sys.argv) > 1 else full
print(f"\nkey figures (Lynx evaluation.py), {pos[DATE][0]} to {pos[DATE][-1]}:")
for k, v in ev.calc_key_figures(pos, prices).items():
    print(f"  {k:16s} {v:6.2f}")
d = prices[DATE]
n = len(d)
for a, b in [(d[0], d[n // 3]), (d[n // 3], d[2 * n // 3]), (d[2 * n // 3], d[-1])]:
    s = ev.calc_key_figures(pos.filter(pl.col(DATE).is_between(a, b)), prices, key_figures=["adjusted_sharpe"])["adjusted_sharpe"]
    print(f"  adjusted_sharpe {a.year}-{b.year}: {s:.2f}")
