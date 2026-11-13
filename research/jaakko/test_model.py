"""Pick speed on train (first 60%), check test once, leak test.
python research/jaakko/test_model.py              -> Lynx example data
python research/jaakko/test_model.py data/x.csv   -> any price csv"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT), str(ROOT / "lynx_materials")]

import numpy as np
import polars as pl
from src.utils import load_prices, DATE
from signals import SPEEDS
from model import generate_positions as model
import evaluation as ev

KEYS = ["adjusted_sharpe", "sharpe_1d_lag", "kurtosis", "skewness", "av_holding", "position_bias"]


def score(pos, prices, start, end):
    pos = pos.filter(pl.col(DATE).is_between(start, end))
    return ev.calc_key_figures(pos, prices, key_figures=KEYS)


def leak_test(prices, cuts=(0.05, 0.5, 0.9), **kw):
    # positions on cut history must equal the full run
    full = model(prices, **kw).drop(DATE).to_numpy()
    for c in cuts:
        n = int(len(prices) * c)
        part = model(prices.slice(0, n), **kw).drop(DATE).to_numpy()
        if np.abs(full[:n] - part).max() > 1e-9:
            return False
    return True


def jumps(prices, k=6):
    # count >6 sigma days per asset (roll gaps show up here)
    px = prices.to_pandas().set_index(DATE).astype(float)
    d = px.diff()
    n = (d.abs() > k * d.rolling(100, min_periods=50).std().shift(1)).sum()
    return n[n > 0].sort_values(ascending=False)


if __name__ == "__main__":
    prices = load_prices(sys.argv[1] if len(sys.argv) > 1 else None)
    prices = prices.with_columns(pl.col(DATE).cast(pl.Date), pl.exclude(DATE).cast(pl.Float64, strict=False))
    d = prices[DATE]
    s0, s1, s2 = d[0], d[int(len(d) * 0.6)], d[-1]
    j = jumps(prices)
    print(f"{prices.width - 1} assets, train {s0}..{s1}, test ..{s2}")
    print(f">6 sigma days: {int(j.sum())}  {dict(j.head(4))}")

    res = {}
    for sp in SPEEDS:
        pos = model(prices, speed=sp)
        res[sp] = (score(pos, prices, s0, s1), score(pos, prices, s1, s2))
        print(f"{sp:5s} train {res[sp][0]['adjusted_sharpe']:.3f}")

    # best on train; if top two are close (<0.1) take 'all'
    rank = sorted(res, key=lambda s: -res[s][0]["adjusted_sharpe"])
    gap = res[rank[0]][0]["adjusted_sharpe"] - res[rank[1]][0]["adjusted_sharpe"]
    pick = rank[0] if gap >= 0.1 else "all"
    print(f"\npick: {pick}  (gap {gap:.2f})")
    for name, r in zip(("train", "test"), res[pick]):
        print(f"  {name}: " + "  ".join(f"{k} {r[k]:.3f}" for k in KEYS))
    print("leak test:", "PASS" if leak_test(prices, speed=pick) else "FAIL")
