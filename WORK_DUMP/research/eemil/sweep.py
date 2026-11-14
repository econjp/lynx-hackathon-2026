"""One-at-a-time parameter sweeps for the eemil model, scored per block.
A parameter is worth setting only if its best value is the same in most blocks.
python research/eemil/sweep.py [prices.csv]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import polars as pl
from validate import load_prices, DATE, daily_pnl, blocks, adj_sharpe, ev
from model import generate_positions, SPEEDS

GRID = {
    "speeds": [(4,), (8,), (16,), (32,), (64,), (4, 8, 16), (8, 16, 32), SPEEDS, (16, 32, 64)],
    "alpha": [1.0, 0.5, 0.3, 0.2, 0.1],
    "vol_span": [30, 60, 100],
    "risk_win": [60, 125, 250, 500],
}

if __name__ == "__main__":
    prices = load_prices(sys.argv[1] if len(sys.argv) > 1 else None)
    prices = prices.with_columns(pl.col(DATE).cast(pl.Date), pl.exclude(DATE).cast(pl.Float64, strict=False))
    pd.set_option("display.width", 200, "display.float_format", "{:.2f}".format)
    for name, values in GRID.items():
        rows = []
        for v in values:
            pos = generate_positions(prices, **{name: v})
            r = daily_pnl(pos, prices)
            kf = ev.calc_key_figures(pos, prices)
            b = blocks(r)
            rows.append({name: str(v), "full": kf["adjusted_sharpe"], **{f"b{i+1}": x for i, x in enumerate(b)},
                         "min": min(b), "hold": kf["av_holding"], "kurt": kf["kurtosis"] - 3})
        print(pd.DataFrame(rows).set_index(name).to_string(), "\n")
