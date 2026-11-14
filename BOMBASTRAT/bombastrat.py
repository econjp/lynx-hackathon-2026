"""BOMBASTRAT: three forecasts, one portfolio. The whole model is in this file.

    from BOMBASTRAT.bombastrat import positions
    pos = positions(prices, macro)          # polars in, polars out (AsOfDate + one column per asset)

    python BOMBASTRAT/bombastrat.py --prices in.csv [oos.csv] --macro in.csv [oos.csv] --out positions.csv

1. TREND      all 27 assets. Slow price trend (Jaakko's pick: EWMAC 32/128 + 64/256 and 12-month momentum).
2. INFLATION  5 bonds + 5 stocks. CPI of the asset's moon rising faster than its 3-year normal -> short.
3. TWINS      pairs that move together (correlation > 0.8). When the gap between them is wide vs its
              1-year average -> bet on it closing.

Each forecast is scaled to the same risk and they are added. Then: position = forecast / volatility,
the whole book is scaled to constant risk, and positions move 30% of the way to target each day.

A position for day T uses data up to and including day T only. Every average is trailing.
All numbers below were fixed before scoring. None was fitted.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "jaakko"))

import numpy as np
import pandas as pd
import polars as pl

from signals import SPEEDS, clean_prices, ewmac, scale_and_cap   # trend signal (Jaakko)
from risk import price_vol                                       # volatility of daily price changes
from portfolio import exante_scale, glide                        # constant book risk, slow execution

DATE = "AsOfDate"
CPI_DAYS = {"bonds": 60, "stocks": 120}   # inflation lookback per asset class
NORMAL = 750                              # "normal" = trailing 3 years
TWIN_CORR = 0.8                           # twins = pairs above this correlation
SPREAD_DAYS = 250                         # twin gap is compared with its 1-year average
RISK_DAYS = 250                           # window for each forecast's own risk
GLIDE = 0.3                               # move 30% toward target per day


def trend(px, vol):
    clean = clean_prices(px, vol)                                         # one-day jumps clipped at 5 vol
    mom = scale_and_cap(clean.diff().rolling(250).sum() / (vol * np.sqrt(250)))
    return ((ewmac(clean, vol, SPEEDS["slow"]) + mom) / 2).fillna(0.0)


def inflation(px, macro):
    f = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    cpi = macro[[c for c in macro.columns if c.startswith("cpi|")]]
    log_cpi = np.log(cpi.reindex(cpi.index.union(px.index)).ffill().reindex(px.index))   # last value at or before T
    for cls, days in CPI_DAYS.items():
        growth = log_cpi.diff(days)
        surprise = growth - growth.rolling(NORMAL, min_periods=250).mean()               # vs its own 3-year normal
        z = surprise / surprise.rolling(NORMAL, min_periods=250).std().replace(0, np.nan)
        for col in z.columns:
            asset = f"{col.split('|')[1]}_{cls}"
            if asset in f:
                f[asset] = -np.tanh(z[col])                                              # hot inflation -> short
    return f.fillna(0.0)


def twins(px):
    f = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    weekly, logp, cols = px.diff(5), np.log(px.where(px > 0)), list(px.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            is_twin = weekly[a].expanding(min_periods=250).corr(weekly[b]) > TWIN_CORR   # all data up to T
            if not is_twin.any():
                continue
            gap = logp[a] - logp[b]
            r = gap.rolling(SPREAD_DAYS, min_periods=120)
            t = np.tanh((gap - r.mean()) / r.std().replace(0, np.nan) / 2).where(is_twin, 0.0).fillna(0.0)
            f[a] -= t                                                                    # gap wide -> short a, long b
            f[b] += t
    return f


def own_risk(f, vol, dp):
    """Trailing volatility of the forecast's own P&L (with the 2-day lag), while it holds positions."""
    units = (f / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    live = (units.abs().sum(axis=1) > 0).shift(2, fill_value=False)
    pnl = (units.shift(2) * dp).sum(axis=1).where(live)
    return pnl.rolling(RISK_DAYS, min_periods=60).std().replace(0, np.nan)


def _frame(x):
    df = x.to_pandas() if isinstance(x, pl.DataFrame) else x.copy()
    df[DATE] = pd.to_datetime(df[DATE])
    return df.set_index(DATE).sort_index().astype(float)


def forecasts(prices, macro):
    px, mac = _frame(prices), _frame(macro)
    vol = price_vol(px)
    return {"trend": trend(px, vol), "inflation": inflation(px, mac), "twins": twins(px)}, px, vol


def positions(prices, macro, use=("trend", "inflation", "twins")):
    f, px, vol = forecasts(prices, macro)
    dp = px.diff()
    total = sum(f[s].div(own_risk(f[s], vol, dp), axis=0).fillna(0.0) for s in use)      # equal risk each, added
    units = (total / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)                 # same risk per asset
    pos = glide(exante_scale(units, px), GLIDE)                                          # constant book risk, slow trading
    pos = pos.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pos.index.name = DATE
    return pl.from_pandas(pos.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))


def _load(paths):
    frames = []
    for p in paths:
        df = pl.read_csv(p, try_parse_dates=True)
        df = df.with_columns(pl.col(DATE).cast(pl.Date), pl.exclude(DATE).cast(pl.Float64, strict=False))
        if any(c.endswith("|Close") for c in df.columns):                                # OHLC file: closes only
            df = df.select(DATE, *[pl.col(c).alias(c.split("|")[0]) for c in df.columns if c.endswith("|Close")])
        frames.append(df)
    return pl.concat([f.select(frames[0].columns) for f in frames]).unique(subset=DATE, keep="last").sort(DATE)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", nargs="+", required=True, help="in-sample csv, then out-of-sample csv")
    ap.add_argument("--macro", nargs="+", required=True, help="in-sample csv, then out-of-sample csv")
    ap.add_argument("--out", default=str(Path(__file__).parent / "positions.csv"))
    a = ap.parse_args()
    prices, macro = _load(a.prices), _load(a.macro)
    pos = positions(prices, macro)
    assert pos.columns == prices.columns and len(pos) == len(prices) and pos.null_count().sum_horizontal()[0] == 0
    pos.write_csv(a.out)
    print(f"{a.out}: {len(pos)} rows, {pos.width - 1} assets, {pos[DATE][0]} to {pos[DATE][-1]}")
