"""FINAL SUBMISSION MODEL: 50% BOMBASTRAT + 50% GOAT, one portfolio.

    python SUBMISSION/model.py --prices data/close_prices_insample.csv <out_of_sample_prices.csv> \
                               --macro  data/macro_data_insample.csv  <out_of_sample_macro.csv> \
                               --out    SUBMISSION/positions.csv

This file is self-contained (numpy, pandas, polars only).

BOMBASTRAT (market neutral, three forecasts with equal risk each, added):
  trend      all 27 assets: slow price trend (EWMAC 32/128 + 64/256 and 12-month momentum)
  inflation  bonds and stocks: CPI running hotter than its 3-year normal -> short
  twins      commodity pairs that move together: bet on a wide gap closing
GOAT (long tilt in stocks and bonds):
  engine 1   EMA 10/100 long-only with a stretch-activated trailing stop on the 10 stocks and bonds
  engine 2   12-month cross-sectional momentum ranked inside each asset class

Both books are scaled to the same constant risk (trailing 250-day covariance) and averaged 50/50.
A position for day T uses data up to and including day T only (Lynx trades it at the close of T+2).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

W_GOAT = 0.5

# ---- from Jaakko's signals

# speed sets, pick one on train data
SPEEDS = {
    "fast": [(12, 48), (20, 80)],
    "mid": [(16, 64), (32, 128), (64, 256)],
    "slow": [(32, 128), (64, 256)],
    "all": [(12, 48), (20, 80), (32, 128), (64, 256)],
}


def clean_prices(px, vol, k=5.0):
    # clip one-day jumps at 5x vol (roll gaps, bad prints). only for signals, P&L uses raw prices
    diff = px.diff()
    clipped = diff.clip(lower=-k * vol.shift(1), upper=k * vol.shift(1)).fillna(diff)
    return clipped.cumsum()


def scale_and_cap(raw, cap=2.0):
    # avg |f| ~ 1 using past data only, then cap at +-2
    return (raw / raw.abs().expanding(min_periods=250).mean()).clip(-cap, cap)


def ewmac(px, vol, speeds):
    # (fast EMA - slow EMA) / vol, averaged over the speed pairs
    fs = []
    for fast, slow in speeds:
        raw = (px.ewm(span=fast, min_periods=fast).mean() - px.ewm(span=slow, min_periods=slow).mean()) / vol
        fs.append(scale_and_cap(raw))
    return sum(fs) / len(fs)


def breakout(px, windows=(80, 160, 320)):
    # where price sits in its N-day high/low range, -1..+1. ~0.95 corr with ewmac, so off by default
    fs = []
    for w in windows:
        hi, lo = px.rolling(w, min_periods=w).max(), px.rolling(w, min_periods=w).min()
        fs.append(2 * (px - lo) / (hi - lo).replace(0, np.nan) - 1)
    return sum(fs) / len(fs)

# ---- from Jaakko's risk


def price_vol(px):
    # 70% fast EWMA vol + 30% its 500d average, so size doesn't swing too much
    diff = px.diff()
    fast = np.sqrt((diff ** 2).ewm(span=35, min_periods=30).mean())
    slow = fast.rolling(500, min_periods=60).mean()
    return (0.7 * fast + 0.3 * slow).replace(0, np.nan)


def size(forecast, vol):
    # units = forecast / vol -> same money risk per asset
    return forecast / vol

# ---- from Jaakko's portfolio


def exante_scale(units, px, window=250, start=60):
    # scale the whole book so x' Cov x = 1 every day (cov of price CHANGES, same units as P&L)
    d, x = px.diff().fillna(0.0).values, units.fillna(0.0).values
    out = np.zeros_like(x)
    for i in range(start, len(x)):
        cov = np.cov(d[max(1, i - window + 1): i + 1].T)
        var = x[i] @ cov @ x[i]
        if var > 0:
            out[i] = x[i] / np.sqrt(var)
    return pd.DataFrame(out, index=units.index, columns=units.columns)


def glide(pos, alpha=0.3):
    # move 30% toward target each day -> less trading, less slippage
    return pos.ewm(alpha=alpha, adjust=False).mean()


# ---------------------------------------------------------------- BOMBASTRAT
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


def bombastrat_positions(prices, macro, use=("trend", "inflation", "twins")):
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


# ---------------------------------------------------------------- GOAT
def get_sector(asset_name: str) -> str:
    """Classifies asset into one of 4 sectors: stocks, bonds, fx, commodity."""
    for s in ("bonds", "stocks", "fx"):
        if asset_name.endswith("_" + s) or asset_name.endswith(s):
            return s
    return "commodity"

def stretch_stop_positions(
    close: np.ndarray,
    gap: np.ndarray,
    trend: np.ndarray,
    thr: float = 7.0,
    up_moves: int = 2
) -> np.ndarray:
    """
    Daily position (1.0 or 0.0) for a single asset with stretch ratchet trailing stop.
    Strictly causal (uses only data up to date t).
    """
    n = len(close)
    pos = np.zeros(n)
    active = out_ = False
    armed = True
    stop = ref = np.nan
    up = 0
    prev_trend = 0

    for i in range(n):
        t = trend[i]
        if t != prev_trend:
            active = out_ = False
        prev_trend = t
        if t == 0:
            continue

        if out_:
            if gap[i] < thr:
                out_ = False
            if out_:
                continue

        if not active and gap[i] >= thr:
            active, armed, stop, ref, up = True, True, np.nan, np.nan, 0

        if active:
            up = up + 1 if close[i] > close[i - 1] else 0
            if not np.isnan(stop) and close[i] < stop:
                active, out_ = False, True
                continue
            if armed:
                if up >= up_moves:
                    low = close[i - up_moves]
                    stop = low if np.isnan(stop) else max(stop, low)
                if not np.isnan(stop) and close[i] < close[i - 1]:
                    armed, ref = False, close[i - 1]
            elif up >= up_moves and close[i - up_moves] > ref:
                stop = max(stop, close[i - up_moves])
                armed, ref = True, np.nan

        pos[i] = 1.0
    return pos

def apply_glide(mat: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """Exponential glide filter smoothing target changes."""
    out = np.empty_like(mat)
    out[0] = mat[0]
    om_a = 1.0 - alpha
    for t in range(1, len(mat)):
        out[t] = om_a * out[t - 1] + alpha * mat[t]
    return out

def apply_deadband(mat: np.ndarray, thr: float = 0.10) -> np.ndarray:
    """10% rebalancing deadband filter suppressing minor turnover."""
    if thr == 0.0:
        return mat
    T, N = mat.shape
    out = np.empty_like(mat)
    out[0] = mat[0]
    eps = 1e-6
    for t in range(1, T):
        target = mat[t]
        curr = out[t - 1]
        denom = np.maximum(np.maximum(np.abs(target), np.abs(curr)), eps)
        rel = np.abs(target - curr) / denom
        mask = rel < thr
        out[t] = np.where(mask, curr, target)
    return out

def goat_positions(prices_input: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    """
    Computes final trading positions for the GOAT Model.
    Input: pl.DataFrame or pd.DataFrame with AsOfDate and 27 asset columns.
    Output: pl.DataFrame with AsOfDate and 27 asset columns.
    """
    if isinstance(prices_input, pl.DataFrame):
        df_px = prices_input.to_pandas()
        date_series = prices_input.get_column(DATE)
    else:
        df_px = prices_input.copy()
        date_series = pl.Series(DATE, df_px[DATE])

    if DATE in df_px.columns:
        df_px = df_px.drop(columns=[DATE])

    cols = df_px.columns.tolist()

    # Asset classifications
    stocks = [c for c in cols if get_sector(c) == "stocks"]
    bonds = [c for c in cols if get_sector(c) == "bonds"]
    fx = [c for c in cols if get_sector(c) == "fx"]
    comms = [c for c in cols if get_sector(c) == "commodity"]
    sb = stocks + bonds

    # Daily 100-day rolling price volatility
    dp = df_px.diff()
    vol = np.sqrt((dp ** 2).rolling(100, min_periods=20).mean()).bfill()

    # Engine 1: Kristofer's EMA 10/100 Stretch Stop Core
    ef = df_px.ewm(span=10, adjust=False).mean()
    es = df_px.ewm(span=100, adjust=False).mean()
    gap = ((ef - es) / vol).fillna(0.0)
    trend = (ef > es).astype(int)

    kris_sigs = {}
    for c in sb:
        kris_sigs[c] = stretch_stop_positions(
            df_px[c].values, gap[c].values, trend[c].values, thr=7.0, up_moves=2
        )
    df_kris = pd.DataFrame(kris_sigs, index=df_px.index)
    units_kris = (df_kris / vol[sb]).reindex(columns=cols).fillna(0.0).to_numpy()

    # Engine 2: Sector-Neutral Cross-Sectional Momentum (252-day lookback)
    ret_xs = df_px.pct_change(252).fillna(0.0)
    parts = []
    for grp in [stocks, bonds, fx, comms]:
        if len(grp) > 0:
            r = ret_xs[grp].rank(axis=1)
            parts.append(2.0 * (r.divide(len(grp)) - 0.5))
    sig_xs = pd.concat(parts, axis=1)[cols]
    units_xs = (sig_xs / vol).fillna(0.0).to_numpy()

    # Dual-Engine Blend (60% Kristofer / 40% XS)
    raw_blend = 0.60 * units_kris + 0.40 * units_xs

    # Layer 4 Execution: Glide + 10% Deadband
    glided = apply_glide(raw_blend, alpha=0.25)
    final_mat = apply_deadband(glided, thr=0.10)

    # Format output frame
    return pl.DataFrame({DATE: date_series, **{c: final_mat[:, j] for j, c in enumerate(cols)}})


# ---------------------------------------------------------------- the 50/50
def final_positions(prices, macro, w_goat=W_GOAT):
    """prices: AsOfDate + one close per asset. macro: AsOfDate + cpi|moon columns. Returns positions, same shape as prices."""
    px = _frame(prices)
    g = goat_positions(prices).to_pandas().set_index(DATE)
    g.index = pd.to_datetime(g.index)
    g = exante_scale(g, px)                                   # GOAT at constant book risk
    b = bombastrat_positions(prices, macro).to_pandas().set_index(DATE)
    b.index = pd.to_datetime(b.index)                         # BOMBASTRAT is already at constant book risk
    pos = (w_goat * g + (1 - w_goat) * b).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pos.index.name = DATE
    return pl.from_pandas(pos.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))


def load(paths):
    """Stack in-sample and out-of-sample files by date (later file wins on overlap). OHLC files: closes only."""
    return _load(paths)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", nargs="+", required=True, help="in-sample csv, then out-of-sample csv")
    ap.add_argument("--macro", nargs="+", required=True, help="in-sample csv, then out-of-sample csv")
    ap.add_argument("--out", default=str(Path(__file__).parent / "positions.csv"))
    a = ap.parse_args()
    prices, macro = load(a.prices), load(a.macro)
    pos = final_positions(prices, macro)
    assert pos.columns == prices.columns and len(pos) == len(prices) and pos.null_count().sum_horizontal()[0] == 0
    pos.write_csv(a.out)
    print(f"{a.out}: {len(pos)} rows, {pos.width - 1} assets, {pos[DATE][0]} to {pos[DATE][-1]}")
