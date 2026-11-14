"""Eemil model: blended EWMAC z-scores -> tanh -> vol sizing -> ex-ante risk -> glide.

Design rule: nothing is picked on the data. Every signal is normalised by its
theoretical random-walk std (closed form), so there is no fitted scaling step,
and the speeds are a fixed geometric grid that is blended with equal weights.

Parameters (all of them): speeds, vol_span, risk_win, alpha, clip_k.
"""
import numpy as np
import pandas as pd
import polars as pl

DATE = "AsOfDate"
SPEEDS = (8, 16, 32, 64)  # fast EMA span; slow span = 4 * fast


def _prep(prices):
    px = prices.to_pandas() if isinstance(prices, pl.DataFrame) else prices.copy()
    px = px.set_index(DATE).astype(float)
    live = px.notna().cummax()            # asset is tradable from its first print
    return px.ffill(), live               # holidays / missing prints: carry last price


def _vol(diff, span):
    return np.sqrt((diff ** 2).ewm(span=span, min_periods=span // 2).mean()).replace(0, np.nan)


def _clean_diff(diff, span, k):
    # clip one-day moves at k * yesterday's vol (roll gaps, bad prints). signals + vol only
    v = _vol(diff, span).shift(1)
    return diff.clip(lower=-k * v, upper=k * v).fillna(diff)


def _ewmac_z(px, vol, fast, slow):
    # (EMA_fast - EMA_slow) / its std if prices were a driftless random walk with daily std = vol
    qf, qs = 1 - 2 / (fast + 1), 1 - 2 / (slow + 1)
    norm = np.sqrt(qs**2 / (1 - qs**2) + qf**2 / (1 - qf**2) - 2 * qs * qf / (1 - qs * qf))
    raw = px.ewm(span=fast, min_periods=slow).mean() - px.ewm(span=slow, min_periods=slow).mean()
    return raw / (vol * norm)


def _risk_scale(units, diff, window, start=60):
    # scale the whole book so the last `window` days of price changes give it unit daily std
    d, x = diff.fillna(0.0).values, units.fillna(0.0).values
    out = np.zeros_like(x)
    for i in range(start, len(x)):
        pnl = d[max(1, i - window + 1): i + 1] @ x[i]
        s = pnl.std()
        if s > 0:
            out[i] = x[i] / s
    return pd.DataFrame(out, index=units.index, columns=units.columns)


def forecast(prices, speeds=SPEEDS, vol_span=60, clip_k=5.0):
    """Blended trend forecast in [-1, 1] and the vol used for sizing (pandas, index = date)."""
    px, live = _prep(prices)
    diff = px.diff()
    cdiff = _clean_diff(diff, vol_span, clip_k)
    vol = _vol(cdiff, vol_span)
    spx = cdiff.fillna(0.0).cumsum()      # jump-free price path for the signal
    f = sum(np.tanh(_ewmac_z(spx, vol, s, 4 * s)) for s in speeds) / len(speeds)
    return f.where(live), vol, diff


def generate_positions(prices, speeds=SPEEDS, vol_span=60, risk_win=250, alpha=0.3, clip_k=5.0):
    f, vol, diff = forecast(prices, speeds, vol_span, clip_k)
    units = f / vol                                   # P&L is units * dP -> equal risk per asset
    pos = _risk_scale(units, diff, risk_win)          # constant ex-ante book risk
    pos = pos.ewm(alpha=alpha, adjust=False).mean()   # glide toward target -> less slippage
    pos = pos.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pos.index.name = DATE
    return pl.from_pandas(pos.reset_index())
