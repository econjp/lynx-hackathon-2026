"""Signals: EWMA crossovers (+ optional breakout)."""
import numpy as np
import pandas as pd

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
