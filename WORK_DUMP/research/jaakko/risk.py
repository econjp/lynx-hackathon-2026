"""Sizing: price-change vol (Lynx P&L is X * dP)."""
import numpy as np


def price_vol(px):
    # 70% fast EWMA vol + 30% its 500d average, so size doesn't swing too much
    diff = px.diff()
    fast = np.sqrt((diff ** 2).ewm(span=35, min_periods=30).mean())
    slow = fast.rolling(500, min_periods=60).mean()
    return (0.7 * fast + 0.3 * slow).replace(0, np.nan)


def size(forecast, vol):
    # units = forecast / vol -> same money risk per asset
    return forecast / vol
