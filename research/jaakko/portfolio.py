"""Portfolio: constant ex-ante risk + slow execution."""
import numpy as np
import pandas as pd


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
