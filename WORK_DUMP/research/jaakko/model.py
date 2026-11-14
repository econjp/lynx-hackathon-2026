"""JP model: EWMAC -> cap -> vol sizing -> ex-ante risk -> glide."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

from signals import SPEEDS, clean_prices, ewmac, breakout
from risk import price_vol, size
from portfolio import exante_scale, glide

DATE = "AsOfDate"


def generate_positions(prices, speed="mid", clean=True, w_breakout=0.0, alpha=0.3):
    px = prices.to_pandas() if isinstance(prices, pl.DataFrame) else prices.copy()
    px = px.set_index(DATE).astype(float)

    vol = price_vol(px)
    sig_px = clean_prices(px, vol) if clean else px     # no roll jumps in signals
    f = ewmac(sig_px, vol, SPEEDS[speed])
    if w_breakout:
        f = (1 - w_breakout) * f + w_breakout * breakout(sig_px)

    pos = exante_scale(size(f, vol), px)
    pos = glide(pos, alpha).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pos.index.name = DATE
    return pl.from_pandas(pos.reset_index())
