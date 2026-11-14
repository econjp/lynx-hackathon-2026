"""Final team model: trend + inflation (+ twins) in ONE portfolio.

    final_positions(prices, macro) -> polars DataFrame (AsOfDate + one column per asset)

Sleeves (each is a forecast per asset, roughly in [-2, 2]):
  trend      Jaakko's slow pick: EWMAC 32/128 + 64/256 and 12-month momentum, 50/50, capped.
  inflation  per moon: CPI growth over k days minus its trailing 3-year mean, divided by its
             trailing 3-year std, tanh. Bonds: -tanh(z, k=60). Stocks: -tanh(z, k=120). Rest: 0.
  twins      pairs whose 5-day changes correlate > 0.8 on all data up to T (a rule, no hand pick).
             Signal = -tanh(z/2) of the log spread vs its 1-year mean; long one leg, short the other.

Combination: every sleeve forecast is divided by the trailing volatility of that sleeve's own
P&L (equal risk budget), the forecasts are summed, and the sum goes through Jaakko's sizing:
units = forecast / vol, book scaled to constant ex-ante risk, 30% daily glide.

Risk caps: the two legs of a twin pair get half weight each in the trend sleeve (one budget),
the bond block is capped at BOND_BETS bets of risk, ultrapure_silicon gets half weight.

Every parameter below was fixed before scoring. Nothing here is fitted on the data, and every
statistic is trailing (rolling / expanding / ewm), so positions at T only use data up to T.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "jaakko"))

import numpy as np
import pandas as pd
import polars as pl

from signals import SPEEDS, clean_prices, ewmac, scale_and_cap   # Jaakko
from risk import price_vol, size                                 # Jaakko
from portfolio import exante_scale, glide                        # Jaakko

DATE = "AsOfDate"
SLEEVES = ("trend", "inflation", "twins")
FINAL_SLEEVES = SLEEVES          # set by the rule in research/eemil/final_combined/final_validation.py (1995-2009 only)

K_BONDS, K_STOCKS = 60, 120      # CPI lookbacks (days)
MACRO_WIN, MACRO_MIN = 750, 250  # trailing 3 years for the CPI mean and std
TWIN_CORR, TWIN_MIN = 0.8, 250   # twin rule: expanding corr of 5-day changes
SPREAD_WIN, SPREAD_MIN = 250, 120
SLEEVE_WIN, SLEEVE_MIN = 250, 60  # trailing window for each sleeve's P&L vol
BOND_BETS = 2.0                  # bond block may carry at most this many average bets of risk
HALF_WEIGHT = ("ultrapure_silicon",)
ALPHA = 0.3                      # glide


def asset_class(a):
    for k in ("bonds", "stocks", "fx"):
        if a.endswith("_" + k):
            return k
    return "commodity"


def _to_pandas(frame):
    df = frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame.copy()
    df[DATE] = pd.to_datetime(df[DATE])
    return df.set_index(DATE).sort_index().astype(float)


# ---------------------------------------------------------------- sleeves
def trend_forecast(spx, vol):
    m250 = scale_and_cap(spx.diff().rolling(250).sum() / (vol * np.sqrt(250)))
    return (ewmac(spx, vol, SPEEDS["slow"]) + m250) / 2


def cpi_z(macro, index, k):
    """Per moon: k-day CPI growth minus its trailing mean, over its trailing std. Trailing only."""
    cpi = macro[[c for c in macro.columns if c.startswith("cpi|")]]
    # calendar days -> trading days: last value at or before T
    logc = np.log(cpi.reindex(cpi.index.union(index)).ffill().reindex(index))
    g = logc.diff(k)
    x = g - g.rolling(MACRO_WIN, min_periods=MACRO_MIN).mean()
    z = x / x.rolling(MACRO_WIN, min_periods=MACRO_MIN).std().replace(0, np.nan)
    return z.rename(columns=lambda c: c.split("|")[1])


def inflation_forecast(px, macro, k_bonds=K_BONDS, k_stocks=K_STOCKS):
    f = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for cls, k in (("bonds", k_bonds), ("stocks", k_stocks)):
        z = cpi_z(macro, px.index, k)
        for moon in z.columns:
            a = f"{moon}_{cls}"
            if a in f:
                f[a] = -np.tanh(z[moon])
    return f


def twin_pairs(px):
    """{(a, b): bool series} = pair is a twin on day T, using all data up to T only."""
    d5 = px.diff(5)
    cols, out = list(px.columns), {}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            on = d5[a].expanding(min_periods=TWIN_MIN).corr(d5[b]) > TWIN_CORR
            if on.any():
                out[(a, b)] = on
    return out


def twins_forecast(px, pairs):
    f = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    logp = np.log(px.where(px > 0))
    for (a, b), on in pairs.items():
        s = logp[a] - logp[b]
        r = s.rolling(SPREAD_WIN, min_periods=SPREAD_MIN)
        t = np.tanh((s - r.mean()) / r.std().replace(0, np.nan) / 2).where(on, 0.0).fillna(0.0)
        f[a] -= t
        f[b] += t
    return f


# ---------------------------------------------------------------- combination and caps
def sleeve_vol(f, vol, dp):
    """Trailing std of the sleeve's own P&L (2-day lag), measured only while the sleeve is on."""
    units = size(f, vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    live = (units.abs().sum(axis=1) > 0).shift(2, fill_value=False)
    pnl = (units.shift(2) * dp).sum(axis=1).where(live)
    return pnl.rolling(SLEEVE_WIN, min_periods=SLEEVE_MIN).std().replace(0, np.nan)


def cap_bonds(f, vol, dp, bets=None, window=250, start=60):
    """Scale the bond forecasts down on days when the bond block's ex-ante risk exceeds
    `bets` times the average stand-alone risk of a live position (one 'bet')."""
    bets = BOND_BETS if bets is None else bets
    bonds = [a for a in f.columns if asset_class(a) == "bonds"]
    if not bonds:
        return f
    x = size(f[bonds], vol[bonds]).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    d = dp[bonds].fillna(0.0).values
    absf = f.abs().values
    live = absf > 0
    bet = np.where(live.any(axis=1), absf.sum(axis=1) / np.maximum(live.sum(axis=1), 1), np.nan)
    scale = np.ones(len(f))
    for i in range(start, len(f)):
        risk = (d[max(1, i - window + 1): i + 1] @ x[i]).std()
        if risk > bets * bet[i] > 0:
            scale[i] = bets * bet[i] / risk
    out = f.copy()
    out[bonds] = f[bonds].mul(scale, axis=0)
    return out


def sleeve_forecasts(prices, macro, k_bonds=K_BONDS, k_stocks=K_STOCKS, caps=True):
    """Raw forecast per sleeve, plus the pieces the sizing needs."""
    px = _to_pandas(prices)
    mac = _to_pandas(macro)
    vol = price_vol(px)
    spx = clean_prices(px, vol)
    pairs = twin_pairs(px)

    trend = trend_forecast(spx, vol).fillna(0.0)
    if caps:  # twin legs share one budget in the trend sleeve
        w = pd.DataFrame(1.0, index=px.index, columns=px.columns)
        for (a, b), on in pairs.items():
            w.loc[on, [a, b]] = 0.5
        trend = trend * w
    fc = {
        "trend": trend,
        "inflation": inflation_forecast(px, mac, k_bonds, k_stocks).fillna(0.0),
        "twins": twins_forecast(px, pairs),
    }
    return fc, px, vol


def combine(fc, px, vol, sleeves=SLEEVES, caps=True, alpha=ALPHA, budget="equal"):
    """budget="equal": every sleeve gets the same risk. budget="breadth": a sleeve's variance budget is
    proportional to the number of assets it trades that day (risk per asset is then about equal)."""
    dp = px.diff()
    total = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for s in sleeves:
        scaled = fc[s].div(sleeve_vol(fc[s], vol, dp), axis=0).fillna(0.0)
        if budget == "breadth":
            scaled = scaled.mul(np.sqrt((fc[s].abs() > 0).sum(axis=1)), axis=0)
        total = total + scaled
    if caps:
        for a in HALF_WEIGHT:
            if a in total:
                total[a] = total[a] * 0.5
        total = cap_bonds(total, vol, dp)
    units = size(total, vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pos = glide(exante_scale(units, px), alpha).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pos.index.name = DATE
    return pl.from_pandas(pos.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))


def final_positions(prices, macro, sleeves=FINAL_SLEEVES, k_bonds=K_BONDS, k_stocks=K_STOCKS, caps=True):
    """prices: AsOfDate + one close column per asset. macro: AsOfDate + gdp|moon, cpi|moon.
    Returns AsOfDate + one position column per asset (traded at close T+2 by Lynx)."""
    fc, px, vol = sleeve_forecasts(prices, macro, k_bonds, k_stocks, caps)
    return combine(fc, px, vol, sleeves, caps)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "lynx_materials"))
    import evaluation as ev
    prices = pl.read_csv(ROOT / "data" / "close_prices_insample.csv", try_parse_dates=True)
    macro = pl.read_csv(ROOT / "data" / "macro_data_insample.csv", try_parse_dates=True)
    prices = prices.with_columns(pl.col(DATE).cast(pl.Date))
    pos = final_positions(prices, macro)
    for k, v in ev.calc_key_figures(pos, prices).items():
        print(f"{k:16s} {v:.3f}")
