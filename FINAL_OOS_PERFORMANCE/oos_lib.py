"""Helpers for the final out-of-sample report.

Everything here re-uses SUBMISSION/model.py unchanged. The only new code takes the final
positions apart into additive pieces, so each piece's P&L adds up exactly to the submitted book:

    final = 0.5 * GOAT + 0.5 * BOMBASTRAT
    BOMBASTRAT = trend + inflation + twins          (exact: ex-ante scale is one number per day, glide is linear)
    GOAT       = engine 1 + engine 2 + deadband     (exact: the deadband's effect is kept as its own small line)

P&L follows lynx_materials/evaluation.py exactly: position of day t earns the price change of t+2,
slippage = 0.02 * |trade| * trailing 100-day price-change std.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "SUBMISSION"), str(ROOT / "lynx_materials")]

import numpy as np
import pandas as pd
import polars as pl

import evaluation as ev
import model as M

DATE = M.DATE
LAG, COSTS, YEAR = ev.POSITION_LAG, -0.02, ev.TRADING_DAYS_PER_YEAR
OOS_START, SPLIT = pd.Timestamp("2017-01-01"), pd.Timestamp("2021-01-01")


# ---------------------------------------------------------------- data
def load_scenario(k):
    prices = M.load([ROOT / "data" / f"close_prices_outofsample{k}.csv"])
    macro = M.load([ROOT / "data" / f"macro_data_outofsample{k}.csv"])
    submitted = pl.read_csv(ROOT / "SUBMISSION" / f"positions_outofsample{k}.csv", try_parse_dates=True)
    submitted = submitted.with_columns(pl.col(DATE).cast(pl.Date))
    return prices, macro, submitted


def to_pd(x):
    df = x.to_pandas() if isinstance(x, pl.DataFrame) else x.copy()
    df[DATE] = pd.to_datetime(df[DATE])
    return df.set_index(DATE).sort_index().astype(float)


def to_pl(df):
    out = df.copy()
    out.index.name = DATE
    return pl.from_pandas(out.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))


def sector(c):
    return M.get_sector(c)


# ---------------------------------------------------------------- decomposition
def exante_factor(units, px, window=250, start=60):
    """The scalar k_t that model.exante_scale multiplies the book by (x' Cov x = 1)."""
    d, x = px.diff().fillna(0.0).values, units.fillna(0.0).values
    k = np.zeros(len(x))
    for i in range(start, len(x)):
        cov = np.cov(d[max(1, i - window + 1): i + 1].T)
        var = x[i] @ cov @ x[i]
        if var > 0:
            k[i] = 1 / np.sqrt(var)
    return pd.Series(k, index=units.index)


def bomba_parts(prices, macro):
    f, px, vol = M.forecasts(prices, macro)
    dp = px.diff()
    units = {s: (f[s].div(M.own_risk(f[s], vol, dp), axis=0).fillna(0.0) / vol)
             .replace([np.inf, -np.inf], np.nan).fillna(0.0) for s in f}
    k = exante_factor(sum(units.values()), px)
    return {s: M.glide(u.mul(k, axis=0), M.GLIDE).fillna(0.0) for s, u in units.items()}


def goat_parts(prices):
    """Re-traces model.goat_positions, keeping engine 1, engine 2 and the deadband's effect apart."""
    px = to_pd(prices)
    cols = list(px.columns)
    stocks = [c for c in cols if sector(c) == "stocks"]
    bonds = [c for c in cols if sector(c) == "bonds"]
    fx = [c for c in cols if sector(c) == "fx"]
    comms = [c for c in cols if sector(c) == "commodity"]
    sb = stocks + bonds
    dp = px.diff()
    vol = np.sqrt((dp ** 2).rolling(100, min_periods=20).mean()).bfill()
    ef, es = px.ewm(span=10, adjust=False).mean(), px.ewm(span=100, adjust=False).mean()
    gap, trend = ((ef - es) / vol).fillna(0.0), (ef > es).astype(int)
    kris = pd.DataFrame({c: M.stretch_stop_positions(px[c].values, gap[c].values, trend[c].values, 7.0, 2)
                         for c in sb}, index=px.index)
    u1 = (kris / vol[sb]).reindex(columns=cols).fillna(0.0).to_numpy()
    ret = px.pct_change(252).fillna(0.0)
    sig = pd.concat([2.0 * (ret[g].rank(axis=1) / len(g) - 0.5) for g in (stocks, bonds, fx, comms)], axis=1)[cols]
    u2 = (sig / vol).fillna(0.0).to_numpy()
    g1, g2 = M.apply_glide(0.6 * u1, 0.25), M.apply_glide(0.4 * u2, 0.25)
    final = M.apply_deadband(g1 + g2, 0.10)
    k = exante_factor(pd.DataFrame(final, index=px.index, columns=cols), px).to_numpy()[:, None]
    mk = lambda a: pd.DataFrame(a * k, index=px.index, columns=cols)
    return {"engine1_stop_trend": mk(g1), "engine2_xs_momentum": mk(g2), "deadband": mk(final - g1 - g2)}


def all_parts(prices, macro, w_goat=M.W_GOAT):
    """Additive pieces of the submitted book (already weighted 50/50)."""
    parts = {f"BOMBA {s}": (1 - w_goat) * p for s, p in bomba_parts(prices, macro).items()}
    parts.update({f"GOAT {s}": w_goat * p for s, p in goat_parts(prices).items()})
    return parts


# ---------------------------------------------------------------- P&L, same maths as evaluation.py
def pnl_frames(pos, prices):
    """Per-instrument daily gross P&L and slippage on the price dates (evaluation.py conventions)."""
    px = to_pd(prices)
    pos = (to_pd(pos) if isinstance(pos, pl.DataFrame) else pos).reindex(px.index).ffill()
    ret = px.diff()
    vol = ret.rolling(ev.VOLATILITY_WINDOW, min_periods=10).std().ffill().shift(1).bfill()
    held = pos.shift(LAG)
    gross = ret * held
    slip = held.diff().abs() * vol * COSTS
    return gross, slip


def daily(pos, prices):
    g, s = pnl_frames(pos, prices)
    return (g.sum(axis=1, min_count=1) + s.sum(axis=1, min_count=1)).fillna(0.0)


# ---------------------------------------------------------------- statistics
def sharpe(r):
    r = pd.Series(r).dropna()
    return r.mean() / r.std() * np.sqrt(YEAR) if r.std() > 0 else np.nan


def adj_sharpe(r):
    r = pd.Series(r).dropna()
    s = sharpe(r)
    return s * (1 + s / 6 * r.skew() - s ** 2 / 24 * r.kurt())


def stats(r):
    r = pd.Series(r).dropna()
    eq = r.cumsum()
    dd = eq - eq.cummax()
    return {"adj. Sharpe": adj_sharpe(r), "Sharpe": sharpe(r), "ann. P&L": r.mean() * YEAR,
            "ann. vol": r.std() * np.sqrt(YEAR), "skew": r.skew(), "kurtosis": r.kurt() + 3,
            "max DD": dd.min(), "hit rate": (r > 0).mean()}


def rolling_adj_sharpe(r, window):
    m = r.rolling(window).mean()
    sd = r.rolling(window).std()
    s = m / sd * np.sqrt(YEAR)
    return s * (1 + s / 6 * r.rolling(window).skew() - s ** 2 / 24 * r.rolling(window).kurt())


def expanding_adj_sharpe(r, min_days=130):
    s = r.expanding(min_days).mean() / r.expanding(min_days).std() * np.sqrt(YEAR)
    return s * (1 + s / 6 * r.expanding(min_days).skew() - s ** 2 / 24 * r.expanding(min_days).kurt())


def drawdowns(r, n=5):
    """The n deepest drawdowns: peak, trough, recovery, depth, length."""
    eq = r.cumsum()
    dd = eq - eq.cummax()
    out, used = [], pd.Series(False, index=r.index)
    for _ in range(n):
        cand = dd.where(~used)
        if cand.min() >= 0 or cand.isna().all():
            break
        trough = cand.idxmin()
        peak = eq[:trough].idxmax()
        after = eq[trough:]
        rec = after[after >= eq[peak]]
        end = rec.index[0] if len(rec) else r.index[-1]
        out.append({"peak": peak.date(), "trough": trough.date(),
                    "recovered": rec.index[0].date() if len(rec) else "not yet",
                    "depth": dd[trough], "days to trough": (r.index.get_loc(trough) - r.index.get_loc(peak))})
        used[peak:end] = True
    return pd.DataFrame(out)
