"""Parameter sweeps for the white paper. Writes whitepaper/sweeps.json and whitepaper/fig_*.png.

    python whitepaper/sweeps.py

Every row changes one setting (or one pair of settings) of BOMBASTRAT and rescores the whole model
with Lynx's evaluation.py from 1997-05-22. Nothing here feeds back into the model.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "research" / "jaakko"), str(ROOT / "lynx_materials")]

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluation as ev
from BOMBASTRAT import bombastrat as bs
from signals import clean_prices, ewmac, scale_and_cap
from portfolio import exante_scale, glide

DATE = "AsOfDate"
prices = pl.read_csv(ROOT / "data" / "close_prices_insample.csv", try_parse_dates=True).with_columns(pl.col(DATE).cast(pl.Date))
macro = pl.read_csv(ROOT / "data" / "macro_data_insample.csv", try_parse_dates=True).with_columns(pl.col(DATE).cast(pl.Date))
px, mac = bs._frame(prices), bs._frame(macro)
vol, dp = bs.price_vol(px), px.diff()
d = prices[DATE]
START = pd.Timestamp("1997-05-22").date()
ERAS = [(START, pd.Timestamp("2002-05-01").date()), (pd.Timestamp("2002-05-01").date(), pd.Timestamp("2009-05-01").date()),
        (pd.Timestamp("2009-05-01").date(), d[-1])]


def score(pos):
    pos = pos.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pos.index.name = DATE
    p = pl.from_pandas(pos.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))
    s = lambda a, b: ev.calc_key_figures(p.filter(pl.col(DATE).is_between(a, b)), prices, key_figures=["adjusted_sharpe"])["adjusted_sharpe"]
    return [round(s(START, d[-1]), 3)] + [round(s(a, b), 3) for a, b in ERAS]


def combine(fs, glide_a=bs.GLIDE, risk_days=bs.RISK_DAYS):
    old = bs.RISK_DAYS
    bs.RISK_DAYS = risk_days
    total = sum(f.div(bs.own_risk(f, vol, dp), axis=0).fillna(0.0) for f in fs)
    bs.RISK_DAYS = old
    units = (total / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return glide(exante_scale(units, px), glide_a)


def trend_f(speeds, mom_days=250):
    clean = clean_prices(px, vol)
    mom = scale_and_cap(clean.diff().rolling(mom_days).sum() / (vol * np.sqrt(mom_days)))
    return ((ewmac(clean, vol, speeds) + mom) / 2).fillna(0.0)


def infl_f(kb=60, ks=120, normal=750):
    old_k, old_n = dict(bs.CPI_DAYS), bs.NORMAL
    bs.CPI_DAYS, bs.NORMAL = {"bonds": kb, "stocks": ks}, normal
    f = bs.inflation(px, mac)
    bs.CPI_DAYS, bs.NORMAL = old_k, old_n
    return f


def twin_f(corr=0.8, spread=250):
    old = bs.TWIN_CORR, bs.SPREAD_DAYS
    bs.TWIN_CORR, bs.SPREAD_DAYS = corr, spread
    f = bs.twins(px)
    bs.TWIN_CORR, bs.SPREAD_DAYS = old
    return f


T0, I0, W0 = trend_f([(32, 128), (64, 256)]), infl_f(), twin_f()
CACHE = HERE / "sweeps.json"
out = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def save():
    CACHE.write_text(json.dumps(out, indent=1))


out["base"] = out.get("base") or score(combine([T0, I0, W0]))
print("base", out["base"], flush=True)

# 1. trend: one EWMAC pair (fast, slow) in place of the two pairs, momentum kept
fasts, ratios = [8, 16, 32, 64, 128], [2, 4, 8]
grid = out.get("trend_grid", {}).get("cells", {})
for f_ in fasts:
    for r in ratios:
        if f"{f_}/{f_ * r}" in grid:
            continue
        grid[f"{f_}/{f_ * r}"] = score(combine([trend_f([(f_, f_ * r)]), I0, W0]))
        print("trend", f_, r, grid[f"{f_}/{f_ * r}"], flush=True)
out["trend_grid"] = {"fasts": fasts, "ratios": ratios, "cells": grid}
out["trend_mom"] = out.get("trend_mom") or {str(m): score(combine([trend_f([(32, 128), (64, 256)], m), I0, W0])) for m in [60, 120, 250, 500]}
print("mom", out["trend_mom"], flush=True)

# 2. inflation lookbacks (bonds x stocks) and the 'normal' window
ks = [20, 60, 120, 250]
save()
out["infl_grid"] = out.get("infl_grid") or {"k": ks, "cells": {f"{a}/{b}": score(combine([T0, infl_f(a, b), W0])) for a in ks for b in ks}}
print("infl grid done", flush=True)
save()
out["infl_normal"] = out.get("infl_normal") or {str(n): score(combine([T0, infl_f(normal=n), W0])) for n in [250, 500, 750, 1000, 1500]}
print("normal", out["infl_normal"], flush=True)

# 3. twins: threshold x spread window
cs, ss = [0.6, 0.7, 0.8, 0.9], [120, 180, 250, 500]
save()
out["twin_grid"] = {"corr": cs, "spread": ss, "cells": {f"{c}/{s}": score(combine([T0, I0, twin_f(c, s)])) for c in cs for s in ss}}
print("twin grid done", flush=True)

# 4. execution and risk
out["glide"] = {str(g): score(combine([T0, I0, W0], glide_a=g)) for g in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]}
out["risk_days"] = {str(r): score(combine([T0, I0, W0], risk_days=r)) for r in [60, 125, 250, 500, 1000]}
print("exec done", flush=True)
# 5. costs
pos0 = combine([T0, I0, W0]); pos0.index.name = DATE
p0 = pl.from_pandas(pos0.fillna(0.0).reset_index()).with_columns(pl.col(DATE).cast(pl.Date)).filter(pl.col(DATE) >= START)
out["costs"] = {str(m): round(ev.calc_key_figures(p0, prices, costs=-0.02 * m, key_figures=["adjusted_sharpe"])["adjusted_sharpe"], 3) for m in [0, 1, 2, 3, 5]}
save()
print("saved sweeps.json")
