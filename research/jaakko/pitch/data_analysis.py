"""Data analysis of the Lynx in-sample data. Rebuilds DATA.md from raw files.
python research/jaakko/pitch/data_analysis.py  ->  research/jaakko/pitch/DATA.md"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(ROOT), str(ROOT / "lynx_materials")]

import numpy as np
import pandas as pd
import polars as pl
import evaluation as ev
from signals import SPEEDS, clean_prices, ewmac, scale_and_cap
from risk import price_vol, size
from portfolio import exante_scale, glide

DATE = "AsOfDate"
px = pd.read_csv(ROOT / "data" / "close_prices_insample.csv", parse_dates=[DATE]).set_index(DATE).astype(float)
ohlc = pd.read_csv(ROOT / "data" / "prices_insample.csv", parse_dates=[DATE]).set_index(DATE).astype(float)
macro = pd.read_csv(ROOT / "data" / "macro_data_insample.csv", parse_dates=[DATE]).set_index(DATE).astype(float)
dP = px.diff()
out = []


def table(title, df, note=""):
    out.append(f"\n## {title}\n" + (f"{note}\n\n" if note else "\n") + df.to_markdown(index=False, floatfmt=".2f"))


def cls(a):
    for k in ("bonds", "stocks", "fx"):
        if a.endswith(k):
            return k
    return "commodity"


CLS = {a: cls(a) for a in px.columns}
PLANETS = ["ganymede", "io", "titan", "callisto", "europa"]

# D1 asset overview
sd = dP.rolling(100, min_periods=50).std().shift(1)
rows = []
for a in px.columns:
    p, d = px[a], dP[a]
    rows.append({"asset": a, "class": CLS[a], "first": p.iloc[0], "last": p.iloc[-1], "min": p.min(),
                 "daily vol dP (x1000)": 1000 * d.std(), "vol % of price": 100 * (d / p.shift(1)).std(),
                 "days >6 sigma": int((d.abs() > 6 * sd[a]).sum()), "biggest day (sigma)": (d.abs() / sd[a]).max(),
                 "flat days %": 100 * (d == 0).mean()})
table("D1. Assets", pd.DataFrame(rows).sort_values(["class", "asset"]),
      f"{px.shape[1]} assets, {px.index[0].date()} to {px.index[-1].date()}, {len(px)} days. "
      "vol of dP is what Lynx P&L uses (units x price change).")

# D2 correlation by class
c = dP.corr()
rows = []
for g1 in ["bonds", "stocks", "fx", "commodity"]:
    r = {"class": g1}
    for g2 in ["bonds", "stocks", "fx", "commodity"]:
        a1 = [a for a in px if CLS[a] == g1]; a2 = [a for a in px if CLS[a] == g2]
        v = [c.loc[x, y] for x in a1 for y in a2 if x != y]
        r[g2] = np.mean(v)
    rows.append(r)
table("D2. Average correlation of daily changes, by class", pd.DataFrame(rows),
      "Diagonal = within the class. High within = the covariance scaling matters there.")

rows = []
for pl_ in PLANETS:
    b, s = f"{pl_}_bonds", f"{pl_}_stocks"
    fx = f"{pl_}_titan_fx" if f"{pl_}_titan_fx" in px else None
    rows.append({"planet": pl_, "bonds-stocks": c.loc[b, s] if b in c and s in c else np.nan,
                 "stocks-fx": c.loc[s, fx] if fx else np.nan, "bonds-fx": c.loc[b, fx] if fx else np.nan})
table("D3. Correlation within each planet", pd.DataFrame(rows), "titan has no fx: the fx pairs are quoted against titan.")

# D4 is there trend? non-overlapping 1-month steps
m = px.resample("ME").last()
r1, r12 = m.diff(), m.diff(12)
rows = []
for g in ["bonds", "stocks", "fx", "commodity"]:
    a = [x for x in px if CLS[x] == g]
    past, nxt = r12[a].shift(1).stack(), r1[a].stack()
    j = pd.concat([past, nxt], axis=1).dropna()
    hit = (np.sign(j[0]) == np.sign(j[1])).mean()
    ac = np.nanmean([m[x].diff().autocorr(1) for x in a])
    rows.append({"class": g, "assets": len(a), "hit rate: 12m trend -> next month": 100 * hit,
                 "autocorr monthly changes": ac, "months": len(j)})
table("D4. Is there trend in the data?", pd.DataFrame(rows),
      "Hit rate > 50% = past 12m direction predicts next month's direction. 50% = coin flip "
      "(+-2% is noise at ~1000 months).")

# the pick: EWMAC slow + 12m momentum
vol = price_vol(px); spx = clean_prices(px, vol)
f = (ewmac(spx, vol, SPEEDS["slow"]) + scale_and_cap(spx.diff().rolling(250).sum() / (vol * np.sqrt(250)))) / 2
pos = glide(exante_scale(size(f, vol), px)).fillna(0.0)
pnl = pos.shift(2) * dP - 0.02 * sd.fillna(0) * pos.diff().abs()   # same cost rule as Lynx, roughly
pnl = pnl.fillna(0.0)

# D5 by class
tot = pnl.sum().sum()
rows = []
for g in ["bonds", "stocks", "fx", "commodity"]:
    a = [x for x in px if CLS[x] == g]
    s = pnl[a].sum(axis=1)
    rows.append({"class": g, "share of P&L %": 100 * pnl[a].sum().sum() / tot,
                 "sharpe of the class": s.mean() / s.std() * np.sqrt(260),
                 "assets making money": f"{int((pnl[a].sum() > 0).sum())}/{len(a)}"})
table("D5. Where the pick makes money (EWMAC slow + 12m momentum)", pd.DataFrame(rows),
      "P&L recomputed here as pos(t-2) x dP minus 0.02 x vol x |trade|, close to Lynx's numbers.")

# D6 per year
p = pl.from_pandas(pos.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))
prices = pl.from_pandas(px.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))
r = ev.calc_key_figures(p, prices, key_figures=["returns"])["returns"].to_pandas().set_index(DATE).iloc[:, 0]
r.index = pd.to_datetime(r.index)
y = r.groupby(r.index.year).agg(lambda x: x.mean() / x.std() * np.sqrt(260) if x.std() > 0 else np.nan)
table("D6. Sharpe per year (the pick, Lynx returns incl. slippage)",
      pd.DataFrame({"year": y.index.astype(str), "sharpe": y.values}).dropna(),
      f"Years with positive sharpe: {(y > 0).sum()} of {y.notna().sum()}. First ~1.5 years = warm-up, no positions.")

# D7 macro
mb = macro.reindex(px.index, method="ffill")
rows = []
for pl_ in PLANETS:
    for k in ("gdp", "cpi"):
        mm = mb[f"{k}|{pl_}"].pct_change(60)
        for asset in (f"{pl_}_bonds", f"{pl_}_stocks", f"{pl_}_titan_fx"):
            if asset not in px:
                continue
            fwd = px[asset].diff(60).shift(-60)  # next 60 days, only used to test, never in the model
            same = px[asset].diff(60)
            idx = slice(None, None, 60)        # non-overlapping
            rows.append({"macro": f"{k} {pl_}", "asset": asset,
                         "corr with next 60d": mm.iloc[idx].corr(fwd.iloc[idx]),
                         "corr same 60d": mm.iloc[idx].corr(same.iloc[idx])})
mt = pd.DataFrame(rows)
table("D7. Macro: does 60d GDP/CPI change predict the planet's assets?", mt,
      "~90 non-overlapping points each, so |corr| < ~0.2 is noise. 'same 60d' = moves together (not a forecast).")

# D8 OHLC sanity + range vol
rows = []
for a in px.columns:
    o, h, l, c_ = (ohlc[f"{a}|{k}"] for k in ("Open", "High", "Low", "Close"))
    bad = ((h < l) | (c_ > h + 1e-12) | (c_ < l - 1e-12)).mean()
    pk = np.sqrt(((h - l) ** 2).mean() / (4 * np.log(2)))  # Parkinson-style, in price units
    rows.append({"asset": a, "bad bars %": 100 * bad, "close = ohlc close": bool(np.allclose(c_, px[a], equal_nan=True)),
                 "range vol / close vol": pk / dP[a].std()})
table("D8. OHLC check", pd.DataFrame(rows),
      "range vol / close vol: Parkinson-style vol from high-low vs close-to-close vol. A steady ratio means it would barely change relative sizing.")

hdr = ("# Data analysis\n\nLynx in-sample data. Generated by data_analysis.py, do not edit by hand.\n"
       "Rerun: `python research/jaakko/pitch/data_analysis.py`\n")
(HERE / "DATA.md").write_text(hdr + "\n".join(out) + "\n", encoding="utf-8")
print(hdr + "\n".join(out))
