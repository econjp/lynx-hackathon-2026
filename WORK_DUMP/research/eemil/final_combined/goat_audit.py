"""Audit of the GOAT model (src/goat_model.py) with the same tests as the trend + inflation + twins model.
Writes GOAT_AUDIT.md and goat_audit.png.

    python research/eemil/final_combined/goat_audit.py        (about 1.5 minutes)

The GOAT code is not changed. `goat()` below is the same model with its settings exposed, built from
the functions in src/goat_model.py, and checked to give identical positions at the default settings.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

import final_validation as fv
from src import goat_model as gm

fm, va, ev = fv.fm, fv.va, fv.ev
DATE, START, END, ERAS = fv.DATE, fv.START, fv.d[-1], fv.ERAS
out = []
md = lambda title, df, note="": out.append(f"\n## {title}\n" + (f"{note}\n\n" if note else "\n") + fv.to_markdown(df))


def goat(prices, thr=7.0, w_b=0.60, alpha=0.25, band=0.10, look=252, engine_b="stop", centred=False):
    """engine_b: 'stop' (Kristofer), 'ema' (EMA 10/100 long or flat, no stop), 'hold' (always long)."""
    px = prices.to_pandas().drop(columns=[DATE])
    cols = list(px.columns)
    groups = [[c for c in cols if gm.get_sector(c) == s] for s in ("stocks", "bonds", "fx", "commodity")]
    sb = groups[0] + groups[1]
    dp = px.diff()
    vol = np.sqrt((dp ** 2).rolling(100, min_periods=20).mean()).bfill()
    ef, es = px.ewm(span=10, adjust=False).mean(), px.ewm(span=100, adjust=False).mean()
    gap, trend = ((ef - es) / vol).fillna(0.0), (ef > es).astype(int)
    if engine_b == "stop":
        b = pd.DataFrame({c: gm.stretch_stop_positions(px[c].values, gap[c].values, trend[c].values, thr=thr, up_moves=2) for c in sb}, index=px.index)
    elif engine_b == "ema":
        b = trend[sb].astype(float)
    else:
        b = pd.DataFrame(1.0, index=px.index, columns=sb)
    units_b = (b / vol[sb]).reindex(columns=cols).fillna(0.0).to_numpy()
    ret = px.pct_change(look).fillna(0.0)
    parts = []
    for g in groups:
        if g:
            r = ret[g].rank(axis=1) / len(g)
            parts.append(2.0 * (r.sub(r.mean(axis=1), axis=0) if centred else r - 0.5))
    units_c = (pd.concat(parts, axis=1)[cols] / vol).fillna(0.0).to_numpy()
    mat = gm.apply_deadband(gm.apply_glide(w_b * units_b + (1 - w_b) * units_c, alpha=alpha), thr=band)
    return pl.DataFrame({DATE: prices[DATE], **{c: mat[:, j] for j, c in enumerate(cols)}})


assert np.abs(goat(fv.prices).drop(DATE).to_numpy() - gm.goat_positions(fv.prices).drop(DATE).to_numpy()).max() == 0.0


def kf(pos, prices=fv.prices, a=START, b=END, costs=-0.02, keys=("adjusted_sharpe",)):
    return ev.calc_key_figures(pos.filter(pl.col(DATE).is_between(a, b)), prices, costs=costs, key_figures=list(keys))


def eras(pos, prices=fv.prices):
    return {"full": kf(pos, prices)["adjusted_sharpe"], **{e: kf(pos, prices, a, b)["adjusted_sharpe"] for e, (a, b) in ERAS.items()}}


def row(name, pos, prices=fv.prices):
    f = kf(pos, prices, keys=fv.KEYS)
    return {"model": name, **eras(pos, prices), "sharpe": f["sharpe"], "sharpe 1d lag": f["sharpe_1d_lag"],
            "2x costs": kf(pos, prices, costs=-0.04)["adjusted_sharpe"], "skew": f["skewness"], "kurt": f["kurtosis"],
            "holding d": f["av_holding"], "bias": f["position_bias"]}


def pnl(pos, prices=fv.prices):
    return kf(pos, prices, keys=["returns"])["returns"].to_pandas().set_index(DATE)["returns"].iloc[3:]


POS = {
    "GOAT (as submitted)": goat(fv.prices),
    "GOAT with the stop removed (EMA 10/100 long or flat)": goat(fv.prices, engine_b="ema"),
    "GOAT with Engine 1 replaced by always holding the 10": goat(fv.prices, engine_b="hold"),
    "GOAT Engine 1 alone (stop, 10 stocks+bonds)": goat(fv.prices, w_b=1.0),
    "always holding the 10 stocks+bonds, nothing else": goat(fv.prices, w_b=1.0, engine_b="hold"),
    "GOAT Engine 2 alone (XS momentum)": goat(fv.prices, w_b=0.0),
    "GOAT Engine 2 alone, ranks centred (true long/short)": goat(fv.prices, w_b=0.0, centred=True),
    "GOAT with Engine 2 centred": goat(fv.prices, centred=True),
    "trend + inflation + twins (Eemil model)": fv.POS[fv.FINAL],
}
t1 = pd.DataFrame([row(k, p) for k, p in POS.items()])
md("1. Scores, same window and evaluator for every row", t1,
   f"Scored from {START} like the other tables in this folder. 'bias' is the long tilt: 1 = always long, 0 = balanced.")
g_all = ev.calc_key_figures(POS["GOAT (as submitted)"], fv.prices)
t1b = pd.DataFrame([{"figure": k, "value": g_all[k]} for k in ("adjusted_sharpe", "sharpe", "sharpe_1d_lag", "kurtosis", "av_holding", "position_bias")])
md("1b. GOAT on the whole file (1995 to 2016), as in the README", t1b)

# ---------------------------------------------------------------- 2. how much is just holding
R = pd.DataFrame({k: pnl(p) for k, p in POS.items()}).dropna()
G, H = R["GOAT (as submitted)"], R["always holding the 10 stocks+bonds, nothing else"]
beta = np.cov(G, H)[0, 1] / H.var()
res = G - beta * H
rows = [{"check": "correlation of GOAT's daily P&L with plain holding of the 10", "value": G.corr(H)},
        {"check": "share of GOAT's P&L variance explained by plain holding (R2)", "value": G.corr(H) ** 2},
        {"check": "Sharpe of what is left after removing holding", "value": res.mean() / res.std() * np.sqrt(260)},
        {"check": "t-stat of that leftover", "value": res.mean() / res.std() * np.sqrt(len(res))}]
for e, (a, b) in ERAS.items():
    x = res.loc[pd.Timestamp(a):pd.Timestamp(b)]
    rows.append({"check": f"leftover Sharpe {e}", "value": x.mean() / x.std() * np.sqrt(260)})
P = POS["GOAT (as submitted)"].to_pandas().set_index(DATE)
P.index = pd.to_datetime(P.index)
ap = (P.shift(2) * fv.PX.diff()).loc[pd.Timestamp(START):].iloc[3:]
cls = pd.Series({a: fm.asset_class(a) for a in P.columns})
sb = cls.isin(["stocks", "bonds"])
rows.append({"check": "share of GOAT's total P&L earned on stocks and bonds", "value": ap.loc[:, sb].sum().sum() / ap.sum().sum()})
rows.append({"check": "share of GOAT's P&L variance from stocks and bonds", "value": ap.loc[:, sb].sum(axis=1).cov(ap.sum(axis=1)) / ap.sum(axis=1).var()})
t2 = pd.DataFrame(rows)
md("2. How much of GOAT is simply being long stocks and bonds?", t2)

# ---------------------------------------------------------------- 3. mirror world
# every price replaced by first price squared / price: each percentage rise becomes a fall, same day-to-day pattern
num = [c for c in fv.prices.columns if c != DATE]
mprices = fv.prices.with_columns([(pl.col(c).first() ** 2 / pl.col(c)).alias(c) for c in num])
jp = lambda p: fm.final_positions(p, fv.macro, sleeves=("trend",), caps=False)
rows = []
for name, fn in [("GOAT (as submitted)", goat), ("GOAT with Engine 1 replaced by always holding the 10", lambda p: goat(p, engine_b="hold")),
                 ("GOAT Engine 1 alone", lambda p: goat(p, w_b=1.0)), ("GOAT Engine 2 alone", lambda p: goat(p, w_b=0.0)),
                 ("Jaakko's trend pick (long/short, for comparison)", jp)]:
    rows.append({"model": name, "real data": kf(fn(fv.prices))["adjusted_sharpe"], "mirrored data": kf(fn(mprices), mprices)["adjusted_sharpe"],
                 "bias real": kf(fn(fv.prices), keys=["position_bias"])["position_bias"],
                 "bias mirrored": kf(fn(mprices), mprices, keys=["position_bias"])["position_bias"]})
t3 = pd.DataFrame(rows)
md("3. Mirror world: what if stocks and bonds had fallen instead of risen?", t3,
   "Each price is replaced by (first price)^2 / price, so every rise becomes a fall of the same size. A rule with a real "
   "edge in both directions keeps most of its score. A rule that lives on the market going up does not.")

# ---------------------------------------------------------------- 4. settings
rows = [{"setting": "as submitted: gap 7, 60/40, glide 25%, deadband 10%, 252d", **eras(POS["GOAT (as submitted)"])}]
GRID = [("gap 6", dict(thr=6.0)), ("gap 8 (Kristofer's original)", dict(thr=8.0)), ("gap 10", dict(thr=10.0)),
        ("blend 50/50", dict(w_b=0.5)), ("blend 70/30", dict(w_b=0.7)), ("blend 40/60", dict(w_b=0.4)),
        ("glide 10%", dict(alpha=0.10)), ("glide 100% (none)", dict(alpha=1.0)),
        ("deadband off", dict(band=0.0)), ("deadband 5%", dict(band=0.05)), ("deadband 20%", dict(band=0.20)),
        ("XS lookback 190d", dict(look=190)), ("XS lookback 310d", dict(look=310))]
for label, kw in GRID:
    rows.append({"setting": label, **eras(goat(fv.prices, **kw))})
t4 = pd.DataFrame(rows)
md("4. One setting changed at a time", t4)

# ---------------------------------------------------------------- 5. leak tests
def trunc_test(fn, cuts=(0.3, 0.5, 0.7, 0.9, 0.97)):
    return va.leak_test(fn, fv.prices, cuts)[1]


def perturb_test(fn, cuts=(0.3, 0.5, 0.7, 0.9), seed=0):
    rng = np.random.default_rng(seed)
    full = fn(fv.prices).drop(DATE).to_numpy()
    worst = 0.0
    for c in cuts:
        n = int(len(fv.prices) * c)
        px = fv.prices.to_pandas().set_index(DATE)
        px.iloc[n:] = px.iloc[n:] * np.exp(rng.normal(0, 0.05, px.iloc[n:].shape)).cumprod(axis=0)
        alt = fn(pl.from_pandas(px.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))).drop(DATE).to_numpy()
        worst = max(worst, np.abs(full[:n] - alt[:n]).max() / max(np.abs(full[:n]).max(), 1e-12))
    return worst


a, b = trunc_test(gm.goat_positions), perturb_test(gm.goat_positions)
short = gm.goat_positions(fv.prices.head(15)).drop(DATE).to_numpy()
early = np.abs(short - gm.goat_positions(fv.prices).head(15).drop(DATE).to_numpy()).max()
t5 = pd.DataFrame([{"test": "shorter history (cut at 30/50/70/90/97%)", "result": "PASS" if a < 1e-9 else "FAIL", "largest difference": f"{a:.1e}"},
                   {"test": "future replaced by random data after 30/50/70/90%", "result": "PASS" if b < 1e-9 else "FAIL", "largest difference": f"{b:.1e}"},
                   {"test": "first 15 days only vs full run (the vol is back-filled over the first 20 days)",
                    "result": "PASS" if early < 1e-9 else "FAIL", "largest difference": f"{early:.1e}"}])
md("5. Forward-looking bias", t5)

# ---------------------------------------------------------------- 6. risk
def risk_path(pos):
    s = kf(pos, keys=["rolling_std"])["rolling_std"].to_pandas().set_index(DATE)["rolling_std"].dropna()
    return s / s.median()


rows = []
for name in ("GOAT (as submitted)", "trend + inflation + twins (Eemil model)"):
    s = risk_path(POS[name])
    Pm = POS[name].to_pandas().set_index(DATE); Pm.index = pd.to_datetime(Pm.index)
    x = (Pm.shift(2) * fv.PX.diff()).loc[pd.Timestamp(START):].iloc[3:]
    sh = (x.apply(lambda c: c.cov(x.sum(axis=1))) / x.sum(axis=1).var())
    rows.append({"model": name, "risk, lowest year vs normal": s.min(), "risk, highest year vs normal": s.max(),
                 "variance share stocks+bonds (37% if even)": sh[sb.values].sum(), "variance share top 5 assets (19% if even)": sh.sort_values().iloc[-5:].sum(),
                 "assets with positive P&L": f"{int((x.sum() > 0).sum())} of 27", "worst calendar year Sharpe": R[name].groupby(R.index.year).apply(lambda v: v.mean() / v.std() * np.sqrt(260)).min()})
t6 = pd.DataFrame(rows)
md("6. Risk: stable over time, spread across assets?", t6, "Risk = rolling 1-year standard deviation of daily P&L divided by its median.")

# ---------------------------------------------------------------- 7. bootstrap
idx = va.boot_idx(len(R), n_boot=2000)
BS = {}
bs = lambda k: BS.setdefault(k, va.boot_scores(R[k].values, idx))
q = lambda x: "{:.2f} to {:.2f}".format(*np.nanpercentile(x, [5, 95]))
rows = []
for m, o in [("GOAT (as submitted)", "trend + inflation + twins (Eemil model)"),
             ("GOAT (as submitted)", "GOAT with Engine 1 replaced by always holding the 10"),
             ("GOAT (as submitted)", "GOAT with the stop removed (EMA 10/100 long or flat)")]:
    rows.append({"model": m, "against": o, "difference": va.adj_sharpe(R[m].values) - va.adj_sharpe(R[o].values),
                 "90% interval": q(bs(m) - bs(o)), "P(model wins)": float(np.nanmean(bs(m) > bs(o)))})
t7 = pd.DataFrame(rows)
md("7. Block bootstrap of the differences", t7, f"2000 paired resamples of 60-day blocks. P&L correlation GOAT vs Eemil model: {R['GOAT (as submitted)'].corr(R['trend + inflation + twins (Eemil model)']):.2f}.")

# 50/50 of the two models, each at the same risk
mix = R["GOAT (as submitted)"] / R["GOAT (as submitted)"].std() + R["trend + inflation + twins (Eemil model)"] / R["trend + inflation + twins (Eemil model)"].std()
t8 = pd.DataFrame([{"model": "50/50 of the two models' P&L, same risk each", "full": va.adj_sharpe(mix.values),
                    **{e: va.adj_sharpe(mix.loc[pd.Timestamp(a):pd.Timestamp(b)].values) for e, (a, b) in ERAS.items()}}])
md("8. For information: the two models together", t8, "A sum of the two daily P&Ls, not one portfolio. Shows whether they are different bets.")

hdr = ("# GOAT model audit\n\nGOAT = 60% Kristofer stretch stop on 10 stocks and bonds + 40% cross-sectional 12-month momentum, glide and deadband "
       "(src/goat_model.py).\nGenerated by research/eemil/final_combined/goat_audit.py, do not edit by hand.\n\n![goat](goat_audit.png)\n")
(HERE / "GOAT_AUDIT.md").write_text(hdr + "\n".join(out) + "\n", encoding="utf-8")

lines = [("GOAT (as submitted)", "#eb6834", 2.0), ("GOAT with Engine 1 replaced by always holding the 10", "#f4b79e", 1.5),
         ("always holding the 10 stocks+bonds, nothing else", "#b9b8b3", 1.5), ("trend + inflation + twins (Eemil model)", "#0b0b0b", 2.0)]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.6, "font.size": 9, "legend.frameon": False})
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=140)
for k, col, lw in lines:
    ax.plot(R.index, (R[k] / R[k].std()).cumsum(), color=col, lw=lw, label=k)
ax.set_title("Cumulative P&L after costs, each line scaled to the same daily risk", loc="left", fontweight="bold")
ax.set_ylabel("daily standard deviations"); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(HERE / "goat_audit.png"); plt.close(fig)

if __name__ == "__main__":
    pd.set_option("display.width", 280, "display.float_format", "{:.2f}".format, "display.max_colwidth", 95)
    for t in (t1, t1b, t2, t3, t4, t5, t6, t7, t8):
        print(t.to_string(index=False), "\n")
