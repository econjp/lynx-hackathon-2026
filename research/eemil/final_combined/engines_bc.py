"""Robustness of Engine B (Kristofer's stretch stop) and Engine C (sector-neutral 12m cross-sectional
momentum), alone and combined with the other sleeves. Writes ENGINES_BC.md and engines_bc.png.

    python research/eemil/final_combined/engines_bc.py        (about 40 seconds)

Same pipeline as the final model for every row: each sleeve is a forecast, sleeves get equal risk
budgets, one portfolio, Jaakko's sizing, Lynx evaluator. Nothing is tuned here.
"""
import sys
import importlib.machinery
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

import final_validation as fv            # data, eras, scoring helpers, the three existing sleeves
fm, va, ev = fv.fm, fv.va, fv.ev
from signals import clean_prices, scale_and_cap      # Jaakko
from risk import price_vol                           # Jaakko

DATE, ROOT = fv.DATE, fv.ROOT
ERAS, START, END = fv.ERAS, fv.START, fv.d[-1]
out = []
md = lambda title, df, note="": out.append(f"\n## {title}\n" + (f"{note}\n\n" if note else "\n") + fv.to_markdown(df))

# Kristofer's code, imported unchanged
_loader = importlib.machinery.SourceFileLoader("kristofer_stop", str(ROOT / "research" / "kristofer" / "Kood mida vaadata"))
ks = importlib.util.module_from_spec(importlib.util.spec_from_loader("kristofer_stop", _loader))
_argv, sys.argv = sys.argv, sys.argv[:1]
_loader.exec_module(ks)
sys.argv = _argv
SB = [a for a in fv.PX.columns if a.endswith(ks.ASSET_SUFFIXES)]
CLASSES = ("bonds", "stocks", "fx", "commodity")


# ---------------------------------------------------------------- sleeves
def engine_b(px, **params):
    """Kristofer: 1 when long, 0 when flat, on the 10 stocks and bonds. params override his settings."""
    old = {k: getattr(ks, k) for k in params}
    for k, v in params.items():
        setattr(ks, k, v)
    try:
        sig = ks.build_signals(px[SB])
    finally:
        for k, v in old.items():
            setattr(ks, k, v)
    return sig.reindex(columns=px.columns).fillna(0.0)


def momentum(px, n):
    vol = price_vol(px)
    return scale_and_cap(clean_prices(px, vol).diff().rolling(n).sum() / (vol * np.sqrt(n)))


def engine_c(px, n=250, by_class=True, centred=False):
    """Rank on n-day momentum, -1 (weakest) .. +1 (strongest), inside each asset class.
    The team formula 2 * (pct rank - 0.5) is not centred: with 5 assets it gives -0.6 .. +1.0, an average
    of +0.2, so the sleeve is net long. centred=True subtracts the class average (true long/short)."""
    m = momentum(px, n)
    if centred:
        parts = [m[[a for a in px if fm.asset_class(a) == k]].rank(axis=1, pct=True) for k in CLASSES]
        return pd.concat([2 * p.sub(p.mean(axis=1), axis=0) for p in parts], axis=1)[px.columns].fillna(0.0)
    if not by_class:
        return (2 * (m.rank(axis=1, pct=True) - 0.5)).fillna(0.0)
    parts = [2 * (m[[a for a in px if fm.asset_class(a) == k]].rank(axis=1, pct=True) - 0.5) for k in CLASSES]
    return pd.concat(parts, axis=1)[px.columns].fillna(0.0)


def always_long(px):
    f = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    f[SB] = 1.0
    return f.where(price_vol(px).notna(), 0.0)


FC = dict(fv.FC_NOCAP)   # trend, inflation, twins (no twin-leg halving, so every sleeve here is raw)
FC.update({"B": engine_b(fv.PX), "C": engine_c(fv.PX), "C all 27": engine_c(fv.PX, by_class=False), "long": always_long(fv.PX),
           "C centred": engine_c(fv.PX, centred=True)})


def book(sleeves, caps=False, fc=FC, px=fv.PX, vol=fv.VOL):
    return fm.combine(fc, px, vol, tuple(sleeves), caps)


def kf(pos, prices=fv.prices, a=START, b=END, costs=-0.02, keys=("adjusted_sharpe",)):
    return ev.calc_key_figures(pos.filter(pl.col(DATE).is_between(a, b)), prices, costs=costs, key_figures=list(keys))


def row(name, pos, prices=fv.prices):
    f = kf(pos, prices, keys=fv.KEYS)
    r = {"model": name, "full": f["adjusted_sharpe"]}
    r.update({e: kf(pos, prices, a, b)["adjusted_sharpe"] for e, (a, b) in ERAS.items()})
    r.update({"sharpe": f["sharpe"], "sharpe 1d lag": f["sharpe_1d_lag"], "2x costs": kf(pos, prices, costs=-0.04)["adjusted_sharpe"],
              "skew": f["skewness"], "kurt": f["kurtosis"], "holding d": f["av_holding"], "bias": f["position_bias"]})
    return r


def pnl(pos, prices=fv.prices):
    return kf(pos, prices, keys=["returns"])["returns"].to_pandas().set_index(DATE)["returns"].iloc[3:]


# ---------------------------------------------------------------- 1. each engine alone
ALONE = {
    "B: Kristofer stop (10 stocks+bonds)": ["B"],
    "always long the same 10 (benchmark for B)": ["long"],
    "C: XS 12m within class": ["C"],
    "C centred (same ranks, class average removed)": ["C centred"],
    "XS 12m, all 27 ranked together": ["C all 27"],
    "A: trend (Jaakko pick)": ["trend"],
    "inflation": ["inflation"],
    "twins": ["twins"],
}
POS = {k: book(v) for k, v in ALONE.items()}
t1 = pd.DataFrame([row(k, p) for k, p in POS.items()])
md("1. Each engine alone", t1,
   f"Scored from {START}. Every row goes through the same sizing (1/vol, constant ex-ante book risk, 30% glide), so B here is "
   "smoother than Kristofer's own on/off backtest. 'bias' is the long tilt: 1 = always long, 0 = balanced.")

# ---------------------------------------------------------------- 2. combinations
COMBOS = {
    "B + C": ["B", "C"],
    "long + C (B replaced by plain holding)": ["long", "C"],
    "B + C centred": ["B", "C centred"],
    "B + C + inflation": ["B", "C", "inflation"],
    "B + C + inflation + twins": ["B", "C", "inflation", "twins"],
    "long + C + inflation + twins": ["long", "C", "inflation", "twins"],
    "C + inflation + twins (no long engine)": ["C", "inflation", "twins"],
    "A + inflation + twins (current final, no caps)": ["trend", "inflation", "twins"],
    "A + C + inflation + twins": ["trend", "C", "inflation", "twins"],
    "A + B + C + inflation + twins (all five)": ["trend", "B", "C", "inflation", "twins"],
}
POS.update({k: book(v) for k, v in COMBOS.items()})
POS["B + C + inflation + twins, with risk caps"] = book(COMBOS["B + C + inflation + twins"], caps=True)

# the team's Dual-Engine exactly as built in research/jaakko/pitch/team_checks.py:
# half of Kristofer's raw 1/vol positions plus half of the XS book (not equal risk)
kvol = np.sqrt((fv.PX.diff() ** 2).rolling(100).mean())
raw_b = (FC["B"] / kvol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
dual = 0.5 * raw_b + 0.5 * POS["C: XS 12m within class"].to_pandas().set_index(DATE).values
dual.index.name = DATE
POS["team Dual-Engine as built (50% raw B + 50% C book)"] = pl.from_pandas(dual.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))

t2 = pd.DataFrame([row(k, POS[k]) for k in [*COMBOS, "B + C + inflation + twins, with risk caps",
                                             "team Dual-Engine as built (50% raw B + 50% C book)"]])
md("2. Combinations, one portfolio, equal risk per sleeve", t2,
   "No risk caps unless the row says so. The 'long + ...' rows are the control: the same model with Kristofer's "
   "rule swapped for simply holding the same 10 assets. The gap between a B row and its 'long' row is what the stop rule itself adds.")

# ---------------------------------------------------------------- 3. is B more than holding?
R = pd.DataFrame({k: pnl(p) for k, p in POS.items()}).dropna()
rb, rl = R["B: Kristofer stop (10 stocks+bonds)"], R["always long the same 10 (benchmark for B)"]
beta = np.cov(rb, rl)[0, 1] / rl.var()
resid = rb - beta * rl
rows = [{"check": "correlation of B's daily P&L with plain holding", "value": rb.corr(rl)},
        {"check": "beta of B on plain holding", "value": beta},
        {"check": "Sharpe of what is left after removing holding", "value": resid.mean() / resid.std() * np.sqrt(260)},
        {"check": "t-stat of that leftover", "value": resid.mean() / resid.std() * np.sqrt(len(resid))}]
for e, (a, b) in ERAS.items():
    x = resid.loc[pd.Timestamp(a):pd.Timestamp(b)]
    rows.append({"check": f"leftover Sharpe {e}", "value": x.mean() / x.std() * np.sqrt(260)})
t3 = pd.DataFrame(rows)
md("3. Engine B vs plain holding", t3, "Regression of B's daily P&L on always-long P&L (same 10 assets, same sizing).")

# ---------------------------------------------------------------- 4. mirror world
# every price path flipped upside down: what rose now falls. A rule with a real, symmetric edge keeps
# working; a rule that lives on the 20-year rise of stocks and bonds does not.
mpx = 2 * fv.PX.iloc[0] - fv.PX
mprices = pl.from_pandas(mpx.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))
mvol = price_vol(mpx)
MFC = {"B": engine_b(mpx), "C": engine_c(mpx), "C centred": engine_c(mpx, centred=True), "long": always_long(mpx),
       "trend": fm.trend_forecast(clean_prices(mpx, mvol), mvol).fillna(0.0)}
rows = []
for name, sl in [("B: Kristofer stop", ["B"]), ("always long the same 10", ["long"]), ("C: XS 12m within class", ["C"]),
                 ("C centred", ["C centred"]), ("A: trend", ["trend"]), ("B + C", ["B", "C"]), ("A + C", ["trend", "C"])]:
    real = book(sl)
    mir = book(sl, fc=MFC, px=mpx, vol=mvol)
    rows.append({"model": name, "real data": kf(real)["adjusted_sharpe"], "mirrored data": kf(mir, mprices)["adjusted_sharpe"],
                 "bias real": kf(real, keys=["position_bias"])["position_bias"],
                 "bias mirrored": kf(mir, mprices, keys=["position_bias"])["position_bias"]})
t4 = pd.DataFrame(rows)
md("4. Mirror world: all price paths flipped upside down", t4,
   "Prices are replaced by 2 x first price minus price, so every rise becomes a fall of the same size and the "
   "day-to-day pattern is unchanged. Long/short rules should score about the same. A long-only rule cannot.")

# ---------------------------------------------------------------- 5. parameter neighbours
rows = []
for label, params in [("default: EMA 10/100, gap 8", {}), ("gap 6", {"GAP_THRESHOLD": 6.0}), ("gap 10", {"GAP_THRESHOLD": 10.0}),
                      ("no stop", {"USE_STOP": False}), ("EMA 20/100", {"FAST_EMA": 20}), ("EMA 10/200", {"SLOW_EMA": 200}),
                      ("EMA 5/50", {"FAST_EMA": 5, "SLOW_EMA": 50})]:
    fc = dict(FC, B=engine_b(fv.PX, **params))
    for name, sl in [("B alone", ["B"]), ("B + C", ["B", "C"])]:
        p = book(sl, fc=fc)
        rows.append({"engine B setting": label, "model": name, "full": kf(p)["adjusted_sharpe"],
                     **{e: kf(p, a=a, b=b)["adjusted_sharpe"] for e, (a, b) in ERAS.items()}})
t5b = pd.DataFrame(rows)
md("5a. Engine B: neighbouring settings", t5b, "If the score only holds at one setting, the setting was fitted.")

rows = []
for n in (60, 120, 190, 250, 310, 375):
    fc = dict(FC, C=engine_c(fv.PX, n))
    for name, sl in [("C alone", ["C"]), ("B + C", ["B", "C"])]:
        p = book(sl, fc=fc)
        rows.append({"lookback d": n, "model": name, "full": kf(p)["adjusted_sharpe"],
                     **{e: kf(p, a=a, b=b)["adjusted_sharpe"] for e, (a, b) in ERAS.items()}})
t5c = pd.DataFrame(rows)
md("5b. Engine C: neighbouring lookbacks", t5c)

# ---------------------------------------------------------------- 6. leak test
def model(sleeves):
    def f(p):
        px = fm._to_pandas(p)
        fc = {"B": engine_b(px), "C": engine_c(px)}
        return fm.combine(fc, px, price_vol(px), tuple(sleeves), False)
    return f


t6 = pd.DataFrame([{"model": n, "result": "PASS" if ok else "FAIL", "largest relative difference": f"{w:.1e}"}
                   for n, (ok, w) in [(n, va.leak_test(model(s), fv.prices)) for n, s in [("B", ["B"]), ("C", ["C"]), ("B + C", ["B", "C"])]]])
md("6. Leak test", t6, "Positions recomputed on histories cut at 30 / 50 / 70 / 90 / 97% must equal the full run.")

# ---------------------------------------------------------------- 7. bootstrap
idx = va.boot_idx(len(R), n_boot=2000)
BS = {}
bs = lambda k: BS.setdefault(k, va.boot_scores(R[k].values, idx))
q = lambda x: "{:.2f} to {:.2f}".format(*np.nanpercentile(x, [5, 95]))
rows = []
for a, b in [("B + C", "long + C (B replaced by plain holding)"),
             ("B + C + inflation + twins", "long + C + inflation + twins"),
             ("B + C + inflation + twins", "A + inflation + twins (current final, no caps)"),
             ("A + C + inflation + twins", "A + inflation + twins (current final, no caps)"),
             ("C + inflation + twins (no long engine)", "A + inflation + twins (current final, no caps)")]:
    rows.append({"model": a, "against": b, "model score": va.adj_sharpe(R[a].values), "against score": va.adj_sharpe(R[b].values),
                 "difference": va.adj_sharpe(R[a].values) - va.adj_sharpe(R[b].values),
                 "90% interval of difference": q(bs(a) - bs(b)), "P(model wins)": float(np.nanmean(bs(a) > bs(b)))})
t7 = pd.DataFrame(rows)
md("7. Block bootstrap of the differences", t7, "2000 paired resamples of 60-day blocks of daily P&L.")

# ---------------------------------------------------------------- 8. correlations
names = {"B: Kristofer stop (10 stocks+bonds)": "B", "always long the same 10 (benchmark for B)": "long", "C: XS 12m within class": "C",
         "A: trend (Jaakko pick)": "A trend", "inflation": "inflation", "twins": "twins"}
c = R[list(names)].rename(columns=names).corr().round(2)
c.insert(0, "daily P&L", c.index)
md("8. P&L correlation between engines", c)

hdr = ("# Engines B and C: performance and robustness\n\nB = Kristofer's EMA 10/100 long-only stretch stop on the 10 stocks and bonds. "
       "C = 12-month cross-sectional momentum ranked inside each asset class.\n"
       "Generated by research/eemil/final_combined/engines_bc.py, do not edit by hand.\n\n![engines](engines_bc.png)\n")
(HERE / "ENGINES_BC.md").write_text(hdr + "\n".join(out) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- chart
lines = [("B + C", "#2a78d6", 1.6), ("long + C (B replaced by plain holding)", "#9ec5f4", 1.6),
         ("B + C + inflation + twins", "#eb6834", 1.6), ("C + inflation + twins (no long engine)", "#1baf7a", 1.6),
         ("A + inflation + twins (current final, no caps)", "#0b0b0b", 2.2)]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.6, "font.size": 9, "legend.frameon": False})
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=140)
for k, col, lw in lines:
    ax.plot(R.index, (R[k] / R[k].std()).cumsum(), color=col, lw=lw, label=k)
ax.set_title("Cumulative P&L after costs, each line scaled to the same daily risk", loc="left", fontweight="bold")
ax.set_ylabel("daily standard deviations")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(HERE / "engines_bc.png")
plt.close(fig)

if __name__ == "__main__":
    pd.set_option("display.width", 260, "display.float_format", "{:.2f}".format, "display.max_colwidth", 60)
    for t in (t1, t2, t3, t4, t5b, t5c, t6, t7, c):
        print(t.to_string(index=False), "\n")
