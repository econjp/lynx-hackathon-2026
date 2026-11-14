"""Audit of the final model against what Lynx says it grades on. Writes CRITERIA_AUDIT.md.

    python research/eemil/final_combined/criteria_audit.py        (about 1.5 minutes)

Lynx's instructions (lynx_materials/lynx-hackathon-instructions.pdf) name these:
  1. net adjusted Sharpe, out of sample          5. understandable, explainable mechanism
  2. no forward-looking bias (disqualification)  6. simple and properly fine-tuned, not overly complicated
  3. risk spread across assets                   7. costs: watch net vs gross Sharpe and holding period
  4. total risk stable over time
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import polars as pl

import final_validation as fv
fm, va, ev = fv.fm, fv.va, fv.ev
DATE, START, END, ERAS = fv.DATE, fv.START, fv.d[-1], fv.ERAS
FINAL = fv.POS[fv.FINAL]
SLEEVES = fv.final_sleeves
out = []
md = lambda title, df, note="": out.append(f"\n## {title}\n" + (f"{note}\n\n" if note else "\n") + fv.to_markdown(df))


def kf(pos, prices=fv.prices, a=START, b=END, costs=-0.02, keys=("adjusted_sharpe",)):
    return ev.calc_key_figures(pos.filter(pl.col(DATE).is_between(a, b)), prices, costs=costs, key_figures=list(keys))


def eras(pos, prices=fv.prices):
    return {"full": kf(pos, prices)["adjusted_sharpe"], **{e: kf(pos, prices, a, b)["adjusted_sharpe"] for e, (a, b) in ERAS.items()}}


# ---------------------------------------------------------------- 1. performance and costs
f = kf(FINAL, keys=["adjusted_sharpe", "sharpe", "sharpe_ex_slippage", "sharpe_1d_lag", "skewness", "kurtosis", "av_holding", "position_bias", "returns"])
r = f["returns"].to_pandas().set_index(DATE)["returns"].iloc[3:]
yearly = r.groupby(r.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(260))
t1 = pd.DataFrame([
    {"figure": "net adjusted Sharpe (the score)", "value": f["adjusted_sharpe"]},
    {"figure": "net Sharpe", "value": f["sharpe"]},
    {"figure": "gross Sharpe (before costs)", "value": f["sharpe_ex_slippage"]},
    {"figure": "lost to costs (gross minus net)", "value": f["sharpe_ex_slippage"] - f["sharpe"]},
    {"figure": "lost to the skew and kurtosis penalty (net minus adjusted)", "value": f["sharpe"] - f["adjusted_sharpe"]},
    {"figure": "Sharpe with one extra day of lag", "value": f["sharpe_1d_lag"]},
    {"figure": "average holding, days", "value": f["av_holding"]},
    {"figure": "skewness", "value": f["skewness"]},
    {"figure": "kurtosis (normal = 3)", "value": f["kurtosis"]},
    {"figure": "position bias (0 = balanced)", "value": f["position_bias"]},
    {"figure": f"calendar years with positive Sharpe, of {len(yearly)}", "value": float((yearly > 0).sum())},
    {"figure": "worst calendar year, Sharpe", "value": yearly.min()},
])
md("1. Score, costs and shape of returns", t1, f"Final model ({fv.FINAL}), scored from {START}.")
ty = pd.DataFrame({"year": yearly.index.astype(str), "sharpe": yearly.values})
md("1b. Sharpe by calendar year", ty)

# ---------------------------------------------------------------- 3. risk spread across assets
P = FINAL.to_pandas().set_index(DATE)
P.index = pd.to_datetime(P.index)
dp = fv.PX.diff()
apnl = (P.shift(2) * dp).loc[pd.Timestamp(START):].iloc[3:]
tot = apnl.sum(axis=1)
share = apnl.apply(lambda x: x.cov(tot)) / tot.var()
cls = pd.Series({a: fm.asset_class(a) for a in P.columns})
bias = kf(FINAL, keys=["position_bias_per_instrument"])["position_bias_per_instrument"]
gross = (P.abs() * fv.VOL).loc[pd.Timestamp(START):]
t3 = pd.DataFrame([{"class": k, "assets": int((cls == k).sum()), "share of P&L variance": share[cls == k].sum(),
                    "share if spread evenly": (cls == k).mean(), "share of P&L": apnl.loc[:, cls == k].sum().sum() / tot.sum(),
                    "average position bias": float(np.mean([bias[a] for a in cls.index[cls == k]]))} for k in ("bonds", "stocks", "fx", "commodity")])
md("3. Where the risk sits, by asset class", t3,
   "Share of P&L variance = how much of the day-to-day swings come from that class. Lynx asks that risk is not concentrated.")
top = share.sort_values(ascending=False)
t3b = pd.DataFrame([{"figure": "largest single-asset share of P&L variance", "value": f"{top.index[0]}: {top.iloc[0]:.2f}"},
                    {"figure": "share of the top 5 assets", "value": f"{top.iloc[:5].sum():.2f}"},
                    {"figure": "share if spread evenly, 1 asset / 5 assets", "value": f"{1 / len(top):.2f} / {5 / len(top):.2f}"},
                    {"figure": "assets with positive P&L", "value": f"{int((apnl.sum() > 0).sum())} of {len(top)}"}])
md("3b. Concentration", t3b)

# share of the final model's P&L variance explained by each sleeve run alone
R = fv.R
X = R[["trend only", "inflation only", "twins only"]]
beta = np.linalg.lstsq(np.c_[np.ones(len(X)), X.values], R[fv.FINAL].values, rcond=None)[0][1:]
vs = [b * np.cov(X[c], R[fv.FINAL])[0, 1] / R[fv.FINAL].var() for b, c in zip(beta, X.columns)]
t3c = pd.DataFrame({"sleeve": [c.replace(" only", "") for c in X.columns], "share of final P&L variance": vs,
                    "target (equal budget)": 1 / len(SLEEVES)})
md("3c. Risk by sleeve", t3c, "From a regression of the final model's daily P&L on the three sleeves run alone.")

# ---------------------------------------------------------------- 4. total risk stable over time
def risk_path(pos):
    s = kf(pos, keys=["rolling_std"])["rolling_std"].to_pandas().set_index(DATE)["rolling_std"].dropna()
    return s / s.median()


rows = []
for name, pos in [("final model", FINAL), ("Lynx 50d baseline", fv.POS["Lynx 50d baseline"])]:
    s = risk_path(pos)
    rows.append({"model": name, "lowest": s.min(), "5% low": s.quantile(0.05), "95% high": s.quantile(0.95), "highest": s.max()})
t4 = pd.DataFrame(rows)
md("4. Is total risk stable over time?", t4, "Rolling 1-year standard deviation of daily P&L, divided by its median. 1.00 = normal risk level.")

# ---------------------------------------------------------------- 2. forward-looking bias, two tests
def model(p, m):
    return fm.final_positions(p, m)


def perturb_test(cuts=(0.3, 0.5, 0.7, 0.9), seed=0):
    """Lynx's first method: two data sets identical up to T, different after. Positions up to T must match."""
    rng = np.random.default_rng(seed)
    full = model(fv.prices, fv.macro).drop(DATE).to_numpy()
    worst = 0.0
    for c in cuts:
        n = int(len(fv.prices) * c)
        cut = fv.prices[DATE][n - 1]
        px = fv.prices.to_pandas().set_index(DATE)
        px.iloc[n:] = px.iloc[n:] * np.exp(rng.normal(0, 0.05, px.iloc[n:].shape)).cumprod(axis=0)
        mc = fv.macro.to_pandas().set_index(DATE)
        k = int((mc.index <= pd.Timestamp(cut)).sum())
        mc.iloc[k:] = mc.iloc[k:] * np.exp(rng.normal(0, 0.01, mc.iloc[k:].shape)).cumprod(axis=0)
        to_pl = lambda d: pl.from_pandas(d.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))
        alt = model(to_pl(px), to_pl(mc)).drop(DATE).to_numpy()
        worst = max(worst, np.abs(full[:n] - alt[:n]).max() / max(np.abs(full[:n]).max(), 1e-12))
    return worst


w = perturb_test()
t2 = pd.DataFrame([
    {"test": "shorter history (cut at 30/50/70/90/97%), positions must match the full run", "result": fv.leak["result"].iloc[0],
     "largest relative difference": fv.leak["largest relative difference"].iloc[0]},
    {"test": "future data replaced by random data after 30/50/70/90%, positions before must match", "result": "PASS" if w < 1e-9 else "FAIL",
     "largest relative difference": f"{w:.1e}"}])
md("2. Forward-looking bias", t2, "Both methods from Lynx's instructions, on prices and macro together.")

# ---------------------------------------------------------------- 6. fine-tuned or fragile? one setting changed at a time
def variant(**kw):
    """Rebuild the final model with module settings changed. Returns positions."""
    alpha = kw.pop("ALPHA", fm.ALPHA)
    old = {k: getattr(fm, k) for k in kw}
    for k, v in kw.items():
        setattr(fm, k, v)
    try:
        fc, px, vol = fm.sleeve_forecasts(fv.prices, fv.macro)
        return fm.combine(fc, px, vol, SLEEVES, True, alpha)
    finally:
        for k, v in old.items():
            setattr(fm, k, v)


rows = [{"setting": "default (as submitted)", **eras(FINAL)}]
GRID = [("glide 10% per day", {"ALPHA": 0.1}), ("glide 100% (no smoothing)", {"ALPHA": 1.0}),
        ("sleeve risk window 125d", {"SLEEVE_WIN": 125}), ("sleeve risk window 500d", {"SLEEVE_WIN": 500}),
        ("bond cap 1.5 bets", {"BOND_BETS": 1.5}), ("bond cap 3 bets", {"BOND_BETS": 3.0}), ("bond cap off", {"BOND_BETS": 99.0}),
        ("CPI normal window 2 years", {"MACRO_WIN": 500}), ("CPI normal window 4 years", {"MACRO_WIN": 1000}),
        ("twin correlation line 0.75", {"TWIN_CORR": 0.75}), ("twin correlation line 0.85", {"TWIN_CORR": 0.85}),
        ("twin spread window 125d", {"SPREAD_WIN": 125, "SPREAD_MIN": 60}), ("twin spread window 500d", {"SPREAD_WIN": 500, "SPREAD_MIN": 250})]
for label, kw in GRID:
    rows.append({"setting": label, **eras(variant(**kw))})
t6 = pd.DataFrame(rows)
md("6. One setting changed at a time", t6,
   "None of these was used to choose anything. The submitted settings are the first row. A defendable model changes little here.")

# twins: which pairs
pairs = fm.twin_pairs(fv.PX)
t6p = pd.DataFrame([{"pair": f"{a} / {b}", "share of days the rule has it on": float(on.loc[pd.Timestamp(START):].mean())} for (a, b), on in pairs.items()])
md("6b. Twin pairs found by the rule", t6p)


def twins_variant(pick):
    fc = dict(fv.FC)
    fc["twins"] = fm.twins_forecast(fv.PX, pick)
    return fm.combine(fc, fv.PX, fv.VOL, SLEEVES, True)


always = pd.Series(True, index=fv.PX.index)
AA = ("antimatter", "aerogel")
commod = {k: v for k, v in pairs.items() if fm.asset_class(k[0]) == "commodity"}
rows = [{"twins sleeve": "rule as submitted", **eras(FINAL)},
        {"twins sleeve": "antimatter / aerogel never traded", **eras(twins_variant({k: v for k, v in pairs.items() if k != AA}))},
        {"twins sleeve": "antimatter / aerogel always on", **eras(twins_variant({**pairs, AA: always}))},
        {"twins sleeve": "the three commodity pairs always on", **eras(twins_variant({k: always for k in commod}))}]
for k in commod:
    rows.append({"twins sleeve": f"only {k[0]} / {k[1]} (rule timing)", **eras(twins_variant({k: pairs[k]}))})
t6t = pd.DataFrame(rows)
md("6c. Does the result hang on one twin pair?", t6t, "Final model with the twins sleeve changed; trend and inflation unchanged.")

# sleeve budgets
def budget_stats(pos):
    Pb = pos.to_pandas().set_index(DATE)
    Pb.index = pd.to_datetime(Pb.index)
    ap = (Pb.shift(2) * dp).loc[pd.Timestamp(START):].iloc[3:]
    sh = (ap.apply(lambda x: x.cov(ap.sum(axis=1))) / ap.sum(axis=1).var()).sort_values(ascending=False)
    k = kf(pos, keys=["kurtosis", "position_bias"])
    return {"top 5 assets, share of variance": sh.iloc[:5].sum(), "kurt": k["kurtosis"], "bias": k["position_bias"]}


rows = []
for label, sl, b in [("equal risk per sleeve (as submitted)", SLEEVES, "equal"),
                     ("risk in proportion to the number of assets a sleeve trades", SLEEVES, "breadth"),
                     ("trend + inflation, equal", ("trend", "inflation"), "equal"),
                     ("trend + inflation, by number of assets", ("trend", "inflation"), "breadth")]:
    pb = fm.combine(fv.FC, fv.PX, fv.VOL, sl, True, budget=b)
    rows.append({"sleeve budget": label, **eras(pb), **budget_stats(pb)})
t6e = pd.DataFrame(rows)
md("6e. How much risk each sleeve gets", t6e,
   "Equal risk per sleeve gives the twins sleeve (2 or 3 pairs) as much risk as trend (27 assets). The alternative rule "
   "follows Lynx's own text: total risk 'contributed to in equal parts from all assets'.")

# leave one asset out
rows = []
for a in fv.PX.columns:
    p = fv.prices.drop(a)
    rows.append({"asset left out": a, **eras(fm.final_positions(p, fv.macro), p)})
loo = pd.DataFrame(rows).sort_values("full")
t6l = pd.concat([loo.head(4), loo.tail(3)])
md("6d. Leave one asset out", t6l,
   f"The model is rebuilt 27 times, each time without one asset. Shown: the 4 that hurt most and the 3 that help most to remove. "
   f"Range of the full score: {loo['full'].min():.2f} to {loo['full'].max():.2f}.")

hdr = ("# Final model vs Lynx's grading criteria\n\nGenerated by research/eemil/final_combined/criteria_audit.py, do not edit by hand.\n"
       "Section numbers follow the criteria listed at the top of the script.\n")
(HERE / "CRITERIA_AUDIT.md").write_text(hdr + "\n".join(out) + "\n", encoding="utf-8")

if __name__ == "__main__":
    pd.set_option("display.width", 260, "display.float_format", "{:.2f}".format, "display.max_colwidth", 90)
    for t in (t1, ty.T, t3, t3b, t3c, t4, t2, t6, t6p, t6t, t6e, t6l):
        print(t.to_string(index=False), "\n")
