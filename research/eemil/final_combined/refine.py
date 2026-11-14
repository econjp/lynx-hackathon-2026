"""Refinement tests: fewer parts, and the questions from the team. Writes REFINE.md.

    python research/eemil/final_combined/refine.py        (about 40 seconds)

Q1. Does the model get worse if the risk caps are removed? (fewer parts)
Q2. Kristofer's EMA rule on all 27 assets, long AND short: is it a better trend engine than Jaakko's pick?
Q3. A deliberate, stated long position in stocks and bonds that inflation switches down: worth it?
Every variant goes through the same pipeline and evaluator. Choices are judged on 1997-2009 only.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import polars as pl

import final_validation as fv
import engines_bc as eb                  # Kristofer's code (eb.ks), helpers
from src import goat_model as gm

fm, va, ev = fv.fm, fv.va, fv.ev
DATE, START, END, ERAS, E2 = fv.DATE, fv.START, fv.d[-1], fv.ERAS, fv.E2
PX, VOL = fv.PX, fv.VOL
RAW = fv.FC_NOCAP                        # trend, inflation, twins forecasts with no caps
SB = [a for a in PX.columns if fm.asset_class(a) in ("stocks", "bonds")]
out = []
md = lambda title, df, note="": out.append(f"\n## {title}\n" + (f"{note}\n\n" if note else "\n") + fv.to_markdown(df))


def kf(pos, a=START, b=END, keys=("adjusted_sharpe",)):
    return ev.calc_key_figures(pos.filter(pl.col(DATE).is_between(a, b)), fv.prices, key_figures=list(keys))


def row(name, pos):
    f = kf(pos, keys=["adjusted_sharpe", "sharpe", "kurtosis", "av_holding", "position_bias"])
    return {"model": name, "full": f["adjusted_sharpe"], f"chosen on: {START.year}-{E2.year}": kf(pos, START, E2)["adjusted_sharpe"],
            **{e: kf(pos, a, b)["adjusted_sharpe"] for e, (a, b) in ERAS.items()},
            "sharpe": f["sharpe"], "kurt": f["kurtosis"], "holding d": f["av_holding"], "bias": f["position_bias"]}


def book(fc, sleeves, caps=False):
    return fm.combine(fc, PX, VOL, tuple(sleeves), caps)


# ---------------------------------------------------------------- trend engines, all long/short on all 27 assets
def kristofer_both_ways(px, stop=True):
    """Kristofer's rule made symmetric: his long rule on the price, and the same rule on the price turned
    upside down for the short side. +1 long, -1 short, 0 flat (stopped out)."""
    old = eb.ks.USE_STOP
    eb.ks.USE_STOP = stop
    try:
        return eb.ks.build_signals(px) - eb.ks.build_signals(-px)
    finally:
        eb.ks.USE_STOP = old


TREND = {
    "Jaakko pick (EWMAC 32/128 + 64/256 + 12m momentum)": RAW["trend"],
    "EMA 10/100, long/short, all 27": kristofer_both_ways(PX, stop=False),
    "EMA 10/100 + Kristofer stop, long/short, all 27": kristofer_both_ways(PX),
}
t1 = pd.DataFrame([row(k, book({"trend": f}, ["trend"])) for k, f in TREND.items()]
                  + [row("for reference: Kristofer stop, long only, 10 stocks+bonds", book(eb.FC, ["B"]))])
md("Q2. Trend engines alone, long and short on all 27 assets", t1)

rows = []
for k, f in TREND.items():
    rows.append(row(f"{k.split(' (')[0]} + inflation + twins", book({**RAW, "trend": f}, fv.TIT)))
t2 = pd.DataFrame(rows)
md("Q2b. The same engines inside the full model (with inflation and twins, no caps)", t2)

# ---------------------------------------------------------------- Q1 fewer parts
V2 = book(RAW, fv.TIT)
t3 = pd.DataFrame([
    row("v1: with 3 risk caps (was submitted as 'final')", fv.POS[fv.FINAL]),
    row("v2: same 3 sleeves, no caps", V2),
    row("v2 without twins", book(RAW, fv.TI)),
    row("v2 without inflation", book(RAW, ("trend", "twins"))),
    row("v2 without trend", book(RAW, ("inflation", "twins"))),
    row("GOAT (as submitted)", gm.goat_positions(fv.prices)),
])
md("Q1. Fewer parts", t3, "v2 = three forecasts, equal risk each, summed, then 1/vol, constant book risk, glide. Nothing else.")

# ---------------------------------------------------------------- Q3 stated long position that inflation switches down
def with_premium(c):
    """Stocks and bonds: forecast = c - tanh(inflation z). c = 0 is the plain inflation rule.
    With c > 0 the model holds stocks and bonds by default and cuts or reverses when inflation runs hot."""
    f = RAW["inflation"].copy()
    live = f[SB].abs() > 0
    f[SB] = (f[SB] + c).where(live, 0.0)
    return {**RAW, "inflation": f}


rows = []
for c in (0.0, 0.25, 0.5, 1.0):
    rows.append(row(f"v2, default long = {c:g}", book(with_premium(c), fv.TIT)))
t4 = pd.DataFrame(rows)
md("Q3. A stated long position in stocks and bonds, switched down by inflation", t4,
   "'default long' is how long the macro sleeve is in stocks and bonds when inflation is at its normal level "
   "(1 = a full position). Inflation one unit above normal takes about 0.76 off. Watch the bias column: this is "
   "a market bet, sized on purpose.")

# what the macro sleeve does to equity exposure around the worst stock months
st = [a for a in PX.columns if fm.asset_class(a) == "stocks"]
eq = (PX[st].diff() / VOL[st].shift(1)).mean(axis=1)            # average stock move in vol units
worst = eq.rolling(60).sum().loc[pd.Timestamp(START):]
bad = worst < worst.quantile(0.10)
sig = RAW["inflation"][st].mean(axis=1).shift(62).loc[worst.index]   # the stock forecast in place before each 60-day window
t5 = pd.DataFrame([{"60-day windows": "worst 10% for stocks", "average macro stock forecast going in": sig[bad].mean()},
                   {"60-day windows": "all other", "average macro stock forecast going in": sig[~bad].mean()}])
md("Q3b. Does the inflation rule cut stock exposure before bad stock periods?", t5,
   "Negative = the rule was short stocks going into the window. If it works as a drawdown signal, the first row is clearly more negative.")

hdr = "# Refinement tests\n\nGenerated by research/eemil/final_combined/refine.py, do not edit by hand.\n"
(HERE / "REFINE.md").write_text(hdr + "\n".join(out) + "\n", encoding="utf-8")

if __name__ == "__main__":
    pd.set_option("display.width", 280, "display.float_format", "{:.2f}".format, "display.max_colwidth", 70)
    for t in (t1, t2, t3, t4, t5):
        print(t.to_string(index=False), "\n")
