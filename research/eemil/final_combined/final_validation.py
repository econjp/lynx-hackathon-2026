"""Validation of the final model (src/final_model.py). Writes FINAL_RESULTS.md and final_sleeves.png.

    python research/eemil/final_combined/final_validation.py        (about 10 seconds)

Every number in FINAL_RESULTS.md comes from this script. Scored with lynx_materials/evaluation.py.
"""
import sys
import importlib.machinery
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "lynx_materials"), str(HERE.parent)]

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluation as ev
import validate as va                                   # research/eemil/validate.py
from src import final_model as fm

DATE = "AsOfDate"
prices = pl.read_csv(ROOT / "data" / "close_prices_insample.csv", try_parse_dates=True).with_columns(pl.col(DATE).cast(pl.Date))
macro = pl.read_csv(ROOT / "data" / "macro_data_insample.csv", try_parse_dates=True).with_columns(pl.col(DATE).cast(pl.Date))
d = prices[DATE]
n = len(d)
KEYS = ["adjusted_sharpe", "sharpe", "sharpe_1d_lag", "skewness", "kurtosis", "av_holding", "position_bias"]
out = []


def to_markdown(df):
    fmt = lambda v: f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1]]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def md(title, df, note=""):
    out.append(f"\n## {title}\n" + (f"{note}\n\n" if note else "\n") + to_markdown(df))


# ---------------------------------------------------------------- models
FC, PX, VOL = fm.sleeve_forecasts(prices, macro)
FC_NOCAP, _, _ = fm.sleeve_forecasts(prices, macro, caps=False)


def book(sleeves, caps=True):
    return fm.combine(FC if caps else FC_NOCAP, PX, VOL, sleeves, caps)


def kristofer_overlay(pos):
    """Kristofer's stretch stop as an overlay: on stocks and bonds, when his rule is stopped out
    (EMA 10/100 trend up, but his position is flat), a long position in the final model is cut to 0.
    Logic imported unchanged from research/kristofer."""
    path = str(ROOT / "research" / "kristofer" / "Kood mida vaadata")
    loader = importlib.machinery.SourceFileLoader("kristofer_stop", path)
    spec = importlib.util.spec_from_loader("kristofer_stop", loader)
    ks = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        loader.exec_module(ks)
    finally:
        sys.argv = argv
    cols = [c for c in PX.columns if c.endswith(ks.ASSET_SUFFIXES)]
    sig = ks.build_signals(PX[cols])
    up = PX[cols].ewm(span=ks.FAST_EMA, adjust=False).mean() > PX[cols].ewm(span=ks.SLOW_EMA, adjust=False).mean()
    stopped = (up & (sig == 0)).values
    p = pos.to_pandas().set_index(DATE)
    p[cols] = p[cols].where(~(stopped & (p[cols].values > 0)), 0.0)
    return pl.from_pandas(p.reset_index()).with_columns(pl.col(DATE).cast(pl.Date)), float(stopped.mean())


TI, TIT = ("trend", "inflation"), ("trend", "inflation", "twins")
POS = {
    "Lynx 50d baseline": va.registry()["lynx50"](prices).fill_null(0.0).fill_nan(0.0),
    "trend only": book(("trend",)),
    "inflation only": book(("inflation",)),
    "twins only": book(("twins",)),
    "trend + inflation": book(TI),
    "trend + inflation + twins": book(TIT),
}

# common scoring window: from the first day every sleeve holds a position
def first_day(pos):
    live = pos.select(pl.sum_horizontal(pl.exclude(DATE).abs()) > 0).to_series()
    return d.filter(live)[0]


first_live = {k: first_day(POS[k]) for k in ("trend only", "inflation only", "twins only")}
START = max(first_live.values())
E1, E2 = d[n // 3], d[2 * n // 3]
ERAS = {f"{START.year}-{E1.year}": (START, E1), f"{E1.year}-{E2.year}": (E1, E2), f"{E2.year}-{d[-1].year}": (E2, d[-1])}
SELECT = (START, E2)      # anything chosen is chosen here; the last era is only a check


def kf(pos, a=START, b=d[-1], costs=-0.02, keys=("adjusted_sharpe",)):
    return ev.calc_key_figures(pos.filter(pl.col(DATE).is_between(a, b)), prices, costs=costs, key_figures=list(keys))


def score(pos, a=START, b=d[-1]):
    return kf(pos, a, b)["adjusted_sharpe"]


def pnl(pos):
    r = kf(pos, keys=["returns"])["returns"].to_pandas().set_index(DATE)["returns"]
    return r.iloc[fm_lag:]


fm_lag = ev.POSITION_LAG + 1

# twins are optional: keep them only if they help on the selection window (1995-2009)
sel_ti, sel_tit = score(POS["trend + inflation"], *SELECT), score(POS["trend + inflation + twins"], *SELECT)
FINAL = "trend + inflation + twins" if sel_tit > sel_ti else "trend + inflation"
final_sleeves = TIT if FINAL.endswith("twins") else TI
assert tuple(fm.FINAL_SLEEVES) == final_sleeves, f"set FINAL_SLEEVES = {final_sleeves} in src/final_model.py"
POS["final, no risk caps"] = book(final_sleeves, caps=False)
POS["Jaakko pick as is (trend, no caps)"] = book(("trend",), caps=False)
try:
    POS["final + Kristofer stop overlay"], stopped_share = kristofer_overlay(POS[FINAL])
except Exception as e:  # optional row
    stopped_share = None
    print("Kristofer overlay skipped:", type(e).__name__, e)


# ---------------------------------------------------------------- main table
def row(name, pos):
    f = kf(pos, keys=KEYS)
    r = {"model": name + (" (FINAL)" if name == FINAL else ""), "full": f["adjusted_sharpe"]}
    r.update({e: score(pos, a, b) for e, (a, b) in ERAS.items()})
    r.update({"sharpe": f["sharpe"], "sharpe 1d lag": f["sharpe_1d_lag"], "2x costs": kf(pos, costs=-0.04)["adjusted_sharpe"],
              "skew": f["skewness"], "kurt": f["kurtosis"], "holding d": f["av_holding"], "bias": f["position_bias"]})
    return r


order = ["Lynx 50d baseline", "trend only", "inflation only", "trend + inflation", "trend + inflation + twins",
         "final + Kristofer stop overlay", "twins only", "final, no risk caps", "Jaakko pick as is (trend, no caps)"]
main = pd.DataFrame([row(k, POS[k]) for k in order if k in POS])
md("1. Models (adjusted Sharpe unless the column says otherwise)", main,
   f"All rows are scored from {START}, the first day every sleeve holds a position, so the eras compare like with like. "
   f"The first era is therefore {START} to {E1}. 'full', the three eras and '2x costs' are adjusted Sharpe. "
   "All rows except the last two use the same risk caps (twin legs share one budget, bond block capped at "
   f"{fm.BOND_BETS:g} bets, ultrapure_silicon half weight). The last three rows are for reference only.")

sel = pd.DataFrame([{"model": "trend + inflation", f"{START.year}-{E2.year}": sel_ti},
                    {"model": "trend + inflation + twins", f"{START.year}-{E2.year}": sel_tit}])
md("2. The one choice made on data: twins in or out", sel,
   f"Rule fixed up front: keep the twins sleeve only if it raises adjusted Sharpe on {START} to {E2}. "
   f"The last era is not used. Result: **{FINAL}** is the final model.")

# ---------------------------------------------------------------- sleeve correlations
R = pd.DataFrame({k: pnl(POS[k]) for k in ["trend only", "inflation only", "twins only", FINAL]}).dropna()
c = R.rename(columns={FINAL: "final model"}).corr().round(2)
c.insert(0, "daily P&L", c.index)
md("3. P&L correlation between sleeves", c, "Each sleeve run alone through the same sizing.")

# ---------------------------------------------------------------- inflation plateau
rows = []
for kb, ks in [(20, 120), (60, 120), (120, 120), (60, 250)]:
    fc, _, _ = fm.sleeve_forecasts(prices, macro, kb, ks)
    for label, sl in [("inflation only", ("inflation",)), ("trend + inflation", TI)]:
        p = fm.combine(fc, PX, VOL, sl)
        rows.append({"bonds k": kb, "stocks k": ks, "model": label, "default": "yes" if (kb, ks) == (fm.K_BONDS, fm.K_STOCKS) else "",
                     "full": score(p), **{e: score(p, a, b) for e, (a, b) in ERAS.items()}})
for label, sub in [("bonds rule alone, k=60", "bonds"), ("stocks rule alone, k=120", "stocks")]:
    f1 = FC["inflation"].copy()
    f1[[a for a in f1 if fm.asset_class(a) != sub]] = 0.0
    p = fm.combine({"inflation": f1}, PX, VOL, ("inflation",))
    rows.append({"bonds k": 60 if sub == "bonds" else "-", "stocks k": 120 if sub == "stocks" else "-", "model": label, "default": "",
                 "full": score(p), **{e: score(p, a, b) for e, (a, b) in ERAS.items()}})
plateau = pd.DataFrame(rows)
md("4. Inflation lookback plateau", plateau,
   "CPI lookback k for bonds in {20, 60, 120} and for stocks in {120, 250}. The defaults (60, 120) were fixed before "
   "this table was made and stay whatever it shows. The table only checks that the neighbours score about the same.")

# ---------------------------------------------------------------- leak test
def truncating(sleeves):
    def model(p):  # macro is cut at the same date as the prices
        return fm.final_positions(p, macro.filter(pl.col(DATE) <= p[DATE][-1]), sleeves=sleeves)
    return model


leak = []
for name, sl in [(FINAL + " (final)", final_sleeves), ("inflation only", ("inflation",))]:
    ok, worst = va.leak_test(truncating(sl), prices)
    leak.append({"model": name, "result": "PASS" if ok else "FAIL", "largest relative difference": f"{worst:.1e}"})
leak = pd.DataFrame(leak)
md("5. Leak test", leak, "Positions recomputed on histories cut at 30 / 50 / 70 / 90 / 97% (prices and macro both cut) "
   "must equal the full run on every shared row.")

# ---------------------------------------------------------------- bootstrap
idx = va.boot_idx(len(R), n_boot=2000)
bf, bt = va.boot_scores(R[FINAL].values, idx), va.boot_scores(R["trend only"].values, idx)
q = lambda x: "{:.2f} to {:.2f}".format(*np.nanpercentile(x, [5, 95]))
boot = pd.DataFrame([
    {"quantity": "final model", "point": va.adj_sharpe(R[FINAL].values), "90% interval": q(bf)},
    {"quantity": "trend only", "point": va.adj_sharpe(R["trend only"].values), "90% interval": q(bt)},
    {"quantity": "final minus trend only", "point": va.adj_sharpe(R[FINAL].values) - va.adj_sharpe(R["trend only"].values), "90% interval": q(bf - bt)},
])
p_beat = float(np.nanmean(bf > bt))
md("6. Block bootstrap: final model vs trend only", boot,
   f"2000 paired resamples of 60-day blocks of daily P&L. P(final beats trend only) = **{p_beat:.2f}**.")

# ---------------------------------------------------------------- write
hdr = (f"# Final model results\n\nData: close_prices_insample.csv + macro_data_insample.csv, {prices.width - 1} assets, {d[0]} to {d[-1]}. "
       "Score: Lynx evaluation.py (2-day lag, 0.02 sigma costs).\nGenerated by research/eemil/final_combined/final_validation.py, do not edit by hand.\n\n"
       "![sleeves](final_sleeves.png)\n")
(HERE / "FINAL_RESULTS.md").write_text(hdr + "\n".join(out) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- chart
C = {"trend only": "#2a78d6", "inflation only": "#eb6834", "twins only": "#1baf7a"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.6, "font.size": 9, "legend.frameon": False})
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=140)
for k, col in C.items():
    lab = k.replace(" only", "") + ("" if k.split()[0] in final_sleeves else " (not in final model)")
    ax.plot(R.index, (R[k] / R[k].std()).cumsum(), color=col, lw=1.3, label=lab)
ax.plot(R.index, (R[FINAL] / R[FINAL].std()).cumsum(), color="#0b0b0b", lw=2.2, label=f"final model: {FINAL}")
ax.axvline(pd.Timestamp(E2), color="#52514e", lw=0.8, ls="--")
ax.text(pd.Timestamp(E2), ax.get_ylim()[1], " check period (not used for choices)", va="top", fontsize=8, color="#52514e")
ax.set_title("Cumulative P&L after costs, each line scaled to the same daily risk", loc="left", fontweight="bold")
ax.set_ylabel("daily standard deviations")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(HERE / "final_sleeves.png")
plt.close(fig)

if __name__ == "__main__":
  pd.set_option("display.width", 250, "display.float_format", "{:.2f}".format)
  for t in (main, sel, c, plateau, leak, boot):
      print(t.to_string(index=False), "\n")
  print(f"final = {FINAL}; scored from {START}; P(final > trend) = {p_beat:.2f}; Kristofer stopped share = {stopped_share}")
