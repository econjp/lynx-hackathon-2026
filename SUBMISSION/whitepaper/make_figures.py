"""Numbers and figures for the white paper.   python SUBMISSION/whitepaper/make_figures.py

Uses SUBMISSION/model.py unchanged and Lynx's evaluation.py. Writes fig_*.png and numbers.json next to this file.
The trend / inflation / twins sensitivity heatmaps come from the BOMBASTRAT sweeps (WORK_DUMP history, sweeps.json).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "SUBMISSION"), str(ROOT / "lynx_materials")]

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import evaluation as ev
import model as M

DATE = M.DATE
prices = M.load([ROOT / "data" / "close_prices_insample.csv"])
macro = M.load([ROOT / "data" / "macro_data_insample.csv"])
px = M._frame(prices)
START = pd.Timestamp("1997-05-22")
ERAS = [("1997-2002", "1997-05-22", "2002-05-01"), ("2002-2009", "2002-05-01", "2009-05-01"), ("2009-2016", "2009-05-01", "2016-12-31")]


def pdf(p):
    d = p.to_pandas().set_index(DATE)
    d.index = pd.to_datetime(d.index)
    return d


def to_pl(d):
    d = d.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d.index.name = DATE
    return pl.from_pandas(d.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))


B = pdf(M.bombastrat_positions(prices, macro))
G = M.exante_scale(pdf(M.goat_positions(prices)), px)
G.index = B.index
BOOKS = {"BOMBASTRAT": B, "GOAT": G}
for w in (0.2, 0.33, 0.5, 0.67, 0.8):
    BOOKS[f"{int(round(w * 100))}% GOAT"] = w * G + (1 - w) * B
BOOKS["Combined (50/50)"] = BOOKS.pop("50% GOAT")


def kf(d, a=START, b="2016-12-31", keys=("adjusted_sharpe",)):
    p = to_pl(d).filter(pl.col(DATE).is_between(pd.Timestamp(a).date(), pd.Timestamp(b).date()))
    return ev.calc_key_figures(p, prices, key_figures=list(keys))


def returns(d):
    r = kf(d, keys=["returns"])["returns"].to_pandas().set_index(DATE)["returns"]
    r.index = pd.to_datetime(r.index)
    return r.iloc[3:]


N = {}
for name, d in BOOKS.items():
    k = kf(d, keys=["adjusted_sharpe", "sharpe", "sharpe_ex_slippage", "sharpe_1d_lag", "skewness", "kurtosis", "av_holding", "position_bias"])
    k.update({e: kf(d, a, b)["adjusted_sharpe"] for e, a, b in ERAS})
    r = returns(d)
    adj = lambda x: ev._adjusted_sharpe(pl.Series(x))
    win = [adj(r.values[i - 756:i]) for i in range(756, len(r) + 1, 5)]
    k["worst_3y"], k["share_3y_above_0.5"] = float(np.min(win)), float(np.mean(np.array(win) > 0.5))
    yr = r.groupby(r.index.year).sum()
    k["positive_years"], k["years"] = int((yr > 0).sum()), int(len(yr))
    N[name] = {a: (round(float(v), 3) if isinstance(v, (float, np.floating, int)) else v) for a, v in k.items()}
    print(name, {a: N[name][a] for a in ("adjusted_sharpe", "1997-2002", "2002-2009", "2009-2016", "position_bias", "worst_3y")}, flush=True)

R = pd.DataFrame({n: returns(BOOKS[n]) for n in ("BOMBASTRAT", "GOAT", "Combined (50/50)")})
N["corr_bomba_goat"] = round(float(R["BOMBASTRAT"].corr(R["GOAT"])), 3)
hold = pd.DataFrame(0.0, index=px.index, columns=px.columns)
sb = [c for c in px if c.endswith(("_bonds", "_stocks"))]
hold[sb] = 1 / M.price_vol(px)[sb]
H = returns(M.exante_scale(hold, px))
N["corr_with_holding"] = {n: round(float(R[n].corr(H.reindex(R.index))), 3) for n in R}
(HERE / "numbers.json").write_text(json.dumps(N, indent=1))

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
COL = {"BOMBASTRAT": "#2a78d6", "GOAT": "#eb6834", "Combined (50/50)": "#0b0b0b"}
plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9a9993", "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED})

# cumulative P&L at equal volatility
fig, ax = plt.subplots(figsize=(8, 3.6))
for n in R:
    c = (R[n] / (R[n].std() * np.sqrt(260)) * 0.10).cumsum() * 100
    ax.plot(c.index, c, color=COL[n], lw=2.0 if n.startswith("Comb") else 1.3, label=n)
ax.set_ylabel("cumulative return, % (10% annual vol)")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, loc="upper left")
ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout(); fig.savefig(HERE / "fig_equity.png", dpi=220); plt.close(fig)

# sub-period scores
names = ["BOMBASTRAT", "GOAT", "Combined (50/50)"]
eras = [e for e, _, _ in ERAS]
fig, ax = plt.subplots(figsize=(8, 3.2)); x = np.arange(len(eras)); w = 0.26
for i, n in enumerate(names):
    v = [N[n][e] for e in eras]
    ax.bar(x + (i - 1) * w, v, w * 0.92, color=COL[n], label=n)
    for xi, y in zip(x, v):
        ax.text(xi + (i - 1) * w, y + 0.03, f"{y:.2f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(x, eras); ax.set_ylabel("adjusted Sharpe"); ax.set_ylim(0, 1.65)
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, ncol=3, loc="upper right"); ax.tick_params(axis="x", length=0)
fig.tight_layout(); fig.savefig(HERE / "fig_eras.png", dpi=220); plt.close(fig)

# weight sensitivity
ws = [("BOMBASTRAT", 0), ("20% GOAT", 20), ("33% GOAT", 33), ("Combined (50/50)", 50), ("67% GOAT", 67), ("80% GOAT", 80), ("GOAT", 100)]
fig, ax = plt.subplots(figsize=(8, 3.0))
ax.axvspan(33, 67, color="#cfe9d9", lw=0)
ax.plot([p for _, p in ws], [N[n]["adjusted_sharpe"] for n, _ in ws], color=INK, lw=1.8, marker="o", label="full period")
ax.plot([p for _, p in ws], [N[n]["worst_3y"] for n, _ in ws], color=MUTED, lw=1.3, marker="s", ls="--", label="worst 3-year window")
for n, p in ws:
    ax.text(p, N[n]["adjusted_sharpe"] + 0.05, f"{N[n]['adjusted_sharpe']:.2f}", ha="center", fontsize=8.5)
ax.set_xlabel("weight on GOAT, % (rest BOMBASTRAT)"); ax.set_ylabel("adjusted Sharpe")
ax.axhline(0, color="#9a9993", lw=0.8); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62))
fig.tight_layout(); fig.savefig(HERE / "fig_weights.png", dpi=220); plt.close(fig)
print("corr", N["corr_bomba_goat"], N["corr_with_holding"])
