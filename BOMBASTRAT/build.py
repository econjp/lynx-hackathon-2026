"""Builds BOMBASTRAT/README.md, the charts and positions_insample.csv. Every number in the README comes from here.

    python BOMBASTRAT/build.py        (about 1 minute)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "research" / "eemil" / "final_combined")]

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

import final_validation as fv            # data, eras, evaluator helpers (research/eemil/final_combined)
from BOMBASTRAT import bombastrat as bs
from src import goat_model as gm
import portfolio as jp_portfolio

ev, va = fv.ev, fv.va
DATE, START, END, ERAS, E2 = fv.DATE, fv.START, fv.d[-1], fv.ERAS, fv.E2
PRICES, MACRO, PX = fv.prices, fv.macro, fv.PX
ERA = list(ERAS)


def kf(pos, a=START, b=END, costs=-0.02, keys=("adjusted_sharpe",), prices=PRICES):
    return ev.calc_key_figures(pos.filter(pl.col(DATE).is_between(a, b)), prices, costs=costs, key_figures=list(keys))


def score(pos, a=START, b=END, prices=PRICES):
    return kf(pos, a, b, prices=prices)["adjusted_sharpe"]


def eras(pos, prices=PRICES):
    return [score(pos, prices=prices)] + [score(pos, a, b, prices) for a, b in ERAS.values()]


def pnl(pos):
    return kf(pos, keys=["returns"])["returns"].to_pandas().set_index(DATE)["returns"].iloc[3:]


def table(header, rows):
    fmt = lambda v: f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows])


# ---------------------------------------------------------------- models
F, _, VOL = bs.forecasts(PRICES, MACRO)
BOMBA = bs.positions(PRICES, MACRO)
BOMBA.write_csv(HERE / "positions_insample.csv")
POS = {
    "BOMBASTRAT": BOMBA,
    "trend": bs.positions(PRICES, MACRO, use=("trend",)),
    "inflation": bs.positions(PRICES, MACRO, use=("inflation",)),
    "twins": bs.positions(PRICES, MACRO, use=("twins",)),
    "GOAT": gm.goat_positions(PRICES),
    "Lynx 50d baseline": fv.POS["Lynx 50d baseline"],
}
R = pd.DataFrame({k: pnl(p) for k, p in POS.items()}).dropna()
K = {k: kf(p, keys=["adjusted_sharpe", "sharpe", "sharpe_ex_slippage", "sharpe_1d_lag", "skewness", "kurtosis", "av_holding", "position_bias"]) for k, p in POS.items()}
b = K["BOMBASTRAT"]
yearly = R["BOMBASTRAT"].groupby(R.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(260))

# drop one sleeve
drop = {s: eras(bs.positions(PRICES, MACRO, use=tuple(x for x in ("trend", "inflation", "twins") if x != s))) for s in ("trend", "inflation", "twins")}


# leak tests (both methods in Lynx's instructions), prices and macro together
def cut_model(p):
    return bs.positions(p, MACRO.filter(pl.col(DATE) <= p[DATE][-1]))


leak_cut = va.leak_test(cut_model, PRICES)[1]
rng = np.random.default_rng(0)
full = BOMBA.drop(DATE).to_numpy()
leak_rand = 0.0
for c in (0.3, 0.5, 0.7, 0.9):
    n = int(len(PRICES) * c)
    px = PRICES.to_pandas().set_index(DATE)
    px.iloc[n:] = px.iloc[n:] * np.exp(rng.normal(0, 0.05, px.iloc[n:].shape)).cumprod(axis=0)
    mc = MACRO.to_pandas().set_index(DATE)
    k = int((mc.index <= pd.Timestamp(PRICES[DATE][n - 1])).sum())
    mc.iloc[k:] = mc.iloc[k:] * np.exp(rng.normal(0, 0.01, mc.iloc[k:].shape)).cumprod(axis=0)
    back = lambda d: pl.from_pandas(d.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))
    alt = bs.positions(back(px), back(mc)).drop(DATE).to_numpy()
    leak_rand = max(leak_rand, np.abs(full[:n] - alt[:n]).max() / np.abs(full[:n]).max())
LEAK = "PASS" if max(leak_cut, leak_rand) < 1e-9 else "FAIL"


# risk: spread and stability
def risk_stats(pos):
    P = pos.to_pandas().set_index(DATE)
    P.index = pd.to_datetime(P.index)
    x = (P.shift(2) * PX.diff()).loc[pd.Timestamp(START):].iloc[3:]
    sh = x.apply(lambda c: c.cov(x.sum(axis=1))) / x.sum(axis=1).var()
    cls = pd.Series({a: fv.fm.asset_class(a) for a in P.columns})
    s = kf(pos, keys=["rolling_std"])["rolling_std"].to_pandas().set_index(DATE)["rolling_std"].dropna()
    s = s / s.median()
    return {"sb": sh[cls.isin(["stocks", "bonds"]).values].sum(), "top5": sh.sort_values().iloc[-5:].sum(),
            "pos_assets": int((x.sum() > 0).sum()), "lo": s.min(), "hi": s.max(), "path": s}


RS = {k: risk_stats(POS[k]) for k in ("BOMBASTRAT", "GOAT", "Lynx 50d baseline")}

# bootstrap
idx = va.boot_idx(len(R), n_boot=2000)
BS = {k: va.boot_scores(R[k].values, idx) for k in ("BOMBASTRAT", "trend", "GOAT")}
q = lambda x: "{:.2f} to {:.2f}".format(*np.nanpercentile(x, [5, 95]))
p_trend, p_goat = float(np.nanmean(BS["BOMBASTRAT"] > BS["trend"])), float(np.nanmean(BS["BOMBASTRAT"] > BS["GOAT"]))

# mirror world (every rise becomes a fall); only meaningful for price-only rules
num = [c for c in PRICES.columns if c != DATE]
MIR = PRICES.with_columns([(pl.col(c).first() ** 2 / pl.col(c)).alias(c) for c in num])
mir_trend = score(bs.positions(MIR, MACRO, use=("trend",)), prices=MIR)
mir_goat = score(gm.goat_positions(MIR), prices=MIR)

# leave beryllium out
p_nb = PRICES.drop("beryllium")
no_beryl = eras(bs.positions(p_nb, MACRO), p_nb)

# inflation lookback neighbours (defaults fixed up front)
old = dict(bs.CPI_DAYS)
PLAT = []
for kb, ks in [(20, 120), (60, 120), (120, 120), (60, 250)]:
    bs.CPI_DAYS.update(bonds=kb, stocks=ks)
    PLAT.append((kb, ks, "yes" if (kb, ks) == (old["bonds"], old["stocks"]) else "", *eras(bs.positions(PRICES, MACRO))))
bs.CPI_DAYS.update(old)


# ---------------------------------------------------------------- volatility scaling variants
def variant(vol_fn=None, book_fn=None, glide_alpha=None):
    saved = (bs.price_vol, bs.exante_scale, bs.GLIDE)
    bs.price_vol = vol_fn or saved[0]
    bs.exante_scale = book_fn or saved[1]
    bs.GLIDE = glide_alpha or saved[2]
    try:
        return bs.positions(PRICES, MACRO)
    finally:
        bs.price_vol, bs.exante_scale, bs.GLIDE = saved


def weekly_cov_scale(units, px, window=250, start=60):
    d5, x = (px.diff(5) / np.sqrt(5)).fillna(0.0).values, units.fillna(0.0).values
    o = np.zeros_like(x)
    for i in range(start, len(x)):
        var = x[i] @ np.cov(d5[max(5, i - window + 1): i + 1].T) @ x[i]
        if var > 0:
            o[i] = x[i] / np.sqrt(var)
    return pd.DataFrame(o, index=units.index, columns=units.columns)


def realised_scale(units, px, window=100):       # the method in Lynx's instructions: trailing std of the strategy's own P&L
    r = (units.shift(2) * px.diff()).sum(axis=1)
    return units.div(r.rolling(window, min_periods=20).std().replace(0, np.nan), axis=0).fillna(0.0)


VOLS = [
    ("as built: blended asset vol, book scaled to constant forecast risk (250d covariance)", {}),
    ("asset vol: Lynx's 100-day window", dict(vol_fn=lambda px: np.sqrt((px.diff() ** 2).rolling(100, min_periods=30).mean()).replace(0, np.nan))),
    ("asset vol: fast (EWMA 20 days)", dict(vol_fn=lambda px: np.sqrt((px.diff() ** 2).ewm(span=20, min_periods=20).mean()).replace(0, np.nan))),
    ("asset vol: slow (500-day window)", dict(vol_fn=lambda px: np.sqrt((px.diff() ** 2).rolling(500, min_periods=60).mean()).replace(0, np.nan))),
    ("book risk: none (1/vol only)", dict(book_fn=lambda u, px: u)),
    ("book risk: Lynx's method (trailing 100d P&L std)", dict(book_fn=realised_scale)),
    ("book risk: covariance over 125 days", dict(book_fn=lambda u, px: jp_portfolio.exante_scale(u, px, window=125))),
    ("book risk: covariance over 500 days", dict(book_fn=lambda u, px: jp_portfolio.exante_scale(u, px, window=500))),
    ("book risk: covariance of weekly changes (holiday-proof)", dict(book_fn=weekly_cov_scale)),
    ("glide 10% per day", dict(glide_alpha=0.1)),
    ("glide 100% (no smoothing)", dict(glide_alpha=1.0)),
]
VT = []
for label, kw in VOLS:
    p = variant(**kw)
    f = kf(p, keys=["kurtosis", "av_holding"])
    rs = risk_stats(p)
    VT.append((label, *eras(p), f["kurtosis"], f["av_holding"], f"{rs['lo']:.2f} to {rs['hi']:.2f}"))

# ---------------------------------------------------------------- charts
INK, GRID = "#0b0b0b", "#e4e3df"
COL = {"BOMBASTRAT": INK, "trend": "#2a78d6", "inflation": "#eb6834", "twins": "#1baf7a", "GOAT": "#6250d6", "Lynx 50d baseline": "#b9b8b3"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "font.size": 9.5, "legend.frameon": False, "axes.titlelocation": "left", "axes.titleweight": "bold"})

fig, ax = plt.subplots(figsize=(11, 4.8), dpi=140)
for k in ("Lynx 50d baseline", "trend", "inflation", "twins", "GOAT", "BOMBASTRAT"):
    ax.plot(R.index, (R[k] / R[k].std()).cumsum(), color=COL[k], lw=2.6 if k == "BOMBASTRAT" else 1.3,
            ls="--" if k == "GOAT" else "-", label={"trend": "trend alone", "inflation": "inflation alone", "twins": "twins alone"}.get(k, k))
ax.axvline(pd.Timestamp(E2), color="#52514e", lw=0.8, ls=":")
ax.text(pd.Timestamp(E2), ax.get_ylim()[1], " check period: not used for any choice", va="top", fontsize=8, color="#52514e")
ax.set_title("Cumulative P&L after costs, every line at the same daily risk"); ax.set_ylabel("daily standard deviations"); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(HERE / "pnl.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 3.6), dpi=140)
names = ["Lynx 50d baseline", "trend", "inflation", "twins", "GOAT", "BOMBASTRAT"]
E = {k: eras(POS[k]) for k in names}
w = 0.8 / len(names)
for i, k in enumerate(names):
    bars = ax.bar(np.arange(4) + i * w, E[k], w, color=COL[k], label=k)
    if k in ("BOMBASTRAT", "GOAT"):
        for x, v in zip(np.arange(4) + i * w, E[k]):
            ax.text(x, v + 0.03, f"{v:.2f}", ha="center", fontsize=7.5)
ax.set_xticks(np.arange(4) + 0.4 - w / 2, ["full"] + ERA); ax.axhline(0, color="#52514e", lw=0.8)
ax.set_title("Adjusted Sharpe by era"); ax.legend(ncol=6, fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(HERE / "eras.png"); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), dpi=140)
for k in ("GOAT", "BOMBASTRAT"):
    rb = kf(POS[k], keys=["rolling_position_bias"])["rolling_position_bias"].to_pandas().set_index(DATE)["rolling_position_bias"]
    axes[0].plot(pd.to_datetime(rb.index), rb, color=COL[k], lw=1.8, label=k)
    axes[1].plot(pd.to_datetime(RS[k]["path"].index), RS[k]["path"], color=COL[k], lw=1.8, label=k)
axes[1].plot(pd.to_datetime(RS["Lynx 50d baseline"]["path"].index), RS["Lynx 50d baseline"]["path"], color=COL["Lynx 50d baseline"], lw=1.3, label="Lynx 50d baseline")
axes[0].axhline(0, color="#52514e", lw=0.8); axes[0].set_ylim(-1, 1); axes[0].set_title("Long tilt over time (0 = market neutral, 1 = all long)"); axes[0].legend()
axes[1].axhline(1, color="#52514e", lw=0.8); axes[1].set_title("Risk level over time (1 = normal)"); axes[1].legend()
fig.tight_layout(); fig.savefig(HERE / "neutral_risk.png"); plt.close(fig)

# ---------------------------------------------------------------- README
g = K["GOAT"]
e_b, e_g, e_t, e_l = E["BOMBASTRAT"], E["GOAT"], E["trend"], E["Lynx 50d baseline"]
corr = R[["trend", "inflation", "twins"]].corr()
n_lines = sum(1 for l in (HERE / "bombastrat.py").read_text().splitlines() if l.strip() and not l.strip().startswith("#"))
sel = lambda pos: score(pos, START, E2)

README = f"""# BOMBASTRAT

**Three simple forecasts, one portfolio, no market bet.** Trend for what trends, inflation for what cares about inflation, twin spreads for what moves together.

| | adjusted Sharpe, full | {ERA[0]} | {ERA[1]} | {ERA[2]} | long tilt | leak test |
|---|---|---|---|---|---|---|
| **BOMBASTRAT** | **{e_b[0]:.2f}** | {e_b[1]:.2f} | {e_b[2]:.2f} | {e_b[3]:.2f} | {b['position_bias']:.2f} | {LEAK} |
| GOAT | {e_g[0]:.2f} | {e_g[1]:.2f} | {e_g[2]:.2f} | {e_g[3]:.2f} | {g['position_bias']:.2f} | |
| Trend alone | {e_t[0]:.2f} | {e_t[1]:.2f} | {e_t[2]:.2f} | {e_t[3]:.2f} | {K['trend']['position_bias']:.2f} | |
| Lynx 50d baseline | {e_l[0]:.2f} | {e_l[1]:.2f} | {e_l[2]:.2f} | {e_l[3]:.2f} | {K['Lynx 50d baseline']['position_bias']:.2f} | |

Lynx evaluator, 2-day lag, costs included. All rows are scored from {START}, the first day all three forecasts hold positions.
Everything on this page is generated by `build.py`. The whole model is `bombastrat.py` ({n_lines} lines of code).

![pnl](pnl.png)

## The 2-minute walkthrough

> **What it is.** We make three separate forecasts for each asset and add them up. Nothing else.
>
> **One: trend.** On all 27 assets. If the price has been rising over the last months to a year, we are long; if falling, short. Markets react slowly to news, so moves continue. This is Jaakko's slow trend signal.
>
> **Two: inflation.** On the five bonds and five stocks. Each moon has its own CPI. When a moon's inflation runs hotter than its own three-year normal, we short that moon's bonds and stocks. The story is the textbook one: inflation up, rates up, bond prices down, and stocks get discounted harder. It holds on all five moons.
>
> **Three: twins.** Some commodity pairs are the same thing in two forms, like ore and refined metal. They move together more than 80% of the time. When the gap between them gets unusually wide against its one-year average, we bet on it closing: long the cheap one, short the expensive one.
>
> **How they are put together.** Each forecast gets the same risk budget. We add them, divide by each asset's volatility so every asset carries similar risk, scale the whole book to a constant risk level, and move 30% of the way to the target each day to save trading costs.
>
> **Why three.** They are different bets. Their daily P&L correlations are {corr.loc['trend','inflation']:.2f}, {corr.loc['trend','twins']:.2f} and {corr.loc['inflation','twins']:.2f}. Each one alone scores {e_t[0]:.2f}, {E['inflation'][0]:.2f} and {E['twins'][0]:.2f}; together {e_b[0]:.2f}. Take any one away and the score drops.
>
> **What it is not.** It is not a bet that markets go up. The average long tilt is {b['position_bias']:.2f}. It is not fitted: every number in the code was fixed before we scored it, and the one choice we made on data used 1997 to 2009 only.
>
> **Where it is weak.** The last era, {ERA[2]}, is the softest at {e_b[3]:.2f}, and the twins part leans on one pair. We show both below.

## How the layers fit

```
Layer 1  SIGNALS      trend (27 assets)   inflation (10 assets)   twins (pairs by rule)
                           |                     |                      |
Layer 2  RISK BUDGET  each forecast divided by the volatility of its own P&L   -> equal risk each
                           +---------------------+----------------------+
                                                 |  add
Layer 3  SIZING       position = forecast / asset volatility                   -> similar risk per asset
                      whole book scaled to constant forecast risk              -> stable total risk
Layer 4  EXECUTION    move 30% toward target each day                          -> low costs, 2-day lag is harmless
```

No overlays, no caps, no stops, no deadband. Each layer does one job.

## Against Lynx's grading criteria

Quotes are from `lynx_materials/lynx-hackathon-instructions.pdf`.

| What Lynx says | What we do | Evidence |
|---|---|---|
| "evaluated based on the net Adjusted Sharpe Ratio" | Three uncorrelated forecasts in one book | **{e_b[0]:.2f}** full ({e_b[1]:.2f} / {e_b[2]:.2f} / {e_b[3]:.2f} by era). Lynx baseline {e_l[0]:.2f}. {int((yearly > 0).sum())} of {len(yearly)} calendar years positive |
| "penalty terms that disincentivize negative skewness and high kurtosis" | tanh on every forecast, constant book risk, glide | Skew {b['skewness']:.2f}, kurtosis {b['kurtosis']:.2f} (normal is 3). The penalty costs {b['sharpe'] - b['adjusted_sharpe']:.2f} |
| "forward-looking-bias ... leads to automatic disqualification" | Only trailing windows. No full-sample statistic anywhere | **{LEAK}** on both tests Lynx describes: shorter history, and future data replaced by random data. Difference {max(leak_cut, leak_rand):.0e} |
| "two-day lag between calculating a desired position and actually receiving a return" | Slow signals | One extra day of lag changes Sharpe from {b['sharpe']:.2f} to {b['sharpe_1d_lag']:.2f} |
| "total portfolio risk ... contributed to in equal parts from all assets" | position = forecast / volatility | Stocks and bonds carry {RS['BOMBASTRAT']['sb']:.0%} of the risk (37% if perfectly even; GOAT {RS['GOAT']['sb']:.0%}). {RS['BOMBASTRAT']['pos_assets']} of 27 assets make money |
| "total risk of your portfolio ... not change too drastically over time" | Book scaled to constant forecast risk every day | Yearly risk stays between {RS['BOMBASTRAT']['lo']:.2f}x and {RS['BOMBASTRAT']['hi']:.2f}x of normal (Lynx baseline up to {RS['Lynx 50d baseline']['hi']:.2f}x) |
| "trade-off between alpha decay and transaction costs" | 30% glide | Holding {b['av_holding']:.0f} days. Costs take {b['sharpe_ex_slippage'] - b['sharpe']:.2f} Sharpe ({b['sharpe_ex_slippage']:.2f} gross, {b['sharpe']:.2f} net). At double costs the score is {kf(BOMBA, costs=-0.04)['adjusted_sharpe']:.2f} |
| "a mechanism that would plausibly work in practice" | Slow reaction (trend), inflation and rates (macro), same goods in two forms (twins) | Long tilt {b['position_bias']:.2f}: none of the score comes from markets going up. With every price path flipped upside down, our trend part scores {mir_trend:.2f} (real data {e_t[0]:.2f}); GOAT scores {mir_goat:.2f} (real data {e_g[0]:.2f}) |
| "no information about the period it will be judged on" | Parameters fixed up front. One choice on data (twins in or out), made on {START.year}-{E2.year} only | Neighbouring settings score about the same (tables below). The last simplification (removing three risk caps) changed the {START.year}-{E2.year} score by {sel(BOMBA) - sel(fv.POS[fv.FINAL]):+.2f}: done for simplicity, not for score |
| "Trend following, reversion and the use of macroeconomic data are all viable" | One simple rule from each family | Remove trend: {drop['trend'][0]:.2f}. Remove inflation: {drop['inflation'][0]:.2f}. Remove twins: {drop['twins'][0]:.2f}. All three earn their place |
| "better off developing a simple strategy and properly fine-tuning it" | Three rules, four layers, no overlays | `bombastrat.py`: {n_lines} lines of code, a handful of fixed numbers, all at the top of the file |
| "you will be asked to explain your strategy, how you arrived at it and the code" | One file, one function per forecast | The walkthrough above. History of what we tested and dropped: `research/eemil/final_combined/` |
| "We reward performance, not originality" | Textbook signals | Beats trend alone in {p_trend:.0%} of 2000 bootstrap resamples (difference {e_b[0] - e_t[0]:.2f}, 90% interval {q(BS['BOMBASTRAT'] - BS['trend'])}) |

![eras](eras.png)

## BOMBASTRAT or GOAT?

| | BOMBASTRAT | GOAT |
|---|---|---|
| Adjusted Sharpe, full | {e_b[0]:.2f} | {e_g[0]:.2f} |
| By era ({' / '.join(ERA)}) | {e_b[1]:.2f} / {e_b[2]:.2f} / {e_b[3]:.2f} | {e_g[1]:.2f} / {e_g[2]:.2f} / {e_g[3]:.2f} |
| On {START.year}-{E2.year} (the period we allow ourselves to choose on) | {sel(BOMBA):.2f} | {sel(POS['GOAT']):.2f} |
| Long tilt (0 = neutral) | {b['position_bias']:.2f} | {g['position_bias']:.2f} |
| Share of risk in stocks and bonds | {RS['BOMBASTRAT']['sb']:.0%} | {RS['GOAT']['sb']:.0%} |
| Kurtosis | {b['kurtosis']:.2f} | {g['kurtosis']:.2f} |
| Holding, days | {b['av_holding']:.0f} | {g['av_holding']:.0f} |
| Score if every price path is flipped upside down | trend part: {mir_trend:.2f} (the macro part cannot be flipped) | {mir_goat:.2f} |
| Uses the macro data | yes | no |
| P(BOMBASTRAT scores higher), bootstrap | {p_goat:.0%} | |

**Honest reading.** On score alone the two cannot be told apart (90% interval of the difference: {q(BS['BOMBASTRAT'] - BS['GOAT'])}). GOAT is steadier in {ERA[2]} because it is long stocks and bonds, which kept rising. BOMBASTRAT reaches a similar full-period score with no such bet, and scores higher on the two earlier eras. If the out-of-sample period has falling stocks or bonds, BOMBASTRAT is the safer submission; if they keep rising, GOAT is.

![neutral](neutral_risk.png)

## Volatility scaling and leverage

**Leverage does not change the score.** Sharpe is return divided by risk, so doubling every position doubles both. What matters is that the risk level stays even, because swings in risk create fat tails, and fat tails are penalised.

We tested whether a different volatility estimate or risk target gains anything. It does not: no version beats the first row, and most land within 0.03 of it, so we keep it and do not tune it.

{table(["version", "full"] + ERA + ["kurtosis", "holding d", "yearly risk range"], VT)}

## Is the inflation lookback a lucky pick?

{table(["bonds, days", "stocks, days", "as built", "full"] + ERA, PLAT)}

## Known weak spots

1. **{ERA[2]} is the softest era** ({e_b[3]:.2f}). Every part is weaker there, not one in particular.
2. **The twins part leans on one pair.** Without beryllium the model scores {no_beryl[0]:.2f} ({no_beryl[1]:.2f} / {no_beryl[2]:.2f} / {no_beryl[3]:.2f}). The five most important assets carry {RS['BOMBASTRAT']['top5']:.0%} of the risk (19% if perfectly even).
3. **The gain over trend alone is likely, not certain**: 90% interval {q(BS['BOMBASTRAT'] - BS['trend'])}.
4. **We saw all three eras while building it.** The parameters were set up front and the choices were judged on {START.year}-{E2.year}, but the team has looked at {ERA[2]} many times. Only the out-of-sample period is a clean test.

## Run it

```bash
python BOMBASTRAT/bombastrat.py --prices data/close_prices_insample.csv <out_of_sample_prices.csv> \\
                                --macro  data/macro_data_insample.csv  <out_of_sample_macro.csv> \\
                                --out    BOMBASTRAT/positions.csv
python BOMBASTRAT/build.py      # rebuilds this page, the charts and positions_insample.csv
```

| file | what |
|---|---|
| `bombastrat.py` | the model |
| `build.py` | every number and chart on this page |
| `positions_insample.csv` | positions for the full in-sample period, Lynx format |
"""
(HERE / "README.md").write_text(README, encoding="utf-8")
print(README[:1500])
print(table(["version", "full"] + ERA + ["kurtosis", "holding d", "yearly risk range"], VT))
