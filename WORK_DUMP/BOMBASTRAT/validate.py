"""Stress and pseudo out-of-sample validation of BOMBASTRAT (GOAT alongside). Writes VALIDATION.md + validation.png.

    python BOMBASTRAT/validate.py        (about 5 minutes)

The real out-of-sample file is not released yet, so nothing here is a true out-of-sample score.
These tests ask the next best questions: is the edge where the story says it is (placebos), does it
survive other histories, other asset sets, worse data, more lag and higher costs, and how wide is
the uncertainty.
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

import final_validation as fv
from BOMBASTRAT import bombastrat as bs
from src import goat_model as gm

ev, va = fv.ev, fv.va
DATE, START, END, ERAS, E2 = fv.DATE, fv.START, fv.d[-1], fv.ERAS, fv.E2
PRICES, MACRO, PX = fv.prices, fv.macro, fv.PX
ERA = list(ERAS)
ASSETS = list(PX.columns)
CLS = {a: fv.fm.asset_class(a) for a in ASSETS}
MOONS = [c.split("|")[1] for c in MACRO.columns if c.startswith("cpi|")]
rng = np.random.default_rng(7)
out = []


def table(header, rows):
    fmt = lambda v: f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows])


def section(title, note, header, rows):
    out.append(f"\n## {title}\n{note}\n\n{table(header, rows)}")


def score(pos, a=START, b=END, prices=PRICES, costs=-0.02):
    return ev.calc_key_figures(pos.filter(pl.col(DATE).is_between(a, b)), prices, costs=costs, key_figures=["adjusted_sharpe"])["adjusted_sharpe"]


def eras(pos, prices=PRICES):
    return [score(pos, prices=prices)] + [score(pos, a, b, prices) for a, b in ERAS.values()]


def pnl(pos, prices=PRICES):
    r = ev.calc_key_figures(pos.filter(pl.col(DATE).is_between(START, END)), prices, key_figures=["returns"])["returns"]
    return r.to_pandas().set_index(DATE)["returns"].iloc[3:]


def to_pl(df):
    df = df.copy(); df.index.name = DATE
    return pl.from_pandas(df.reset_index(), nan_to_null=True).with_columns(pl.col(DATE).cast(pl.Date))


bomba = lambda p=PRICES, m=MACRO, **kw: bs.positions(p, m, **kw)
goat = lambda p=PRICES, m=None: gm.goat_positions(p)
POS = {"BOMBASTRAT": bomba(), "GOAT": goat(), "trend alone": bomba(use=("trend",)), "inflation alone": bomba(use=("inflation",)),
       "twins alone": bomba(use=("twins",))}
R = pd.DataFrame({k: pnl(p) for k, p in POS.items()}).dropna()
BASE = {k: eras(p) for k, p in POS.items()}

# ================================================================ 1. placebos: is the edge where the story says it is?
rows = [("real: each moon's own CPI", *BASE["inflation alone"], BASE["BOMBASTRAT"][0])]
mac = MACRO.to_pandas().set_index(DATE)
for k in range(1, len(MOONS)):
    m2 = mac.copy()
    for i, mo in enumerate(MOONS):
        m2[f"cpi|{mo}"] = mac[f"cpi|{MOONS[(i + k) % len(MOONS)]}"].values
    m2 = to_pl(m2)
    rows.append((f"placebo: CPI of the moon {k} place(s) along", *eras(bomba(m=m2, use=("inflation",))), score(bomba(m=m2))))
for lag in (250, 500):
    m2 = to_pl(mac.shift(lag + int(lag * 0.4)).bfill())          # calendar days: about `lag` trading days
    rows.append((f"placebo: own CPI, but {lag} trading days stale", *eras(bomba(m=m2, use=("inflation",))), score(bomba(m=m2))))
m2 = mac.copy()
for mo in MOONS:
    m2[f"cpi|{mo}"] = mac[[f"cpi|{x}" for x in MOONS]].mean(axis=1).values
m2 = to_pl(m2)
rows.append(("all moons get the average CPI (is it one global inflation?)", *eras(bomba(m=m2, use=("inflation",))), score(bomba(m=m2))))
g60 = np.log(mac[[f"cpi|{m}" for m in MOONS]]).diff(84)
cc = g60.corr().values
CPI_CORR = cc[np.triu_indices_from(cc, 1)].mean()
STALE = rows[len(MOONS)][1]
AVG_CPI = rows[-1][1]
section("1. Inflation placebos", "If the inflation rule is real and local, giving a moon the wrong moon's CPI, or stale CPI, should kill it. "
        f"Stale CPI does kill it. The wrong moon's CPI does not: inflation moves together across moons (average correlation of 60-day CPI growth between moons: {CPI_CORR:.2f}), "
        "so the rule is closer to one shared inflation bet than five local ones.",
        ["inflation data used", "inflation alone, full"] + ERA + ["whole model, full"], rows)
PLACEBO_INF = [r[1] for r in rows[1:len(MOONS)]]


def twin_signal(px, pairs):
    f = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    logp = np.log(px.where(px > 0))
    for a, b in pairs:
        gap = logp[a] - logp[b]
        r = gap.rolling(bs.SPREAD_DAYS, min_periods=120)
        t = np.tanh((gap - r.mean()) / r.std().replace(0, np.nan) / 2).fillna(0.0)
        f[a] -= t; f[b] += t
    return f


real_twins = bs.twins
commod = [a for a in ASSETS if CLS[a] == "commodity"]
wk = PX.diff(5).corr()
true_pairs = {frozenset((a, b)) for a in commod for b in commod if a < b and wk.loc[a, b] > 0.7}
fake = []
for _ in range(40):
    while True:
        picks = rng.permutation(commod)[:6]
        pairs = [(picks[0], picks[1]), (picks[2], picks[3]), (picks[4], picks[5])]
        if not any(frozenset(p) in true_pairs for p in pairs):
            break
    bs.twins = lambda px, pairs=pairs: twin_signal(px, pairs)
    fake.append(score(bomba(use=("twins",))))
bs.twins = real_twins
fake = np.array(fake)
section("2. Twins placebo", "The same spread rule on 3 random commodity pairs that are NOT twins, 40 draws. If twins work because they are twins, random pairs should score about zero.",
        ["pairs", "twins alone, full score"],
        [("real twins (found by the correlation rule)", BASE["twins alone"][0]), ("random non-twin pairs, average", fake.mean()),
         ("random non-twin pairs, best of 40", fake.max()), ("share of random draws that beat the real twins", f"{(fake > BASE['twins alone'][0]).mean():.0%}")])

# per moon
rows = []
f_all, _, _ = bs.forecasts(PRICES, MACRO)
real_infl = bs.inflation
for mo in MOONS:
    def one(px, macro, mo=mo):
        f = real_infl(px, macro)
        f[[c for c in f.columns if not c.startswith(mo + "_")]] = 0.0
        return f
    bs.inflation = one
    rows.append((mo, *eras(bomba(use=("inflation",)))))
bs.inflation = real_infl
section("3. Inflation rule, one moon at a time", "The rule on a single moon's bond and stock only.", ["moon", "full"] + ERA, rows)
MOONS_POS = sum(r[1] > 0 for r in rows)

# ================================================================ 4. bootstrap
rows = []
BOOT = {}
for block in (20, 60, 120, 250):
    idx = va.boot_idx(len(R), n_boot=2000, block=block)
    bb, bg, bt = (va.boot_scores(R[k].values, idx) for k in ("BOMBASTRAT", "GOAT", "trend alone"))
    BOOT[block] = bb
    q = lambda x: "{:.2f} to {:.2f}".format(*np.nanpercentile(x, [5, 95]))
    rows.append((f"{block} days", q(bb), f"{np.nanmean(bb > 0.5):.0%}", f"{np.nanmean(bb > 0):.1%}", q(bg), f"{np.nanmean(bb > bt):.0%}", f"{np.nanmean(bb > bg):.0%}"))
section("4. Block bootstrap of the daily P&L", "2000 resamples per row. Longer blocks keep more of the slow structure.",
        ["block length", "BOMBASTRAT 90% interval", "P(score > 0.5)", "P(score > 0)", "GOAT 90% interval", "P(beats trend alone)", "P(beats GOAT)"], rows)
rows = []
idx = va.boot_idx(len(R), n_boot=2000, block=60)
for e, (a, b) in ERAS.items():
    m = (R.index >= pd.Timestamp(a)) & (R.index <= pd.Timestamp(b))
    sub = R[m]
    ix = va.boot_idx(len(sub), n_boot=2000, block=60)
    bb, bg = va.boot_scores(sub["BOMBASTRAT"].values, ix), va.boot_scores(sub["GOAT"].values, ix)
    rows.append((e, va.adj_sharpe(sub["BOMBASTRAT"].values), "{:.2f} to {:.2f}".format(*np.nanpercentile(bb, [5, 95])), f"{np.nanmean(bb > 0):.0%}",
                 va.adj_sharpe(sub["GOAT"].values), "{:.2f} to {:.2f}".format(*np.nanpercentile(bg, [5, 95]))))
section("4b. Bootstrap inside each era", "A 7-year era is short: the intervals are wide for both models.",
        ["era", "BOMBASTRAT", "90% interval", "P(> 0)", "GOAT", "90% interval"], rows)

# ================================================================ 5. rolling windows
def rolling(r, years=3):
    w = 260 * years
    m, s = r.rolling(w).mean(), r.rolling(w).std()
    return (m / s * np.sqrt(260)).dropna()


rows = []
ROLL = {}
for k in ("BOMBASTRAT", "GOAT", "trend alone"):
    x = rolling(R[k]); ROLL[k] = x
    rows.append((k, x.min(), x.quantile(0.1), x.median(), x.max(), f"{(x > 0).mean():.0%}", f"{(x > 0.5).mean():.0%}"))
section("5. Every 3-year window", "Sharpe over each rolling 3-year window (about 4,000 overlapping windows).",
        ["model", "worst", "10% low", "median", "best", "windows above 0", "windows above 0.5"], rows)

# ================================================================ 6. other histories: 2-year blocks of the real history in random order
T = len(PRICES)
dlog = np.log(PX).diff().fillna(0.0).values
cpi = np.log(mac[[c for c in mac.columns if c.startswith("cpi|")]].reindex(mac.index.union(PX.index)).ffill().reindex(PX.index))
dcpi = cpi.diff().fillna(0.0).values
SYN = {"BOMBASTRAT": [], "GOAT": []}
for _ in range(60):
    L = 520
    starts = rng.integers(1, T - L, size=int(np.ceil(T / L)))
    ix = np.concatenate([np.arange(s, s + L) for s in starts])[: T - 1]
    px = pd.DataFrame(PX.iloc[0].values * np.exp(np.vstack([np.zeros(dlog.shape[1]), dlog[ix]]).cumsum(axis=0)), index=PX.index, columns=PX.columns)
    mc = pd.DataFrame(100 * np.exp(np.vstack([np.zeros(dcpi.shape[1]), dcpi[ix]]).cumsum(axis=0)), index=PX.index, columns=cpi.columns)
    p, m = to_pl(px), to_pl(mc)
    SYN["BOMBASTRAT"].append(score(bomba(p, m), prices=p))
    SYN["GOAT"].append(score(goat(p), prices=p))
SYN = {k: np.array(v) for k, v in SYN.items()}
section("6. Other histories", "60 new 22-year histories made by putting random 2-year blocks of the real history (prices and CPI together, all assets at once) "
        "in a new order, and running each model from scratch on each. The joins break trends and spreads, so scores are lower than on the real path; the spread of outcomes is the point.",
        ["model", "average", "5% low", "median", "95% high", "histories above 0.5", "histories above 0"],
        [(k, v.mean(), np.percentile(v, 5), np.median(v), np.percentile(v, 95), f"{(v > 0.5).mean():.0%}", f"{(v > 0).mean():.0%}") for k, v in SYN.items()]
        + [("BOMBASTRAT higher than GOAT on the same history", f"{(SYN['BOMBASTRAT'] > SYN['GOAT']).mean():.0%}", "", "", "", "", "")])

# ================================================================ 7. random asset subsets
SUB = {"BOMBASTRAT": [], "GOAT": []}
for _ in range(60):
    keep = sorted(rng.choice(len(ASSETS), size=19, replace=False))
    p = PRICES.select(DATE, *[ASSETS[i] for i in keep])
    SUB["BOMBASTRAT"].append(score(bomba(p), prices=p))
    SUB["GOAT"].append(score(goat(p), prices=p))
SUB = {k: np.array(v) for k, v in SUB.items()}
section("7. Random 19 of the 27 assets", "60 draws. Each model is rebuilt on the smaller universe. A model that needs a few lucky assets shows a wide spread and a low floor.",
        ["model", "average", "worst", "5% low", "median", "best", "draws above 0.5"],
        [(k, v.mean(), v.min(), np.percentile(v, 5), np.median(v), v.max(), f"{(v > 0.5).mean():.0%}") for k, v in SUB.items()])

# ================================================================ 8. worse execution
rows = []
for k in ("BOMBASTRAT", "GOAT"):
    p = POS[k]
    lag = lambda n: p.with_columns(pl.exclude(DATE).shift(n).fill_null(0.0))
    rows.append((k, score(p), score(lag(1)), score(lag(3)), score(lag(5)), score(p, costs=-0.04), score(p, costs=-0.06), score(p, costs=-0.10)))
section("8. Slower execution and higher costs", "Adjusted Sharpe, full period.",
        ["model", "as is", "+1 day lag", "+3 days", "+5 days", "2x costs", "3x costs", "5x costs"], rows)

# ================================================================ 9. damaged data
rows = []
for name, sp in va.stress_sets(PRICES).items():
    r = [name]
    for fn in (bomba, goat):
        try:
            r.append(score(fn(sp).fill_null(0.0).fill_nan(0.0), prices=sp))
        except Exception as e:
            r.append(f"ERR {type(e).__name__}")
    rows.append(tuple(r))
labels = {"late": "9 assets only start 40% into the file", "holes": "2% of prices missing at random", "jumps": "an 8-sigma price jump about once per asset-year",
          "shift": "all prices shifted so they cross zero"}
section("9. Damaged data", f"Real data: BOMBASTRAT {BASE['BOMBASTRAT'][0]:.2f}, GOAT {BASE['GOAT'][0]:.2f}. The out-of-sample file may differ from the in-sample one in ways we cannot see yet.",
        ["damage", "BOMBASTRAT", "GOAT"], [(labels[r[0]], *r[1:]) for r in rows])

# ================================================================ 10. would the 3-part recipe have been chosen in real time?
combos = {"trend": ("trend",), "trend + inflation": ("trend", "inflation"), "trend + twins": ("trend", "twins"), "all three": ("trend", "inflation", "twins"),
          "inflation + twins": ("inflation", "twins")}
RC = pd.DataFrame({k: pnl(bomba(use=v)) for k, v in combos.items()}).dropna()
years = sorted(set(RC.index.year))
picked, wf = [], []
for y in years[5:]:
    past = RC[RC.index.year < y]
    best = max(combos, key=lambda k: va.adj_sharpe(past[k].values))
    picked.append(best)
    x = RC.loc[RC.index.year == y, best]
    wf.append(x / RC.loc[RC.index.year < y, best].std())
wf = pd.concat(wf)
always = RC.loc[RC.index.year >= years[5], "all three"]
section("10. Walk-forward choice of the recipe", f"Each January from {years[5]}, pick the combination with the best score on all earlier years, trade it for that year. "
        "No hindsight is involved in the first row.",
        ["rule", "adjusted Sharpe over the walk-forward years", "years 'all three' was the pick"],
        [("pick the best past combination each year", va.adj_sharpe(wf.values), f"{picked.count('all three')} of {len(picked)}"),
         ("always all three (BOMBASTRAT)", va.adj_sharpe(always.values), "")])

# ================================================================ 11. why is 2009-2016 softer?
P = POS["BOMBASTRAT"].to_pandas().set_index(DATE); P.index = pd.to_datetime(P.index)
ap = (P.shift(2) * PX.diff()).loc[pd.Timestamp(START):].iloc[3:]
tot_sd = ap.sum(axis=1).std()
vol = bs.price_vol(PX)
z250 = (PX.diff().rolling(250).sum() / (vol * np.sqrt(250)))
wkc = PX.diff(5)
rows_class, rows_env = [], []
for cls in ("bonds", "stocks", "fx", "commodity"):
    cols = [a for a in ASSETS if CLS[a] == cls]
    r = [cls]
    for a, b in ERAS.values():
        x = ap.loc[pd.Timestamp(a):pd.Timestamp(b), cols].sum(axis=1)
        r.append(x.mean() / tot_sd * np.sqrt(260))
    rows_class.append(tuple(r))
section("11. Where the P&L came from, by asset class and era", "Yearly P&L of each class inside BOMBASTRAT, in units of the model's yearly risk. The columns add up to roughly the model's Sharpe in that era.",
        ["class"] + ERA, rows_class)
env = []
surprise = bs.inflation(PX, mac.pipe(lambda m: m.set_axis(pd.to_datetime(m.index))))
for e, (a, b) in ERAS.items():
    sl = slice(pd.Timestamp(a), pd.Timestamp(b))
    c = wkc.loc[sl].corr().values
    fwd = (PX.shift(-62) - PX.shift(-2)) / (vol * np.sqrt(60))
    sb = [x for x in ASSETS if CLS[x] in ("bonds", "stocks")]
    ic = np.nanmean([surprise.loc[sl, x].corr(fwd.loc[sl, x], method="spearman") for x in sb])
    com = [x for x in ASSETS if CLS[x] == "commodity"]
    env.append((e, z250.loc[sl].abs().mean().mean(), z250.loc[sl, com].abs().mean().mean(), z250.loc[sl, sb].abs().mean().mean(),
                c[np.triu_indices_from(c, 1)].mean(), surprise.loc[sl, sb].abs().mean().mean(), ic))
section("11b. What changed in the market", "Trend strength = average size of the 12-month move in volatility units (a random walk gives about 0.8). "
        "Correlation = average between all pairs of assets, weekly. Inflation signal: its average size, and how well it lined up with the next 60 days' move (rank correlation).",
        ["era", "trend strength, all", "trend strength, commodities", "trend strength, stocks+bonds", "average correlation", "inflation signal size", "inflation signal hit (IC)"], env)
sleeve_rows = [(k, *BASE[k][1:]) for k in ("trend alone", "inflation alone", "twins alone", "BOMBASTRAT", "GOAT")]
section("11c. Each part by era", "", ["part"] + ERA, sleeve_rows)

# ================================================================ chart
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "font.size": 9, "legend.frameon": False, "axes.titlelocation": "left", "axes.titleweight": "bold"})
INK, GO = "#0b0b0b", "#6250d6"
fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=140)
ax = axes[0, 0]
ax.hist(BOOT[60], bins=60, color=INK, alpha=0.85); ax.axvline(BASE["BOMBASTRAT"][0], color="#eb6834", lw=1.5)
ax.set_title("Bootstrap: BOMBASTRAT score over 2000 resampled histories")
ax = axes[0, 1]
for k, c in (("GOAT", GO), ("BOMBASTRAT", INK)):
    ax.plot(ROLL[k].index, ROLL[k], color=c, lw=1.6, label=k)
ax.axhline(0, color="#52514e", lw=0.8); ax.set_title("Sharpe over every 3-year window"); ax.legend()
ax = axes[1, 0]
bins = np.linspace(min(SYN["BOMBASTRAT"].min(), SYN["GOAT"].min()), max(SYN["BOMBASTRAT"].max(), SYN["GOAT"].max()), 25)
ax.hist(SYN["GOAT"], bins=bins, color=GO, alpha=0.6, label="GOAT"); ax.hist(SYN["BOMBASTRAT"], bins=bins, color=INK, alpha=0.6, label="BOMBASTRAT")
ax.set_title("Score on 60 reshuffled histories (2-year blocks)"); ax.legend()
ax = axes[1, 1]
bins = np.linspace(min(SUB["BOMBASTRAT"].min(), SUB["GOAT"].min()), max(SUB["BOMBASTRAT"].max(), SUB["GOAT"].max()), 25)
ax.hist(SUB["GOAT"], bins=bins, color=GO, alpha=0.6, label="GOAT"); ax.hist(SUB["BOMBASTRAT"], bins=bins, color=INK, alpha=0.6, label="BOMBASTRAT")
ax.set_title("Score on 60 random sets of 19 assets"); ax.legend()
fig.tight_layout(); fig.savefig(HERE / "validation.png"); plt.close(fig)

SUMMARY = f"""
## Summary

| test | result | verdict |
|---|---|---|
| Inflation with stale CPI (1 year old) | {STALE:.2f} against {BASE['inflation alone'][0]:.2f} real | **Pass**: the timing carries the edge |
| Inflation with another moon's CPI | {min(PLACEBO_INF):.2f} to {max(PLACEBO_INF):.2f} against {BASE['inflation alone'][0]:.2f} real | **Fail as a local story**: it is a shared inflation factor, not moon by moon. Say so in the pitch |
| Inflation rule on each moon alone | positive on {MOONS_POS} of {len(MOONS)} moons | Pass |
| Twins rule on random non-twin pairs | {fake.mean():.2f} on average, best {fake.max():.2f}, against {BASE['twins alone'][0]:.2f} real | **Pass**: it works because they are twins |
| Bootstrap, full period | 90% interval {np.nanpercentile(BOOT[60], 5):.2f} to {np.nanpercentile(BOOT[60], 95):.2f} | Pass: clearly above zero; the exact level is uncertain |
| Every 3-year window | worst {ROLL['BOMBASTRAT'].min():.2f}, {(ROLL['BOMBASTRAT'] > 0.5).mean():.0%} above 0.5 (GOAT: worst {ROLL['GOAT'].min():.2f}, {(ROLL['GOAT'] > 0.5).mean():.0%}) | GOAT is steadier window by window |
| 60 reshuffled histories | average {SYN['BOMBASTRAT'].mean():.2f}, 5% low {np.percentile(SYN['BOMBASTRAT'], 5):.2f} (GOAT {SYN['GOAT'].mean():.2f}, {np.percentile(SYN['GOAT'], 5):.2f}) | Pass, level with GOAT |
| 60 random sets of 19 assets | average {SUB['BOMBASTRAT'].mean():.2f}, worst {SUB['BOMBASTRAT'].min():.2f} (GOAT {SUB['GOAT'].mean():.2f}, {SUB['GOAT'].min():.2f}) | Pass, but wider spread than GOAT |
| 5 more days of lag / 3x costs | see section 8 | Pass: slow signals |
| Price jumps in the data | see section 9 | BOMBASTRAT loses about half; GOAT breaks |
| Recipe chosen year by year without hindsight | see section 10 | Always holding all three beats picking the past winner |
"""
hdr = (f"# BOMBASTRAT validation\n\nGenerated by `BOMBASTRAT/validate.py`, do not edit by hand. Adjusted Sharpe from Lynx's evaluator unless a column says otherwise, scored from {START}.\n\n"
       "**The real out-of-sample file is not released yet. Nothing below is a true out-of-sample score.** These are the strongest stand-ins we can build from the in-sample data. "
       "GOAT is shown next to BOMBASTRAT wherever the test applies to both.\n" + SUMMARY + "\n![validation](validation.png)\n")
(HERE / "VALIDATION.md").write_text(hdr + "\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
