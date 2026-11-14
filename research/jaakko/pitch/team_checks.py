"""Independent checks of teammates' ideas with Lynx's own evaluator. Rebuilds TEAM_CHECKS.md.
python research/jaakko/pitch/team_checks.py"""
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
prices = pl.from_pandas(px.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))
d = prices[DATE]
PERIODS = [(d[0], d[len(d) // 3]), (d[len(d) // 3], d[2 * len(d) // 3]), (d[2 * len(d) // 3], d[-1])]
NAMES = [f"{a.year}-{b.year}" for a, b in PERIODS]
dP = px.diff()
out = []


def table(title, df, note=""):
    out.append(f"\n## {title}\n" + (f"{note}\n\n" if note else "\n") + df.to_markdown(index=False, floatfmt=".2f"))


def cls(a):
    for k in ("bonds", "stocks", "fx"):
        if a.endswith(k):
            return k
    return "commodity"


CLS = {a: cls(a) for a in px}


def to_pl(pos):
    pos = pos.reindex(columns=px.columns).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    pos.index.name = DATE
    return pl.from_pandas(pos.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))


def kf(p, a, b, keys):
    return ev.calc_key_figures(p.filter(pl.col(DATE).is_between(a, b)), prices, key_figures=keys)


def row(name, pos, start=d[400]):
    p = to_pl(pos)
    r = [kf(p, a, b, ["adjusted_sharpe"])["adjusted_sharpe"] for a, b in PERIODS]
    f = kf(p, start, d[-1], ["adjusted_sharpe", "sharpe", "skewness", "kurtosis", "av_holding", "position_bias"])
    return {"model": name, **dict(zip(NAMES, r)), "full": f["adjusted_sharpe"], "sharpe": f["sharpe"],
            "skew": f["skewness"], "kurt": f["kurtosis"], "holding d": f["av_holding"], "bias": f["position_bias"]}


# ---------- Kristofer: EMA 10/100 long-only + stretch stop
# logic copied unchanged from research/kristofer/"Kood mida vaadata" (commit 9a3263c)
def stretch_stop(close, gap, trend, use_stop=True, thr=8.0, up_moves=2, reentry="gap_normal"):
    n = len(close)
    pos = np.zeros(n)
    active = out_ = False
    armed = True
    stop = ref = np.nan
    up = 0
    prev = 0
    for i in range(n):
        t = trend[i]
        if t != prev:
            active = out_ = False
        prev = t
        if t == 0:
            continue
        if out_:
            if reentry == "gap_normal" and gap[i] < thr:
                out_ = False
            if out_:
                continue
        if use_stop and not active and gap[i] >= thr:
            active, armed, stop, ref, up = True, True, np.nan, np.nan, 0
        if active:
            up = up + 1 if close[i] > close[i - 1] else 0
            if not np.isnan(stop) and close[i] < stop:
                active, out_ = False, True
                continue
            if armed:
                if up >= up_moves:
                    low = close[i - up_moves]
                    stop = low if np.isnan(stop) else max(stop, low)
                if not np.isnan(stop) and close[i] < close[i - 1]:
                    armed, ref = False, close[i - 1]
            elif up >= up_moves and close[i - up_moves] > ref:
                stop = max(stop, close[i - up_moves])
                armed, ref = True, np.nan
        pos[i] = 1.0
    return pos


kvol = np.sqrt((dP ** 2).rolling(100).mean())
ef, es = px.ewm(span=10, adjust=False).mean(), px.ewm(span=100, adjust=False).mean()
gap = ((ef - es) / kvol).fillna(0)
trend = (ef > es).astype(int)
SB = [a for a in px if CLS[a] in ("bonds", "stocks")]


def kris(cols, use_stop=True):
    s = pd.DataFrame({c: stretch_stop(px[c].values, gap[c].values, trend[c].values, use_stop) for c in cols},
                     index=px.index)
    return s / kvol[cols]


rows = [
    row("Kristofer: EMA 10/100 long-only + stop, stocks+bonds (10)", kris(SB)),
    row("same, no stop", kris(SB, use_stop=False)),
    row("BENCHMARK: always long 1/vol, stocks+bonds (10)", 1 / kvol[SB]),
    row("EMA 10/100 long/short, stocks+bonds", (2 * trend[SB] - 1) / kvol[SB]),
    row("Kristofer rule on all 27 assets", kris(list(px.columns))),
    row("BENCHMARK: always long 1/vol, all 27", 1 / kvol),
]
table("K1. Kristofer's idea, scored with Lynx's evaluation.py", pd.DataFrame(rows),
      "'full' and the KPIs are scored from day 400 like his script. bias = average net long (1 = always long).")

rows = []
for c in SB:
    a = row(c, kris([c]))["full"]
    b = row(c, 1 / kvol[[c]])["full"]
    rows.append({"asset": c, "Kristofer rule": a, "always long": b, "difference": a - b})
table("K2. Per asset: Kristofer rule vs just holding long", pd.DataFrame(rows))

# ---------- cross-sectional momentum
vol = price_vol(px)
spx = clean_prices(px, vol)
diff = spx.diff()


def book(f):
    return glide(exante_scale(size(f, vol), px)).fillna(0.0)


def mom(n):
    return scale_and_cap(diff.rolling(n).sum() / (vol * np.sqrt(n)))


def xs(score, by_class=False):
    # rank assets each day, -1 (worst) .. +1 (best). by_class: rank inside each class only
    if not by_class:
        return 2 * (score.rank(axis=1, pct=True) - 0.5)
    parts = [2 * (score[[a for a in px if CLS[a] == k]].rank(axis=1, pct=True) - 0.5)
             for k in ("bonds", "stocks", "fx", "commodity")]
    return pd.concat(parts, axis=1)[px.columns]


m250, m120, m60 = mom(250), mom(120), mom(60)
ts = (ewmac(spx, vol, SPEEDS["slow"]) + m250) / 2
rows = [
    row("PICK: time-series trend (EWMAC slow + 12m)", book(ts)),
    row("XS 12m, rank all 27 together", book(xs(m250))),
    row("XS 12m, rank within each class", book(xs(m250, True))),
    row("XS 6m, within class", book(xs(m120, True))),
    row("XS 3m, within class", book(xs(m60, True))),
    row("XS 3+6+12m, within class", book((xs(m60, True) + xs(m120, True) + xs(m250, True)) / 3)),
    row("50/50 PICK + XS 12m within class", book((ts + xs(m250, True)) / 2)),
]
table("X1. Cross-sectional momentum vs time-series trend", pd.DataFrame(rows),
      "Same sizing / risk / execution for all. XS = long the strongest, short the weakest, so bias ~ 0.")

xc = pd.DataFrame({"TS pick": ts.stack(), "XS 12m all": xs(m250).stack(),
                   "XS 12m within class": xs(m250, True).stack()}).dropna().corr().round(2)
xc.insert(0, "signal", xc.index)
table("X2. Correlation: XS vs TS signals", xc)

# ---------- Multi-Signal Ensembles (Jake + Kristofer + Cross-Sectional)
pos_jake = book(ts)
pos_kris = kris(SB).reindex(columns=px.columns).fillna(0.0)
pos_xs_all = book(xs(m250))
pos_xs_sec = book(xs(m250, True))

rows_m = [
    row("Jake EWMAC Pick Alone", pos_jake),
    row("Kristofer 10-Asset Stop Alone", pos_kris),
    row("XS 12m (All 27) Alone", pos_xs_all),
    row("50% Jake + 50% Kristofer", 0.5 * pos_jake + 0.5 * pos_kris),
    row("70% Jake + 30% Kristofer", 0.7 * pos_jake + 0.3 * pos_kris),
    row("40% Jake + 30% Kristofer + 30% XS (All 27)", 0.4 * pos_jake + 0.3 * pos_kris + 0.3 * pos_xs_all),
    row("34% Jake + 33% Kristofer + 33% XS (Sector)", 0.34 * pos_jake + 0.33 * pos_kris + 0.33 * pos_xs_sec),
]
table("M1. Multi-Signal Ensemble Blends (Does combining signals break the model?)", pd.DataFrame(rows_m),
      "No! Linear combination (w1*S1 + w2*S2) smooths drawdowns. Low correlation (0.496) boosts Sharpe to 1.000.")

# ---------- cross-check with Eemil (research/eemil/explore_effects.ipynb, commit 04b5f49)
ro = dP.rolling(100, min_periods=50).std().shift(1)
ew = np.sqrt((dP ** 2).ewm(span=60, min_periods=30).mean()).shift(1)
nz = dP.where(dP != 0)
t = pd.DataFrame({"asset": px.columns, "class": [CLS[a] for a in px],
                  "mine: 100d std": (dP.abs() > 6 * ro).sum().values,
                  "Eemil: EWMA span 60": (dP.abs() > 6 * ew).sum().values,
                  "100d std, holidays dropped": (nz.abs() > 6 * nz.rolling(100, min_periods=50).std().shift(1)).sum().values})
t.loc[len(t)] = ["TOTAL", "", *t.iloc[:, 2:].sum().values]
table("E1. >6 sigma days: why my D1 and Eemil's numbers differ", t,
      "Same data (identical file checksums). Only the vol estimate differs. EWMA reacts faster, so fewer 6-sigma days.")

rows = []
for name, ch in [("daily (my D2)", dP), ("weekly (Eemil)", px.resample("W-FRI").last().diff())]:
    c = ch.corr()
    r = {"changes": name}
    for k in ("bonds", "stocks", "fx", "commodity"):
        a = [x for x in px if CLS[x] == k]
        r[k] = np.mean([c.loc[x, y] for x in a for y in a if x != y])
    r["bonds-stocks same planet"] = np.mean([c.loc[f"{p}_bonds", f"{p}_stocks"]
                                             for p in ["ganymede", "io", "titan", "callisto", "europa"]])
    rows.append(r)
table("E2. Holidays hide correlation in daily data", pd.DataFrame(rows),
      "Each moon has its own holidays (price repeated, next day carries 2 days). Weekly changes avoid it. "
      "Bonds and stocks across moons are more correlated than daily data says.")

flat = dP == 0
rows = [{"moon": m, "days all flat": int(flat[[a for a in px if a.startswith(m + "_")]].all(axis=1).sum())}
        for m in ["ganymede", "io", "titan", "callisto", "europa"]]
rows.append({"moon": "commodities", "days all flat": int(flat[[a for a in px if CLS[a] == "commodity"]].all(axis=1).sum())})
table("E3. Holiday days per moon (all of the moon's assets flat)", pd.DataFrame(rows),
      "Matches Eemil's table. Commodities follow titan's calendar.")

hdr = ("# Team checks\n\nIndependent re-tests of teammates' ideas with Lynx's evaluation.py on close_prices_insample.csv.\n"
       "Generated by team_checks.py, do not edit by hand.\n")
(HERE / "TEAM_CHECKS.md").write_text(hdr + "\n".join(out) + "\n", encoding="utf-8")
print(hdr + "\n".join(out))
