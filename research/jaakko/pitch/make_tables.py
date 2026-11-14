"""Builds every number in the pitch from raw data. Anyone can rerun it and get the same tables.
python research/jaakko/pitch/make_tables.py  ->  research/jaakko/pitch/TABLES.md"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(ROOT), str(ROOT / "lynx_materials")]

import numpy as np
import pandas as pd
import polars as pl
import evaluation as ev
from src.utils import DATE
from src.baseline import champion_model as team_baseline
from signals import SPEEDS, clean_prices, ewmac, breakout, scale_and_cap
from risk import price_vol, size
from portfolio import exante_scale, glide

DATA = ROOT / "data" / "close_prices_insample.csv"
prices = pl.read_csv(DATA, try_parse_dates=True).with_columns(pl.col(DATE).cast(pl.Date))
px = prices.to_pandas().set_index(DATE).astype(float)
d = prices[DATE]
PERIODS = [(d[0], d[len(d) // 3]), (d[len(d) // 3], d[2 * len(d) // 3]), (d[2 * len(d) // 3], d[-1])]
NAMES = [f"{a.year}-{b.year}" for a, b in PERIODS]
out = []


def pl_pos(pos):
    pos = pos.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pos.index.name = DATE
    return pl.from_pandas(pos.reset_index()).with_columns(pl.col(DATE).cast(pl.Date))


def kf(pos, a=None, b=None, costs=-0.02):
    if a is not None:
        pos = pos.filter(pl.col(DATE).is_between(a, b))
    return ev.calc_key_figures(pos, prices, costs=costs, key_figures=[
        "adjusted_sharpe", "sharpe", "sharpe_1d_lag", "sharpe_ex_slippage", "skewness", "kurtosis",
        "av_holding", "position_bias", "returns", "sharpe_per_instrument"])


def daily(r):
    s = r["returns"]
    return s[s.columns[-1]].to_numpy()


def table(title, df, note=""):
    out.append(f"\n## {title}\n" + (f"{note}\n\n" if note else "\n") + df.to_markdown(index=False, floatfmt=".2f"))


# same layers after the signal for every candidate
vol = price_vol(px)
spx = clean_prices(px, vol)
diff = spx.diff()
book = lambda f: pl_pos(glide(exante_scale(size(f, vol), px)))
mom = lambda n: scale_and_cap(diff.rolling(n).sum() / (vol * np.sqrt(n)))
slow = ewmac(spx, vol, SPEEDS["slow"])
m250 = mom(250)

SIGNALS = {
    "EWMAC slow (32/128+64/256)": slow,
    "EWMAC slow + 12m momentum 50/50": (slow + m250) / 2,
    "EWMAC mid": ewmac(spx, vol, SPEEDS["mid"]),
    "EWMAC fast": ewmac(spx, vol, SPEEDS["fast"]),
    "12m momentum": m250,
    "momentum 50+100+200d": (mom(50) + mom(100) + mom(200)) / 3,
    "price - MA200": scale_and_cap((spx - spx.rolling(200).mean()) / (vol * np.sqrt(200))),
    "breakout 80/160/320": scale_and_cap(breakout(spx)),
    "cross-sectional 12m momentum": 2 * (m250.rank(axis=1, pct=True) - 0.5),
    "sign(EWMAC slow), on/off": np.sign(slow),
}
POS = {k: book(f) for k, f in SIGNALS.items()}

# baselines with their own sizing
dpx = px.diff()
lynx = np.sign(dpx.rolling(50).sum() / np.sqrt((dpx ** 2).rolling(100).sum())) / dpx.rolling(100).std()
lynx = lynx.replace([np.inf, -np.inf], np.nan).fillna(0.0)
pnl = (lynx.shift(2) * dpx).sum(axis=1)
POS["Lynx baseline (sign 50d)"] = pl_pos(lynx.div(pnl.rolling(100).std().replace(0, np.nan), axis=0))
ret = prices.select(DATE, pl.exclude(DATE).diff())
POS["team baseline (40/80 tanh)"] = team_baseline(ret).with_columns(pl.col(DATE).cast(pl.Date))

# T1: signals across 3 periods
rows = []
for k, pos in POS.items():
    r = [kf(pos, a, b) for a, b in PERIODS]
    full = kf(pos)
    rows.append({"signal": k, **{n: x["adjusted_sharpe"] for n, x in zip(NAMES, r)},
                 "worst": min(x["adjusted_sharpe"] for x in r), "full": full["adjusted_sharpe"],
                 "holding d": full["av_holding"]})
table("T1. Signals, adjusted sharpe per period", pd.DataFrame(rows),
      "Same sizing / risk / execution for every signal (baselines use their own). 'worst' = worst period.")

# T2: KPIs for the two candidates
rows = []
for k in ["EWMAC slow (32/128+64/256)", "EWMAC slow + 12m momentum 50/50", "team baseline (40/80 tanh)", "Lynx baseline (sign 50d)"]:
    r = kf(POS[k])
    x = np.nan_to_num(daily(r))
    eq = np.cumsum(x)
    dd = (eq - np.maximum.accumulate(eq)).min() / (x.std() * np.sqrt(260))
    rows.append({"model": k, "adj sharpe": r["adjusted_sharpe"], "sharpe": r["sharpe"],
                 "sharpe +1d lag": r["sharpe_1d_lag"], "sharpe no costs": r["sharpe_ex_slippage"],
                 "skew": r["skewness"], "kurtosis": r["kurtosis"], "holding d": r["av_holding"],
                 "bias": r["position_bias"], "max DD (yrs of vol)": dd})
table("T2. KPIs, full period 1995-2016", pd.DataFrame(rows),
      "max DD in units of annual vol (1.0 = lost one year's worth of volatility).")

# T3: stress tests on the pick
PICK = "EWMAC slow + 12m momentum 50/50"
base = kf(POS[PICK])
x = np.nan_to_num(daily(base))
top10 = np.argsort(x)[-10:]
x_no10 = np.delete(x, top10)
rng = np.random.default_rng(0)
f = SIGNALS[PICK]
placebo = []
for _ in range(10):
    # shift each asset's signal by a random 1-10 years: same signal shape, timing broken
    sh = f.apply(lambda c: np.roll(c.values, rng.integers(260, 2600)))
    placebo.append(kf(book(sh))["adjusted_sharpe"])
pos_assets = sum(v > 0 for v in base["sharpe_per_instrument"].values())
rows = [
    {"test": "base", "result": base["adjusted_sharpe"]},
    {"test": "1 extra day of lag (sharpe)", "result": base["sharpe_1d_lag"]},
    {"test": "2x slippage", "result": kf(POS[PICK], costs=-0.04)["adjusted_sharpe"]},
    {"test": "without the 10 best days (sharpe)", "result": x_no10.mean() / x_no10.std() * np.sqrt(260)},
    {"test": "placebo: signal shifted in time, mean of 10", "result": np.mean(placebo)},
    {"test": "placebo: best of 10", "result": np.max(placebo)},
    {"test": f"assets with positive sharpe (of {len(base['sharpe_per_instrument'])})", "result": pos_assets},
]
table(f"T3. Stress tests on the pick ({PICK}), full period", pd.DataFrame(rows),
      "If the edge were luck or look-ahead, the placebo would be close to the base.")

# T4: leak test
def leak(fn, cuts=(0.3, 0.6, 0.9)):
    full = fn(prices).drop(DATE).to_numpy()
    for c in cuts:
        n = int(len(prices) * c)
        part = fn(prices.slice(0, n)).drop(DATE).to_numpy()
        if np.abs(full[:n] - part).max() > 1e-9:
            return "FAIL"
    return "PASS"

def pick_model(p):
    q = p.to_pandas().set_index(DATE).astype(float)
    v = price_vol(q); s = clean_prices(q, v)
    m = scale_and_cap(s.diff().rolling(250).sum() / (v * np.sqrt(250)))
    return pl_pos(glide(exante_scale(size((ewmac(s, v, SPEEDS["slow"]) + m) / 2, v), q)))

table("T4. Leak test (cut history -> same positions)", pd.DataFrame([{"model": PICK, "result": leak(pick_model)}]))

# T5: signal correlations
names = ["EWMAC slow (32/128+64/256)", "12m momentum", "price - MA200", "breakout 80/160/320", "cross-sectional 12m momentum"]
st = pd.DataFrame({k: SIGNALS[k].stack() for k in names}).dropna()
c = st.corr().round(2)
c.insert(0, "signal", c.index)
table("T5. Correlation between signals", c, "High = same idea, little diversification.")

hdr = f"# Signal pitch tables\n\nData: {DATA.name}, {px.shape[1]} assets, {d[0]} to {d[-1]}. Score: Lynx evaluation.py.\nGenerated by make_tables.py, do not edit by hand.\n"
(HERE / "TABLES.md").write_text(hdr + "\n".join(out) + "\n", encoding="utf-8")
print(hdr + "\n".join(out))
