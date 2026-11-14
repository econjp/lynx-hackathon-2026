"""Validation harness: is model A really better than model B, or is it noise?

python research/eemil/validate.py                 -> Lynx example data
python research/eemil/validate.py path/to/x.csv   -> any price csv (use this at 09:00)

Any model is a function  prices (polars, AsOfDate + assets) -> positions (same shape).
For each model it reports:
  full     adjusted Sharpe on the whole file (Lynx evaluator)
  b1..bK   adjusted Sharpe in K equal chronological blocks  (stability)
  lag+1    Sharpe with one extra day of lag                 (short-term alpha / leak smell)
  2xcost   adjusted Sharpe with double slippage             (turnover dependence)
  hold     average holding in days
  kurt     excess kurtosis of daily P&L
  leak     truncation test on every row, not only the last one
  stress   adjusted Sharpe on damaged copies of the data (late start, holidays, jumps, shift)
  boot     block-bootstrap 90% interval, and P(model beats the reference) on paired resamples
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "lynx_materials")]

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

import evaluation as ev
from src.utils import load_prices, DATE

ANN = np.sqrt(260)


# ---------------------------------------------------------------- metrics
def adj_sharpe(r):
    """Lynx adjusted Sharpe on a daily P&L array (same estimators as evaluation.py)."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 30 or r.std(ddof=1) == 0:
        return np.nan
    s = r.mean() / r.std(ddof=1) * ANN
    return s * (1 + s / 6 * stats.skew(r, bias=False) - s**2 / 24 * stats.kurtosis(r, bias=False))


def daily_pnl(pos, prices, costs=-0.02):
    out = ev.calc_key_figures(pos, prices, costs=costs, key_figures=["returns"])["returns"]
    return out.to_pandas().set_index(DATE)["returns"]


def blocks(r, k=5):
    r = r.dropna()
    return [adj_sharpe(x) for x in np.array_split(r.values, k)]


# ---------------------------------------------------------------- leak test
def leak_test(model, prices, cuts=(0.3, 0.5, 0.7, 0.9, 0.97)):
    """Positions computed on a truncated history must equal the full run on every shared row."""
    full = model(prices).drop(DATE).to_numpy()
    worst = 0.0
    for c in cuts:
        n = int(len(prices) * c)
        part = model(prices.slice(0, n)).drop(DATE).to_numpy()
        a, b = np.nan_to_num(full[:n]), np.nan_to_num(part)
        scale = max(np.abs(a).max(), 1e-12)
        worst = max(worst, np.abs(a - b).max() / scale)
    return worst < 1e-9, worst


# ---------------------------------------------------------------- stress data
def _to_pl(px):
    px.index.name = DATE
    return pl.from_pandas(px.reset_index(), nan_to_null=True).with_columns(pl.col(DATE).cast(pl.Date))


def stress_sets(prices, seed=0):
    """Damaged copies of the price file. A robust model should lose little on each."""
    rng = np.random.default_rng(seed)
    px = prices.to_pandas().set_index(DATE).astype(float)
    n, m = px.shape
    sig = px.diff().std()
    out = {}

    late = px.copy()                                   # two assets only start 40% in
    late.iloc[: int(n * 0.4), : max(1, m // 3)] = np.nan
    out["late"] = late

    hol = px.copy()                                    # 2% missing prints (exchange holidays)
    hol = hol.mask(rng.random(px.shape) < 0.02)
    hol.iloc[0] = px.iloc[0]
    out["holes"] = hol

    jump = px.copy()                                   # ~1 roll gap per asset-year, +-8 sigma level shift
    hits = rng.random(px.shape) < 1 / 260
    jump += (hits * rng.choice([-8.0, 8.0], size=px.shape) * sig.values).cumsum(axis=0)
    out["jumps"] = jump

    out["shift"] = px - px.median()                    # prices cross zero; P&L in dP is unchanged
    return {k: _to_pl(v) for k, v in out.items()}


# ---------------------------------------------------------------- bootstrap
def boot_idx(n, n_boot=2000, block=60, seed=1):
    """Circular block bootstrap indices, shared across models so comparisons are paired."""
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, k))
    return ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_boot, -1)[:, :n]


def boot_scores(r, idx):
    return np.array([adj_sharpe(r[i]) for i in idx])


# ---------------------------------------------------------------- models
def registry():
    sys.path[:0] = [str(HERE), str(ROOT / "research" / "jaakko")]
    from src.baseline import champion_model
    import importlib.util

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    jp = load("jp_model", ROOT / "research" / "jaakko" / "model.py")
    em = load("em_model", HERE / "model.py")

    def lynx50(prices):  # Lynx quickstart style: sign of 50d move / vol
        cols = [c for c in prices.columns if c != DATE]
        px = prices.to_pandas().set_index(DATE).astype(float)
        d = px.diff()
        pos = np.sign(d.rolling(50).sum()) / d.rolling(100, min_periods=10).std()
        return _to_pl(pos[cols])

    def team(prices):
        cols = [c for c in prices.columns if c != DATE]
        return champion_model(prices.select(DATE, pl.col(cols).diff()))

    return {
        "lynx50": lynx50,
        "team40/80": team,
        "jp_fast": lambda p: jp.generate_positions(p, speed="fast"),
        "jp_all": lambda p: jp.generate_positions(p, speed="all"),
        "eemil": em.generate_positions,
    }


# ---------------------------------------------------------------- report
def report(prices, models, ref="team40/80", k=5, n_boot=2000):
    rows, pnl = [], {}
    stress = stress_sets(prices)
    for name, fn in models.items():
        pos = fn(prices)
        r = daily_pnl(pos, prices)
        pnl[name] = r
        kf = ev.calc_key_figures(pos, prices)
        row = {"model": name, "full": kf["adjusted_sharpe"]}
        row.update({f"b{i + 1}": v for i, v in enumerate(blocks(r, k))})
        row["lag+1"] = kf["sharpe_1d_lag"]
        row["2xcost"] = adj_sharpe(daily_pnl(pos, prices, costs=-0.04))
        row["hold"] = kf["av_holding"]
        row["kurt"] = kf["kurtosis"] - 3
        row["leak"] = "ok" if leak_test(fn, prices)[0] else "FAIL"
        for sname, sp in stress.items():
            try:
                row[sname] = ev.calc_key_figures(fn(sp), sp, key_figures=["adjusted_sharpe"])["adjusted_sharpe"]
            except Exception as e:  # a crash on bad data is a finding, not a harness error
                row[sname] = f"ERR {type(e).__name__}"
        rows.append(row)

    # paired bootstrap on the days every model has P&L
    df = pd.DataFrame(pnl).dropna()
    idx = boot_idx(len(df), n_boot)
    bs = {name: boot_scores(df[name].values, idx) for name in df}
    for row in rows:
        b = bs[row["model"]]
        row["lo5"], row["hi95"] = np.nanpercentile(b, [5, 95])
        row[f"P(>{ref})"] = np.nan if row["model"] == ref else float(np.nanmean(b > bs[ref]))
    return pd.DataFrame(rows).set_index("model"), df


if __name__ == "__main__":
    prices = load_prices(sys.argv[1] if len(sys.argv) > 1 else None)
    prices = prices.with_columns(pl.col(DATE).cast(pl.Date), pl.exclude(DATE).cast(pl.Float64, strict=False))
    print(f"{prices.width - 1} assets, {len(prices)} days, {prices[DATE][0]} .. {prices[DATE][-1]}\n")
    table, pnl = report(prices, registry())
    pd.set_option("display.width", 250, "display.float_format", "{:.2f}".format)
    print(table.to_string())
    print("\nP&L correlation between models:")
    print(pnl.corr().to_string())
