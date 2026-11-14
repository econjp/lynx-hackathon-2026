"""One command from Lynx CSVs to the positions file.

    python submission/run_final.py \
        --prices data/close_prices_insample.csv  path/to/close_prices_outofsample.csv \
        --macro  data/macro_data_insample.csv    path/to/macro_data_outofsample.csv \
        --out    submission/positions.csv

Give the in-sample file first, then the out-of-sample file (the history is needed to warm the
signals up). Files are stacked by date; if a date is in both, the later file wins. A price file
with Open/High/Low/Close columns ("asset|Close") also works: only the closes are used.
--from-date YYYY-MM-DD writes only the rows from that date on.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl

from src.final_model import DATE, final_positions


def load(paths, closes_only=False):
    frames = []
    for p in paths:
        df = pl.read_csv(p, try_parse_dates=True)
        df = df.rename({c: DATE for c in df.columns if c.lower() in ("asofdate", "date", "dates")})
        df = df.with_columns(pl.col(DATE).cast(pl.Date), pl.exclude(DATE).cast(pl.Float64, strict=False))
        if closes_only and any(c.endswith("|Close") for c in df.columns):
            df = df.select(DATE, *[pl.col(c).alias(c.split("|")[0]) for c in df.columns if c.endswith("|Close")])
        frames.append(df)
    cols = frames[0].columns
    for p, f in zip(paths, frames):
        missing = set(cols) - set(f.columns)
        if missing:
            sys.exit(f"{p} is missing columns: {sorted(missing)}")
    return pl.concat([f.select(cols) for f in frames]).unique(subset=DATE, keep="last").sort(DATE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", nargs="+", required=True)
    ap.add_argument("--macro", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "submission" / "positions.csv"))
    ap.add_argument("--from-date", type=date.fromisoformat, default=None)
    a = ap.parse_args()

    prices, macro = load(a.prices, closes_only=True), load(a.macro)
    pos = final_positions(prices, macro)
    assert pos.columns == prices.columns and len(pos) == len(prices)
    assert pos.null_count().sum_horizontal()[0] == 0
    if a.from_date:
        pos = pos.filter(pl.col(DATE) >= a.from_date)
    pos.write_csv(a.out)
    print(f"{a.out}: {len(pos)} rows, {pos.width - 1} assets, {pos[DATE][0]} to {pos[DATE][-1]}")
    if macro[DATE][-1] < prices[DATE][-1]:
        print(f"WARNING: macro ends {macro[DATE][-1]}, prices end {prices[DATE][-1]}; the last CPI value is carried forward")


if __name__ == "__main__":
    main()
