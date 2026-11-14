"""One command from Lynx CSVs to the positions file.

    python submission/run_final.py \
        --prices data/close_prices_insample.csv  path/to/close_prices_outofsample.csv \
        --macro  data/macro_data_insample.csv    path/to/macro_data_outofsample.csv \
        --out    submission/positions.csv

By default, runs the GOAT Model (Option B: Kristofer Stretch Stop + Sector-Neutral XS Momentum + 10% Deadband).
Pass --model NOT_FINALMODEL_eemil to run Eemil's exploratory 3-sleeve candidate.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl

from src.goat_model import DATE, goat_positions
import src.NOT_FINALMODEL_eemil as NOT_FINALMODEL_eemil


def load(paths, closes_only=False):
    frames = []
    for p in paths:
        with open(p) as fh:
            sep = ";" if ";" in fh.readline() else ","
        df = pl.read_csv(p, separator=sep, try_parse_dates=True)
        df = df.rename({c: DATE for c in df.columns if c.lower() in ("asofdate", "date", "dates")})
        if df.schema[DATE] == pl.String:
            df = df.with_columns(pl.coalesce(pl.col(DATE).str.to_date("%d.%m.%Y", strict=False),
                                             pl.col(DATE).str.to_date("%Y-%m-%d", strict=False)).alias(DATE))
        df = df.with_columns(pl.col(DATE).cast(pl.Date), pl.exclude(DATE).cast(pl.Float64, strict=False))
        df = df.select([c for c in df.columns if c == DATE or df[c].null_count() < len(df)])  # drop empty/junk columns
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
    ap.add_argument("--prices", nargs="+", required=True, help="Path to in-sample and optional out-of-sample prices CSV")
    ap.add_argument("--macro", nargs="*", default=[], help="Path to in-sample and optional out-of-sample macro CSV")
    ap.add_argument("--out", default=str(ROOT / "submission" / "positions.csv"))
    ap.add_argument("--from-date", type=date.fromisoformat, default=None)
    ap.add_argument("--model", choices=["goat", "NOT_FINALMODEL_eemil"], default="goat",
                    help="Model architecture: 'goat' (Primary GOAT Model) or 'NOT_FINALMODEL_eemil' (Eemil candidate)")
    a = ap.parse_args()

    prices = load(a.prices, closes_only=True)

    if a.model == "goat":
        print("Executing Primary Model: The GOAT Model (Dual-Engine + 10% Deadband)...")
        pos = goat_positions(prices)
    else:
        print("Executing Exploratory Candidate: NOT_FINALMODEL_eemil (Trend + CPI + Twins)...")
        if not a.macro:
            sys.exit("Error: --macro is required for NOT_FINALMODEL_eemil")
        macro = load(a.macro)
        pos = NOT_FINALMODEL_eemil.final_positions(prices, macro)

    assert pos.columns == prices.columns and len(pos) == len(prices)
    assert pos.null_count().sum_horizontal()[0] == 0
    if a.from_date:
        pos = pos.filter(pl.col(DATE) >= a.from_date)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos.write_csv(out_path)
    print(f"SUCCESS: {a.out} generated ({len(pos)} rows x {pos.width - 1} assets, {pos[DATE][0]} to {pos[DATE][-1]})")


if __name__ == "__main__":
    main()
