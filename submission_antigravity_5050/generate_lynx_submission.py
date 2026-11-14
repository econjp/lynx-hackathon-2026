"""
generate_lynx_submission.py - Official 3-Stage Lynx Submission Pipeline
======================================================================
Stockholm Lynx Hackathon 2026
Team: BOMBAGOAT (50/50 BOMBASTRAT + GOAT Model Ensemble)

Automates the exact submission protocol required by the Lynx portal:
  1. Finds prices and macro for each stage:
     - 'insample'
     - 'outofsample1'
     - 'outofsample2'
  2. Runs the 50/50 BOMBAGOAT ensemble independently on each full-history dataset.
  3. Outputs the exact filenames required by Lynx:
     - <team>_insample.csv
     - <team>_outofsample1.csv
     - <team>_outofsample2.csv
  4. Automatically verifies Lynx Lookahead / Causality:
     Compares in-sample rows across all 3 files to guarantee 0.00e+00 difference!

Usage:
    # Run right now for in-sample:
    python generate_lynx_submission.py --team team06

    # Run when OOS files drop into data folder:
    python generate_lynx_submission.py --team team06 --data-dir ../data
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lynx_materials"))
sys.path.insert(0, str(REPO_ROOT / "research" / "jaakko"))

import numpy as np
import polars as pl

from src.goat_model import DATE, goat_positions
from BOMBASTRAT.bombastrat import positions as bomba_positions


def load_file(path: Path) -> pl.DataFrame:
    """Loads a Lynx CSV file cleanly with robust separator and date parsing."""
    with open(path, "r", encoding="utf-8-sig") as fh:
        sep = ";" if ";" in fh.readline() else ","
    df = pl.read_csv(path, separator=sep, try_parse_dates=True)
    df = df.rename({c: DATE for c in df.columns if c.lower() in ("asofdate", "date", "dates")})
    if df.schema[DATE] == pl.String:
        df = df.with_columns(
            pl.coalesce(
                pl.col(DATE).str.to_date("%d.%m.%Y", strict=False),
                pl.col(DATE).str.to_date("%Y-%m-%d", strict=False),
            ).alias(DATE)
        )
    df = df.with_columns(
        pl.col(DATE).cast(pl.Date),
        pl.exclude(DATE).cast(pl.Float64, strict=False),
    )
    # Drop empty/junk columns
    df = df.select([c for c in df.columns if c == DATE or df[c].null_count() < len(df)])
    # Closes only if OHLC file
    if any(c.endswith("|Close") for c in df.columns):
        df = df.select(
            DATE,
            *[pl.col(c).alias(c.split("|")[0]) for c in df.columns if c.endswith("|Close")],
        )
    return df.sort(DATE)


def run_bombagoat_model(prices: pl.DataFrame, macro: pl.DataFrame | None) -> pl.DataFrame:
    """Executes the 50/50 Multi-Strategy BOMBAGOAT ensemble."""
    pos_goat = goat_positions(prices)
    if macro is None:
        print("    [Notice] No macro data available. Allocating 100% to GOAT model.")
        return pos_goat
    pos_bomba = bomba_positions(prices, macro)
    cols = [c for c in pos_goat.columns if c != DATE]
    pos_blend = pos_goat.select(DATE).with_columns([
        (0.50 * pos_goat[c] + 0.50 * pos_bomba[c]).alias(c) for c in cols
    ])
    return pos_blend


def find_dataset(data_dir: Path, stage: str, prefix: str) -> Path | None:
    """Finds either close_prices_<stage>.csv or prices_<stage>.csv or macro_data_<stage>.csv."""
    candidates = [
        data_dir / f"{prefix}_{stage}.csv",
        data_dir / f"{prefix}_{stage}.CSV",
        data_dir / f"close_{prefix}_{stage}.csv",
        data_dir / f"{stage}_{prefix}.csv",
    ]
    if prefix == "prices":
        candidates.insert(0, data_dir / f"close_prices_{stage}.csv")
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Official Lynx 3-File Submission Generator.")
    parser.add_argument("--team", default="team06", help="Team name prefix for submission files (default: team06)")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"), help="Directory containing the datasets")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent), help="Directory to save submission CSVs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print(f"LYNX HACKATHON 2026: OFFICIAL SUBMISSION GENERATOR - TEAM '{args.team}'")
    print("Strategy: BOMBAGOAT (50/50 BOMBASTRAT + GOAT Multi-Strategy Ensemble)")
    print("================================================================================")
    print(f"Data directory:   {data_dir}")
    print(f"Output directory: {out_dir}\n")

    stages = ["insample", "outofsample1", "outofsample2"]
    results: dict[str, pl.DataFrame] = {}
    output_files: dict[str, Path] = {}

    for stage in stages:
        px_path = find_dataset(data_dir, stage, "prices")
        mac_path = find_dataset(data_dir, stage, "macro_data")

        if not px_path:
            print(f"[*] Stage '{stage}': Price file not found in {data_dir} (waiting for release)...")
            continue

        print(f"[+] Processing Stage: '{stage}'")
        print(f"    Prices: {px_path.name}")
        prices = load_file(px_path)

        macro = None
        if mac_path:
            print(f"    Macro:  {mac_path.name}")
            macro = load_file(mac_path)

        pos = run_bombagoat_model(prices, macro)

        # Asserts
        assert pos.columns == prices.columns, f"Columns mismatch in {stage}!"
        assert len(pos) == len(prices), f"Row count mismatch in {stage}!"
        assert pos.null_count().sum_horizontal()[0] == 0, f"Found nulls in {stage} positions!"

        out_file = out_dir / f"{args.team}_{stage}.csv"
        pos.write_csv(out_file)
        results[stage] = pos
        output_files[stage] = out_file

        print(f"    --> Successfully wrote: {out_file.name} ({len(pos)} rows x {pos.width} cols)")

    # Perform Lynx Causality Truncation Check across available stages
    if "insample" in results and len(results) > 1:
        print("\n--------------------------------------------------------------------------------")
        print("VERIFYING LYNX ZERO-LOOKAHEAD RULE ACROSS GENERATED STAGES:")
        insample_df = results["insample"]
        insample_dates = insample_df[DATE]

        for stage, df in results.items():
            if stage == "insample":
                continue
            # Slice OOS file to matching in-sample dates
            oos_sliced = df.filter(pl.col(DATE).is_in(insample_dates))
            cols = [c for c in insample_df.columns if c != DATE]
            
            diffs = []
            for c in cols:
                max_d = (insample_df[c] - oos_sliced[c]).abs().max()
                diffs.append(max_d)
            max_diff = max(diffs)
            
            if max_diff == 0.0:
                print(f"  [PASS] 'insample' vs '{stage}': Max diff = 0.00e+00 (PERFECT ZERO LOOKAHEAD!)")
            else:
                print(f"  [WARN] 'insample' vs '{stage}': Max diff = {max_diff:.4e}")
        print("--------------------------------------------------------------------------------")

    print("\n================================================================================")
    print("SUBMISSION SUMMARY:")
    for stage, fpath in output_files.items():
        print(f"  * {fpath.name}  ({fpath.stat().st_size:,} bytes)")
    print("================================================================================")


if __name__ == "__main__":
    main()
