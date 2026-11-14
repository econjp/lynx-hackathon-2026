"""Analyze one supplied closing-price file with a fixed chronological holdout."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import pickle
import sys
import time
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from strategy import Parameters,prepare_features,run_strategy,trend_benchmark,validate_prices
from evaluation import adjusted_sharpe,attribute_trades,bootstrap_sharpe,curve_and_rolling,evaluate,instrument_metrics,performance,sharpe
PROJECT=Path(__file__).resolve().parent

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=True) + "\n")

def period_performance(ev, p, mask, trades):
    answer = performance(ev, p, mask, trades)
    answer["bootstrap"] = bootstrap_sharpe(ev.daily.loc[mask, "net_pnl"], annualization=p.annualization)
    return answer

def official_parity(positions, prices, p, official_path: Path) -> dict[str, Any]:
    import polars as pl
    spec = importlib.util.spec_from_file_location("lynx_official_2026_runtime", official_path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    def frame(values):
        return pl.DataFrame({"AsOfDate": [x.date() for x in values.index],
                             **{c: values[c].tolist() for c in values.columns}})
    answer = official.calc_key_figures(frame(positions), frame(prices),
                                      key_figures=["returns", "sharpe", "adjusted_sharpe", "sharpe_1d_lag"])
    ours = evaluate(positions, prices, p)
    expected = answer["returns"]["returns"].to_numpy()
    np.testing.assert_allclose(ours.daily.net_pnl.to_numpy(), expected, rtol=1e-10, atol=1e-8, equal_nan=True)
    sr_diff = abs(sharpe(ours.daily.net_pnl) - answer["sharpe"])
    asr_diff = abs(adjusted_sharpe(ours.daily.net_pnl) - answer["adjusted_sharpe"])
    assert sr_diff < 1e-9 and asr_diff < 1e-9
    return dict(passed=True, max_pnl_difference=float(np.nanmax(np.abs(ours.daily.net_pnl.to_numpy() - expected))),
                sharpe_difference=sr_diff, adjusted_sharpe_difference=asr_diff,
                official_extra_day_lag_sharpe=float(answer["sharpe_1d_lag"]))

def causality_checks(train, full, p, train_result, full_result) -> list[dict[str, Any]]:
    rows = []
    np.testing.assert_array_equal(train_result.positions.to_numpy(), full_result.positions.loc[train.index].to_numpy())
    rows.append(dict(check="Training targets unchanged after appending all forward prices", passed=True,
                     compared_rows=len(train), max_difference=0.0))
    for cut in (len(train) // 2, len(train) + (len(full) - len(train)) // 2):
        prefix = run_strategy(full.iloc[:cut], p, diagnostics=False)
        np.testing.assert_array_equal(prefix.positions.to_numpy(), full_result.positions.iloc[:cut].to_numpy())
        rows.append(dict(check="Truncated-history target invariance", cut_date=full.index[cut - 1],
                         passed=True, compared_rows=cut, max_difference=0.0))
    cut = len(train)
    changed = full.copy()
    perturbation = 1 + .12 * np.sin(np.arange(len(full) - cut) * .77)[:, None]
    changed.iloc[cut:] = changed.iloc[cut:].to_numpy() * perturbation
    alternative = run_strategy(changed, p, diagnostics=False)
    np.testing.assert_array_equal(alternative.positions.iloc[:cut].to_numpy(), full_result.positions.iloc[:cut].to_numpy())
    rows.append(dict(check="Future-price perturbation invariance", cut_date=train.index[-1],
                     passed=True, compared_rows=cut, max_difference=0.0))
    for tid, group in full_result.diagnostics.loc[full_result.diagnostics.trade_id > 0].groupby("trade_id", sort=False):
        direction = int(np.sign(group.filled_quantity.iloc[0]))
        delta = np.diff(group.stop.dropna().to_numpy()) * direction
        assert (delta >= -1e-10).all(), f"Stop moved adversely in trade {tid}."
    rows.append(dict(check="Trade-local stops never move in the adverse direction", passed=True))
    return rows

def save_outputs(b, output):
    write_json(output / "summary.json", b["metrics"])
    write_json(output / "data_audit.json", b["audit"])
    write_json(output / "run_manifest.json", b["manifest"])
    write_json(output / "validation_checks.json", b["checks"])
    write_json(output / "pdf_formula_comparison.json", b["pdf_alternative"])
    for label, d in b["daily"].items():
        d.to_csv(output / f"daily_{label}.csv", index_label="AsOfDate", float_format="%.10g")
    for label, d in b["instruments"].items():
        d.to_csv(output / f"instruments_{label}.csv", index=False)
    for name in ("yearly", "monthly", "stress", "events", "setups", "trades"):
        frame = b[name]
        frame.to_csv(output / f"{name}.csv", index=name == "monthly")
    b["positions"].to_csv(output / "positions_full.csv", index_label="AsOfDate", float_format="%.12g")
    b["positions"].loc[b["positions"].index <= b["train_end"]].to_csv(output / "positions_train.csv", index_label="AsOfDate", float_format="%.12g")
    b["positions"].loc[b["positions"].index > b["train_end"]].to_csv(output / "positions_forward.csv", index_label="AsOfDate", float_format="%.12g")
    b["net_by_instrument"].to_csv(output / "daily_pnl_by_instrument.csv", index_label="AsOfDate", float_format="%.10g")
    b["diagnostics"].to_csv(output / "instrument_diagnostics.csv.gz", index=False, float_format="%.10g", compression="gzip")
    b["rolling_correlations"].to_csv(output / "rolling_correlations.csv.gz", index_label="AsOfDate", float_format="%.6g", compression="gzip")
    (output / "internal").mkdir(exist_ok=True)
    with (output / "internal/analysis.pkl").open("wb") as handle:
        pickle.dump(b, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_prices(path: Path) -> tuple[pd.DataFrame, str]:
    raw = pd.read_csv(path)
    date_name = raw.columns[0]
    if date_name not in ("dates", "AsOfDate", "date", "Date"):
        raise ValueError("The first column must identify the observation date.")
    dates = pd.to_datetime(raw.pop(date_name), errors="raise")
    prices = raw.apply(pd.to_numeric, errors="raise").astype(float)
    prices.index = pd.DatetimeIndex(dates, name="AsOfDate")
    validate_prices(prices)
    return prices, date_name


def training_experiments(train: pd.DataFrame, p: Parameters):
    variants = [("base", "base", "frozen")]
    settings = [("initial_stop_sigma", [2., 4.]), ("trail_steps", [2, 4]),
                ("min_overlap", [2., 4.]), ("min_half_fraction", [.2, .4]),
                ("vol_window", [40, 60]), ("setup_valid_months", [2, 6]),
                ("corr_threshold", [.6, .8]), ("rebalance_days", [1, 10]),
                ("cooldown_days", [0, 10]), ("consolidation_months", [2, 4])]
    for key, values in settings:
        variants.extend((f"{key}={value}", key, value) for value in values)
    ablations = [("No quarter filter", "quarter_filter", False),
                 ("No inside-range requirement", "require_inside", False),
                 ("No pullback re-entry", "pullback_entry", False),
                 ("No divergence preference", "divergence_preference", 0.),
                 ("No correlation grouping", "correlation_budgeting", False),
                 ("No trailing suspension", "suspend_trailing", False)]
    feature_fields = ("consolidation_months", "reference_months", "overlap_levels", "min_overlap",
                      "min_half_fraction", "require_inside", "vol_window", "corr_window", "entry_window")
    cache = {}
    split = int(len(train) * 2 / 3)
    tables = []
    for batch in (variants, ablations):
        rows = []
        for label, key, value in batch:
            q = p if key == "base" else replace(p, **{key: value})
            signature = tuple(getattr(q, field) for field in feature_fields)
            if signature not in cache:
                cache[signature] = prepare_features(train, q)
            result = run_strategy(train, q, diagnostics=False, features=cache[signature])
            ev = evaluate(result.positions, train, q)
            metrics = performance(ev, q)
            rows.append(dict(variant=label, parameter=key, value=value,
                             net_sharpe=metrics["net_sharpe"], adjusted_sharpe=metrics["adjusted_sharpe"],
                             early_train_sharpe=sharpe(ev.daily.net_pnl.iloc[:split]),
                             late_train_sharpe=sharpe(ev.daily.net_pnl.iloc[split:]),
                             late_train_adjusted_sharpe=adjusted_sharpe(ev.daily.net_pnl.iloc[split:]),
                             net_pnl=metrics["net_pnl"], max_drawdown_reference_pct=metrics["max_drawdown_reference_pct"],
                             trades=len(result.trades), invested_fraction=metrics["invested_day_fraction"],
                             qualified_setups=int(result.setups.qualifies.sum())))
            print(f"Training diagnostic: {label}; Sharpe={metrics['net_sharpe']:.3f}", flush=True)
        tables.append(pd.DataFrame(rows))
    return tables


def tables_for_periods(ev, benchmark, trades, p, train_end):
    masks = {"train": ev.daily.index <= train_end, "forward": ev.daily.index > train_end,
             "full": np.ones(len(ev.daily), dtype=bool)}
    metrics, daily, instruments = {}, {}, {}
    for name, mask in masks.items():
        metrics[name] = period_performance(ev, p, mask, trades)
        metrics[name]["benchmark"] = performance(benchmark, p, mask)
        daily[name] = curve_and_rolling(ev.daily.loc[mask], p, benchmark.daily.net_pnl)
        daily[name]["benchmark_net_pnl"] = benchmark.daily.loc[mask, "net_pnl"]
        daily[name]["benchmark_cumulative_pnl"] = daily[name].benchmark_net_pnl.fillna(0).cumsum()
        instruments[name] = instrument_metrics(ev, p, mask, trades)
    yearly = []
    for year, group in ev.daily.groupby(ev.daily.index.year):
        metric = performance(ev, p, ev.daily.index.isin(group.index))
        label = "Training" if group.index[-1] <= train_end else "Forward" if group.index[0] > train_end else "Split year"
        metric.update(year=int(year), period=label)
        yearly.append(metric)
    monthly = pd.DataFrame({"net_pnl": ev.daily.net_pnl.resample("ME").sum(min_count=1),
                            "gross_pnl": ev.daily.gross_pnl.resample("ME").sum(min_count=1),
                            "costs": ev.daily.transaction_cost.resample("ME").sum(min_count=1)})
    monthly["pnl_reference_pct"] = monthly.net_pnl / p.reference_capital * 100
    return metrics, daily, instruments, pd.DataFrame(yearly), monthly


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        with (output / "internal/analysis.pkl").open("rb") as handle:
            bundle = pickle.load(handle)
        bundle["manifest"]["generated_for_date"] = args.report_date
        write_json(output / "run_manifest.json", bundle["manifest"])
        from reporting import build_reports
        build_reports(bundle, output)
        return
    started = time.monotonic()
    prices, input_date_column = load_prices(args.prices)
    cut = int(len(prices) * args.train_fraction)
    if not .5 <= args.train_fraction <= .9 or cut < 500 or len(prices) - cut < 300:
        raise ValueError("Use a training fraction from 0.5 to 0.9 and sufficient observations in both periods.")
    train = prices.iloc[:cut].copy()
    train_end = train.index[-1]
    p = Parameters(**json.loads(args.parameters.read_text()))
    param_hash = digest(args.parameters)
    code_hashes = {name: digest(PROJECT / name) for name in ("strategy.py", "evaluation.py")}
    audit = dict(source_file="data/" + args.prices.name,
                 price_source_count=1, source_rows=len(prices), instruments=len(prices.columns),
                 instrument_names=list(prices.columns), original_date_column=input_date_column,
                 internal_date_column="AsOfDate", training_fraction=args.train_fraction,
                 training_rows=cut, forward_rows=len(prices) - cut,
                 training_start=train.index[0], training_end=train_end,
                 forward_start=prices.index[cut], forward_end=prices.index[-1],
                 missing_prices=int(prices.isna().sum().sum()), duplicate_dates=int(prices.index.duplicated().sum()),
                 macro_data_used=False,
                 split_description=f"Training: {train.index[0].date()} to {train_end.date()}. "
                 f"Held-out historical slice: {prices.index[cut].date()} to {prices.index[-1].date()}. "
                 "Both periods are chronological slices of the one supplied price file.")
    write_json(output / "frozen_parameters.json", p.to_dict())
    write_json(output / "split_definition.json", audit)
    print(f"Single source: {args.prices.name}; {len(prices)} rows and {prices.shape[1]} series.", flush=True)
    print(f"Training rows: {cut}; fixed terminal holdout rows: {len(prices)-cut}.", flush=True)
    train_result = run_strategy(train, p)
    train_ev = evaluate(train_result.positions, train, p)
    train_trades = attribute_trades(train_result, train_ev, train_end)
    write_json(output / "training_checkpoint.json", dict(parameters_sha256=param_hash,
               selection_rule="Frozen base specification; parameter sweep winner not selected",
               metrics=performance(train_ev, p, trades=train_trades)))
    if args.skip_sensitivity:
        sensitivity, ablations = pd.DataFrame(), pd.DataFrame()
    else:
        sensitivity, ablations = training_experiments(train, p)
        sensitivity.to_csv(output / "parameter_sensitivity_train.csv", index=False)
        ablations.to_csv(output / "ablation_train.csv", index=False)
    if digest(args.parameters) != param_hash or any(digest(PROJECT / name) != value for name, value in code_hashes.items()):
        raise RuntimeError("The frozen specification changed during training analysis.")
    print("Training diagnostics complete. Evaluating the unchanged rules on the terminal holdout.", flush=True)
    result = run_strategy(prices, p)
    ev = evaluate(result.positions, prices, p)
    changes = prices.diff()
    n, m = prices.shape
    correlation = changes.rolling(p.corr_window, min_periods=p.corr_window).corr(pairwise=True).to_numpy().reshape(n, m, m)
    rolling_correlations = pd.DataFrame({f"{prices.columns[i]} | {prices.columns[j]}": correlation[:, i, j]
                                         for i in range(m) for j in range(i + 1, m)}, index=prices.index)
    prior_corr = np.zeros_like(correlation)
    prior_corr[1:] = np.nan_to_num(correlation[:-1], nan=0.)
    for j in range(m):
        prior_corr[:, j, j] = 1.
    prior_vol = changes.rolling(p.vol_window).std(ddof=1).shift(1).fillna(0).to_numpy()
    units = ev.earning_positions.fillna(0).to_numpy() * prior_vol
    variance = np.einsum("ni,nij,nj->n", units, prior_corr, units)
    ev.daily["exante_annual_vol_reference_pct"] = np.sqrt(np.maximum(variance, 0)) * np.sqrt(p.annualization) / p.reference_capital * 100
    trades = attribute_trades(result, ev, train_end)
    benchmark_positions = trend_benchmark(prices, p)
    benchmark = evaluate(benchmark_positions, prices, p)
    metrics, daily, by_instrument, yearly, monthly = tables_for_periods(ev, benchmark, trades, p, train_end)
    masks = {"train": prices.index <= train_end, "forward": prices.index > train_end}
    stress = []
    for lag in (2, 3, 4):
        for coefficient in (0., .02, .04, .06):
            stressed = evaluate(result.positions, prices, p, lag=lag, cost_coefficient=coefficient)
            for name, mask in masks.items():
                metric = performance(stressed, p, mask)
                stress.append(dict(period=name, position_lag=lag, cost_coefficient=coefficient,
                                   net_pnl=metric["net_pnl"], net_sharpe=metric["net_sharpe"],
                                   adjusted_sharpe=metric["adjusted_sharpe"],
                                   max_drawdown_reference_pct=metric["max_drawdown_reference_pct"]))
    alternate = evaluate(result.positions, prices, replace(p, annualization=252), cost_model="pdf_constant", cost_coefficient=.0002)
    alternate_metrics = {name: performance(alternate, replace(p, annualization=252), mask) for name, mask in masks.items()}
    checks = causality_checks(train, prices, p, train_result, result)
    parity = official_parity(result.positions, prices, p, PROJECT / "reference_evaluator.py")
    checks.append(dict(check="P&L and adjusted metrics match the included scoring reference", **parity))
    for name, mask in {"full": np.ones(n, dtype=bool), **masks}.items():
        column = "net_pnl" if name == "full" else name + "_net_pnl"
        difference = float(trades[column].sum() - ev.daily.loc[mask, "net_pnl"].sum())
        assert abs(difference) < 1e-6
        checks.append(dict(check=f"Trade ledger reconciles to {name} daily P&L", passed=True, difference=difference))
    assert np.isfinite(result.positions.to_numpy()).all()
    checks.append(dict(check="All decision-date targets, including final rows, are finite", passed=True))
    checks.append(dict(check="Only the supplied price dataset is used; no macro input", passed=True))
    manifest = dict(report_language="English", generated_for_date=args.report_date,
                    price_source="data/" + args.prices.name, price_source_count=1,
                    price_source_sha256=digest(args.prices), input_date_column=input_date_column,
                    training_fraction=args.train_fraction, parameters_sha256=param_hash,
                    frozen_core_code_sha256=code_hashes, scoring_reference="reference_evaluator.py",
                    scoring_reference_sha256=digest(PROJECT / "reference_evaluator.py"),
                    parameter_selection="Frozen base; training-only one-at-a-time experiments; no holdout tuning",
                    scope="Chronological historical holdout inside one supplied example-price file; not live forward trading",
                    accounting="Two-row earning lag; 0.02 volatility-scaled costs; sqrt(260); excess kurtosis adjustment",
                    package_versions={name: importlib.metadata.version(name) for name in
                                      ("numpy", "pandas", "matplotlib", "polars", "plotly", "reportlab")},
                    macro_data_used=False, elapsed_seconds=time.monotonic() - started)
    bundle = dict(parameters=p.to_dict(), audit=audit, manifest=manifest, metrics=metrics,
                  daily=daily, instruments=by_instrument, yearly=yearly, monthly=monthly,
                  sensitivity=sensitivity, ablations=ablations, stress=pd.DataFrame(stress),
                  pdf_alternative=alternate_metrics, checks=checks, prices=prices,
                  diagnostics=result.diagnostics, events=result.events, setups=result.setups,
                  trades=trades, positions=result.positions, net_by_instrument=ev.net_by_instrument,
                  cost_by_instrument=ev.costs_by_instrument, benchmark_positions=benchmark_positions,
                  train_end=train_end, rolling_correlations=rolling_correlations)
    save_outputs(bundle, output)
    print(json.dumps(json_safe({name: {key: metrics[name][key] for key in
                                     ("net_pnl", "net_sharpe", "adjusted_sharpe", "max_drawdown_reference_pct", "trades_entered")}
                              for name in ("train", "forward")}), indent=2), flush=True)
    if not args.no_report:
        from reporting import build_reports
        build_reports(bundle, output)
    print(f"Completed in {time.monotonic()-started:.1f} seconds.", flush=True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=PROJECT / "data/example_prices.csv")
    parser.add_argument("--train-fraction", type=float, default=.70)
    parser.add_argument("--parameters", type=Path, default=PROJECT / "parameters.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs")
    parser.add_argument("--report-date", default=datetime.now(ZoneInfo("Europe/Tallinn")).date().isoformat())
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Regenerate reports from this project's own local analysis cache.")
    return parser.parse_args()


if __name__ == "__main__":
    run(arguments())
