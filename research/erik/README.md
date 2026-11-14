# Example Price Strategy

A standalone Python implementation of the consolidation/expansion strategy using **only the included `data/example_prices.csv`**. All comments, documentation and reports are in English. No macro data, neighboring project or supplementary price dataset is required.

Open [the PDF report](outputs/Strategy_Report.pdf), [the offline interactive dashboard](outputs/Strategy_Dashboard.html), or [the output guide](OUTPUT_GUIDE.md).

## Dataset and split

The supplied file has 5,956 daily observations for `diamonds`, `butter`, `cars`, `cucumber` and `bananas`. The first column, `dates`, is parsed as the internal `AsOfDate` index. Prices are finite and positive, and dates are sorted and unique.

| Slice | Dates | Observations |
| --- | --- | ---: |
| Training, first 70% | 2000-01-03 through 2016-02-12 | 4,169 |
| Historical holdout, final 30% | 2016-02-15 through 2023-01-11 | 1,787 |

This is a chronological holdout within the supplied file, not a live forward record or a separate future data feed. Parameters remain fixed for the holdout. Training-only parameter diagnostics do not replace the base model with a sweep winner. Signal state, queued orders and any positions continue across the split.

## Run

Use Python 3.12 or newer. Run from this folder:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_analysis.py
```

The dataset and scoring-reference module are included. Default paths are project-local. An alternate split can be requested with `--train-fraction`; changing it creates a different research experiment and should not be confused with a fresh independent test after examining the file.

```bash
python run_analysis.py --prices data/example_prices.csv --train-fraction 0.70
```

`--no-report` writes numerical results only. `--skip-sensitivity` omits the training-only experiment tables. `--report-only` regenerates reports from this project's own local analysis cache. A normal full run recreates all report assets from the included input.

## Trading rules

At each new month, compare the last three completed months with the preceding three. A valid consolidation has a contained closing range, repeated crossings of interior levels, and sufficient closes in both range halves. Freeze its boundaries for three months.

Long entries require a close above both last completed quarter closes and above the frozen upper boundary. Shorts mirror this below the lower boundary. After an initial breakout trade, a confirmed monthly-reference pullback can provide a same-direction re-entry, subject to the quarter filter, setup validity and cooldown.

An order decided at close t fills at close t+1 and first earns the price change ending at t+2. The initial stop is three signal-date daily volatility units from the actual fill. Three favorable close-to-close steps allow ratcheting; an adverse close suspends ratchets until three favorable steps reclaim the stored reference. Stops never move adversely. A stop breach requests next-close execution rather than a guaranteed fill at the stop level.

Sizing uses trailing price-change volatility and fixed independent-idea risk sleeves. Signed correlation groups share a sleeve, and confirmed relative-strength divergence can redistribute risk inside a group. These modifiers remain conditional: the dataset must actually supply the necessary relationships. No asset classes are inferred from the series names, and no strategy-P&L volatility scaling is applied.

`parameters.json` documents the base values. The report defines operational choices, including setup replacement while flat, re-entry cooldown, quantity resizes, terminal orders and initial-stop timing.

## Score and units

- Primary net P&L uses a two-row earning-position lag and costs of 0.02 times lagged daily price-change standard deviation per unit changed.
- Net Sharpe annualizes with sqrt(260). Adjusted Sharpe uses unbiased skewness and excess kurtosis.
- `reference_evaluator.py` is included for direct accounting parity checks. The alternate constant-cost formula is reported separately as a sensitivity calculation.
- P&L uses a fixed illustrative reference capital of 100,000 units. These are example-series units, not verified investable currency returns. Position sizes do not compound with equity.
- All flat days are retained. Open positions are marked to the final available close, and no terminal liquidation is invented.
- Use `positions_full.csv` with the complete source history to reproduce the score; a holdout-only slice omits earlier lagged boundary decisions.

## Outputs

| Output | Contents |
| --- | --- |
| `Strategy_Report.pdf` | English strategy, parameters, training/holdout results, uncertainty, diagnostics and limitations. |
| `Strategy_Dashboard.html` | Offline period, rolling-metric, instrument, trade and pair-correlation viewers. |
| `daily_train.csv`, `daily_forward.csv`, `daily_full.csv` | Daily P&L, costs, drawdown, exposure and 20/63/126/260-row rolling metrics. |
| `positions_full.csv` | Finite decision-date quantities for every source date and series. |
| `trades.csv`, `events.csv`, `setups.csv` | Trade attribution, execution/trailing events and monthly consolidation statistics. |
| `instrument_diagnostics.csv.gz` | Daily price, volatility, references, stops, quantities and signal states. |
| `rolling_correlations.csv.gz` | All ten distinct pair correlations using trailing 100-row price changes. |
| `parameter_sensitivity_train.csv`, `ablation_train.csv` | Training-only one-at-a-time parameter and component experiments. |
| `stress.csv` | Cost and fixed-order delay sensitivity by slice. |
| `summary.json`, `split_definition.json`, `run_manifest.json` | Machine-readable scores, fixed split and source/configuration fingerprints. |
| `validation_checks.json`, `report_quality_checks.json` | Accounting/causality checks and the scope of presentation checks. |

The Python tests cover stop behavior, setup timing, next-close fills, initial risk distance, earning and cost lag, correlation risk sharing, prefix invariance and scoring-reference parity. The real-data checks also reconcile the trade ledger and verify that the full history does not change any earlier training target. The dashboard controller is checked offline; browser rendering is not represented as verified.
