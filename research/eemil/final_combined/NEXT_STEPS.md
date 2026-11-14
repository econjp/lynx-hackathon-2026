# Final combined model: what is here and what is next

## What is here
| file | what |
|---|---|
| `src/final_model.py` | the model: `final_positions(prices, macro)`. Reuses Jaakko's `signals.py`, `risk.py`, `portfolio.py` |
| `final_model.ipynb` | plots, performance table, validation, verdict. Start here |
| `final_validation.py` | computes every number. Writes `FINAL_RESULTS.md` and `final_sleeves.png` (about 10 seconds) |
| `FINAL_RESULTS.md` | the tables as markdown |
| `engines_bc.py`, `ENGINES_BC.md` | Engine B (Kristofer stop) and Engine C (sector-neutral XS momentum): alone, combined, vs plain holding, mirror-world test, parameter neighbours, leak test, bootstrap |
| `submission/run_final.py` | in-sample + out-of-sample CSVs in, positions CSV out, one command |
| `submission/positions_insample.csv` | final model positions, full in-sample period, Lynx format |

```bash
python research/eemil/final_combined/final_validation.py
python submission/run_final.py --prices data/close_prices_insample.csv <oos_prices.csv> \
                               --macro  data/macro_data_insample.csv  <oos_macro.csv> \
                               --out    submission/positions.csv
```
`run_final.py` was tested by splitting the in-sample files in two at 2013: the stacked run gives exactly the
same positions as the full run, also from the Open/High/Low/Close file, and a run on the first part alone
matches the full run on every shared day.

## When Lynx releases the out-of-sample data
1. Run the command above with the new files as the second argument of `--prices` and `--macro`.
2. Check the printed date range and that there is no macro warning.
3. Do not change any parameter after seeing the new data.

## Open questions, in order of importance
1. **The last era is weak for every sleeve.** See table 1 in the notebook. The model's edge in 2009-2016
   comes mostly from the twins sleeve. Decide how to say this in the pitch.
2. **The bond inflation rule earns in one era only** (table 4, last rows). The stock rule is steadier.
   Dropping the bond rule now would be a choice made after seeing all three eras, so it stays in. Worth
   checking per moon before the pitch.
3. **Twins carry a lot of the gain.** One of the three pairs (antimatter / aerogel) sits right at the 0.8
   correlation line, so the rule switches it on and off. Check the result with that pair always on and
   always off; if the score moves a lot, say so.
4. **Kristofer's stop overlay** (notebook section 11) is not in the submitted model. It is a hard on/off
   cut, has not been leak-tested, and its gain is in the first era. Needs a proper check before it goes in.
5. **Risk caps cost a little in-sample** (row "final, no risk caps"). Kept for safety. The bond cap level
   (2 bets) was fixed up front and has not been varied.
6. **Sleeve weights are equal by rule.** No weight was tuned. If someone wants to tune them, use 1997-2009 only.
7. **Scoring window.** All rows are scored from the first day every sleeve holds a position (mid 1997),
   so the first era is shorter than in Jaakko's tables and the numbers differ slightly from his.

## Pitch line
Trend for what trends, inflation for what cares about inflation, twin spreads for what moves together.
Three separate bets (P&L correlations near zero), one portfolio, no long tilt.
