# FINAL OOS PERFORMANCE: BOMBA-GOAT @ Lynx

How the submitted 50/50 BOMBASTRAT + GOAT book did on Lynx's out-of-sample data (2017-01 to 2026-07, two paths that split in 2021).

| file | what |
|---|---|
| `FINAL OOS PERFORMANCE - bombagoat @ lynx.pdf` | the report: plots and analysis, code hidden |
| `FINAL OOS PERFORMANCE - bombagoat @ lynx.ipynb` | the same report with code, rerunnable |
| `oos_lib.py` | helpers: exact split of the book into its pieces, P&L by Lynx's `evaluation.py` rules |

**Headline (Lynx adjusted Sharpe):** in-sample 1.15 · OOS 2017-2026 **0.76 (path 1) / 0.69 (path 2)** · shared 2017-2020: 0.52 · after the split: 0.84 / 0.83.

Inputs: `data/*_outofsample{1,2}.csv` (Lynx's release) and `SUBMISSION/positions_outofsample{1,2}.csv` (what we handed in, which `model.py` reproduces exactly).
