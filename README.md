# lynx-hackathon-2026

Stockholm Lynx Hackathon (Nov 2026). #2 on raw out-of-sample Sharpe (0.958).  
**Team:** Jaakko Pakarinen, Eemil Ylä-Nikkilä, Erik Joonas Salumäe, Kristofer Vaiksaar

| folder | what |
|---|---|
| **[`SUBMISSION/`](./SUBMISSION/)** | final model, leak test, positions |
| **[`FINAL_OOS_PERFORMANCE/`](./FINAL_OOS_PERFORMANCE/)** | out-of-sample performance report (0.76 / 0.69 adj. Sharpe) |
| **[`FINAL_BACKTEST.ipynb`](./FINAL_BACKTEST.ipynb)** | end-to-end backtest notebook |
| [`WORK_DUMP/`](./WORK_DUMP/) | research, earlier models and notes |
| `data/` | in-sample prices and macro |
| `lynx_materials/` | evaluator and starter notebooks |

**Model:** 50% BOMBASTRAT (trend + inflation + twins) + 50% GOAT (ratchet stop equity/bond long + cross-sectional momentum).

```bash
pip install polars pandas numpy pyarrow matplotlib
python SUBMISSION/model.py --prices data/close_prices_insample.csv <oos_prices.csv> \
                           --macro  data/macro_data_insample.csv  <oos_macro.csv> --out SUBMISSION/positions.csv
python SUBMISSION/check.py SUBMISSION/positions.csv
```
