# lynx-hackathon-2026

Stockholm Lynx Hackathon. **Team:** Jaakko Pakarinen, Eemil Ylä-Nikkilä, Erik Joonas Salumäe, Kristofer Vaiksaar

| folder | what |
|---|---|
| **[`SUBMISSION/`](./SUBMISSION/)** | the final submission: model, leak test, positions. Start here |
| [`WORK_DUMP/`](./WORK_DUMP/) | all research, earlier models, audits and notes |
| `data/` | the in-sample files from Lynx |
| `lynx_materials/` | Lynx's instructions and evaluator |

**Final model:** 50% BOMBASTRAT (trend + inflation + twins, market neutral) + 50% GOAT (stop-managed long in stocks and bonds + cross-sectional momentum), one portfolio.

```bash
pip install polars pandas numpy pyarrow matplotlib
python SUBMISSION/model.py --prices data/close_prices_insample.csv <oos_prices.csv> \
                           --macro  data/macro_data_insample.csv  <oos_macro.csv> --out SUBMISSION/positions.csv
python SUBMISSION/check.py SUBMISSION/positions.csv
```
