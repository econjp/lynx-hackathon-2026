# Kristofer's research

Open `template.ipynb` to start testing your models.

Deadline for individual models: Thursday night (23:59).

---

## Independent Backtest: EMA 10/100 Long-Only + Stretch Stop

Evaluated on `data/close_prices_insample.csv` (1995–2016, 27 assets) using Lynx's official `evaluation.py` (2-day lag, $0.02 \times \sigma$ slippage).

### K1. Adjusted Sharpe Across Three Eras (Scored from day 400)

| Model | 1995–2002 | 2002–2009 | 2009–2016 | Full Period | Nominal Sharpe | Kurtosis | Holding (days) | Bias |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kristofer: EMA 10/100 long-only + stop (10 stocks+bonds)** | **0.90** | **1.02** | **1.07** | **1.00** | **1.23** | **5.22** | **47.0** | **1.00** |
| Same, without stop (pure EMA 10/100) | 0.80 | 0.99 | 1.11 | 0.94 | 1.19 | 5.66 | 61.8 | 1.00 |
| **Benchmark: Always long 1/vol (same 10 assets)** | 0.86 | 0.82 | 1.00 | 0.97 | 1.20 | 5.33 | 273.0 | 1.00 |
| EMA 10/100 long/short (same 10 assets) | 0.64 | 0.76 | 0.60 | 0.67 | 0.72 | 5.80 | 49.9 | 0.30 |
| Kristofer rule on all 27 assets | 0.77 | 0.94 | 0.60 | 0.72 | 0.78 | 4.82 | 37.6 | 1.00 |
| Benchmark: Always long 1/vol (all 27 assets) | 0.02 | 0.72 | 0.61 | 0.44 | 0.45 | 4.08 | 270.3 | 1.00 |

### K2. Asset-by-Asset (Rule vs Passive Buy-and-Hold)

The rule beats passive buy-and-hold on 4 of 10 assets (Ganymede stocks +0.18, Callisto stocks +0.17, Europa stocks +0.03, Io bonds +0.04), and is slightly below on Ganymede/Titan bonds.

### Quick Quant Notes & Ensemble Potential:
1. **Backtest confirmed**: Your code has no lookahead and scores **0.993 Adjusted Sharpe** on the official evaluator.
2. **The 3-close stop works**: The stop adds **+0.06 to +0.07 Sharpe** and trims kurtosis from 5.66 to 5.22.
3. **Caveat on drift**: Passive buy-and-hold on these 10 assets scores 0.97 because stocks and bonds were in a 20-year structural bull market (1995–2016). The model is 100% net long (`bias = 1.0`), so it relies on the bull market continuing.
4. **Low correlation with trend (0.496)**: Because your model is long stocks/bonds with an active stop, its correlation with Jake's multi-speed trend is only **0.496**.
5. **Ensemble result**: Blending 50% Jake Trend + 50% Kristofer reaches **1.02 Adjusted Sharpe (1.28 nominal Sharpe)** across all three periods! Full details in `research/jaakko/pitch/TEAM_CHECKS.md`.
