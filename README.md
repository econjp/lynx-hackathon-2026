# lynx-hackathon-2026

Code and research repository for the Stockholm Lynx Hackathon (Nov 14).

**Team:** Jaakko Pakarinen, Eemil Ylä-Nikkilä, Erik Joonas Salumäe, Kristofer Vaiksaar

---

## 🚀 Live Research Dashboard (Hack Day Nov 14)

All strategy models, data analysis, and teammate idea backtests are verified using Lynx's official evaluator on `data/close_prices_insample.csv`.

👉 **[Open Full Strategy Pitch & Tables (research/jaakko/pitch/README.md)](./research/jaakko/pitch/README.md)**  
👉 **[Open Complete Findings Summary (research/jaakko/FINDINGS.md)](./research/jaakko/FINDINGS.md)**  
👉 **[Open Teammate Verification Report (research/jaakko/pitch/TEAM_CHECKS.md)](./research/jaakko/pitch/TEAM_CHECKS.md)**

### Key Results at a Glance:
1. **Core Trend Pick (EWMAC slow + 12m momentum)**:
   - Positive across all three 7-year regimes: **1.17** (1995–02) / **0.73** (2002–09) / **0.40** (2009–16), full-period **0.80–0.83**.
   - Holding time ~65 days (safely avoids 2-day lag and slippage penalties). Passes causality leak test.
2. **Kristofer's Model (EMA 10/100 + Stretch Stop)**:
   - Scored **0.993 Adjusted Sharpe** on 10 stocks+bonds.
   - The trailing stop adds **+0.07 Sharpe** and cuts tail risk.
   - Low correlation with trend (**0.496**). **50/50 Blend pushes full Sharpe to 1.02**!
3. **Cross-Sectional Momentum**:
   - Ranked across all 27 assets: 0.80 full, but collapses to **0.26** recently (0.87 correlated with trend).
   - Ranked within sectors: collapses to **0.11** recently (3m and 6m are negative).
   - *Verdict*: Use as a secondary 20–30% add-on sleeve, not the primary engine.
4. **Macro Lead (Eemil & Jake Pooled)**:
   - Rising CPI predicts falling bonds over 20–60 days (pooled $t = -2.9$, verified across all 5 moons). Best non-trend orthogonal signal.

---

## Instructions & Materials
- [lynx_materials/lynx-hackathon-instructions.pdf](./lynx_materials/lynx-hackathon-instructions.pdf)
- [lynx_materials/email_4_11.txt](./lynx_materials/email_4_11.txt)

---

## Setup
```bash
pip install polars numpy scipy matplotlib jupyter
```

---

## Quickstart
Run current team baseline model:
```bash
python src/baseline.py
```

---

## Repo Structure
- `lynx_materials/`: Original files provided by Lynx (PDF, email, example data, evaluator, and notebooks)
- `src/`: Shared production code (`baseline.py` model and `utils.py` helpers)
- `research/`: Individual sandbox folders for each with `template.ipynb` (DL own model: Thursday)
- `NOTES.md`: Questions to ask Lynx + strategy ideas from calls
- `EXPERIMENTS.md`: Leaderboard; tested models and scores
- `BACKLOG.md`: Ideas to try and timeline

---

## Key Rules from Lynx
- **2-day lag:** Orders execute at close T+2. Fast day trading doesn't work; focus on multi-week trends.
- **Cash units:** Return is units * price delta ($P\&L = X \cdot \Delta P$). Positions must be sized by price volatility ($\sigma(\Delta P)$).
- **Slippage:** Deducted as $0.02 \times \sigma(\Delta P)$ per traded unit. Keep average holding time $\ge 15$ days to avoid burning alpha.
- **Scoring:** Adjusted Sharpe ratio (penalizes excess kurtosis quadratically). Smooth signal beats peaky binary style signals.
- **No lookahead:** Submissions must pass truncation causality check.
- **No external data:** Only the CSV given on Saturday can be used.
