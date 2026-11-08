# lynx-hackathon-2026

Code and research repository for the Stockholm Lynx Hackathon (Nov 14).

**Team:** Jaakko Pakarinen, Eemil Ylänikkilä, Erik Joonas Salumäe, Kristofer Vaiksaar

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
