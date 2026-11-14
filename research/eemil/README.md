# Eemil's research: validation harness + a no-picking trend model

## Run
```bash
python research/eemil/validate.py [prices.csv]   # all team models side by side (~1 min)
python research/eemil/sweep.py [prices.csv]      # parameter sweeps for the eemil model, per block
```
With no argument both use the Lynx example data. At 09:00, pass the real CSV.

## Files
- `validate.py`: scores every model on the full file, 5 chronological blocks, +1 day lag, double
  costs, a truncation leak test on every row, four damaged copies of the data, and a paired
  block bootstrap (90% interval, probability of beating the team baseline).
- `model.py`: EWMAC z-scores (closed-form random-walk normalisation, nothing fitted) at speeds
  8/16/32/64, tanh, equal-weight blend, 1/vol sizing, ex-ante book risk scaling, 30% glide.
- `sweep.py`: one-at-a-time parameter sweeps, scored per block.

## Results on the example data (adjusted Sharpe)
| model | full | b1 | b2 | b3 | b4 | b5 | lag+1 | 2x cost | hold | ex. kurt | 90% interval | P(> team) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lynx50 | 0.69 | 0.24 | 0.98 | 0.26 | 1.11 | 0.81 | 0.64 | 0.60 | 15d | 0.28 | 0.34 to 1.04 | 0.11 |
| team40/80 | 0.86 | 0.06 | 0.91 | 1.10 | 1.10 | 1.09 | 0.79 | 0.78 | 15d | 0.58 | 0.51 to 1.17 | ref |
| jp_fast | 0.81 | 0.25 | 0.89 | 0.84 | 0.97 | 0.96 | 0.84 | 0.78 | 40d | 1.00 | 0.47 to 1.13 | 0.29 |
| jp_all | 0.67 | 0.12 | 0.69 | 0.66 | 0.72 | 0.98 | 0.67 | 0.65 | 54d | 0.99 | 0.31 to 0.99 | 0.06 |
| eemil | 0.67 | -0.08 | 0.82 | 0.70 | 0.76 | 0.98 | 0.67 | 0.64 | 47d | 0.50 | 0.30 to 1.01 | 0.04 |

Stress (adjusted Sharpe on damaged data: two assets start 40% late / 2% missing prints /
one 8-sigma level jump per asset-year / prices shifted through zero):

| model | late | holes | jumps | shift |
|---|---|---|---|---|
| lynx50 | 0.65 | 0.21 | 0.61 | 0.69 |
| team40/80 | 0.84 | 0.75 | 0.66 | 0.86 |
| jp_fast | 0.77 | 0.69 | 0.59 | 0.81 |
| jp_all | 0.71 | 0.61 | 0.52 | 0.67 |
| eemil | 0.67 | 0.61 | 0.55 | 0.67 |

All five pass the leak test.

## What this says
1. **The models are not statistically separable.** Every 90% interval is about +-0.35 wide and
   they all overlap. Leaderboard gaps under ~0.3 on this file are noise.
2. **Speed is the only parameter that matters.** Single-speed sweep: 4 -> 0.35, 8 -> 0.62,
   16 -> 0.82, 32 -> 0.62, 64 -> 0.32. The best speed is 16 in blocks 2-3, 8 in block 4 and
   32-64 in block 5, so the peak moves. The team baseline and jp_fast win because they sit near
   16/64 on this synthetic file; that will not necessarily carry to real data.
3. **Glide (alpha), vol span and risk window are flat** (0.64 to 0.67 across the whole grid).
   No point tuning them. Shorter risk window lowers kurtosis (0.31 at 60d vs 0.69 at 500d).
4. **Block 1 (2000-2004) is ~0 for every model.** Trend did not pay there at all.
5. **The team baseline leans on quick reaction**: it drops 0.86 -> 0.79 with one more day of
   lag and holds 14.9 days. jp_fast does not drop (0.81 -> 0.84) and holds 40 days.
6. **Missing prints break the Lynx baseline** (0.69 -> 0.21). The others lose about 0.1.
   Level jumps cost every model 0.15 to 0.25.

## Caveat on the eemil score
0.67 is with the speed grid fixed before looking at the sweep. Dropping the 64 speed gives 0.76,
but that choice would be made after seeing the data, so it is not the reported number.
