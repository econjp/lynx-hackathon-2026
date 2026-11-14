# All findings (Jake)

Real data = close_prices_insample.csv, 27 assets, 1995-2016. Scored with Lynx's evaluation.py.
Every number below comes from a script in `pitch/`. Rerun them to check.

| file | built by | what |
|---|---|---|
| pitch/PITCH.md | by hand | the signal pitch |
| pitch/TABLES.md | make_tables.py | signals per period, KPIs, stress tests, leak test, correlations |
| pitch/DATA.md | data_analysis.py | assets, correlations, trend hit rates, P&L by class, per year, macro, OHLC |
| pitch/TEAM_CHECKS.md | team_checks.py | Kristofer's idea, cross-sectional momentum, cross-check with Eemil |
| archive/prep_13_11/ | - | Friday prep on example + futures data |

## Signal
- time-series trend works in all 3 periods. pick: EWMAC slow + 12m momentum 50/50, 1.17 / 0.73 / 0.40 (T1)
- all trend signals are 0.7-0.9 correlated (T5)
- on/off signals and fast speeds are worst lately (T1)
- picked after seeing all periods (~10 tries), so true number is a bit lower

## Cross-sectional momentum (X1, X2)
- XS ranked across all 27: 0.80 full but 0.26 lately. 0.87 corr with TS trend, so it's mostly the same bet
  (bonds/stocks rise, commodities fall, ranking across classes ~ trend)
- real relative momentum (rank within each class): 0.60 full, 0.11 lately. 6m and 3m: ~0
- -> weaker than TS trend in the latest period. ok as a small add-on, not as the main signal

## Kristofer: EMA 10/100 long-only + stretch stop (K1, K2)
- his number is right: 1.00 on stocks+bonds
- but just holding those 10 assets long (1/vol) gives 0.97. the rule adds ~0.03
- per asset the rule beats holding in 4 of 10
- the 10 assets were picked after seeing results, and it's 100% net long
- -> it's a bet that bonds and stocks keep rising, not a timing edge.
  if the test period has rising rates, it loses. the stop itself adds ~0.06

## Data (DATA.md, TEAM_CHECKS.md E1-E3)
- data is identical in all repos (file checksums)
- few jumps. >6 sigma counts depend on the vol estimate: 56 (100d std) vs 29 (Eemil, EWMA 60) (E1)
- each moon has its own holidays, commodities follow titan. daily correlations are too low:
  bonds 0.41 daily vs 0.54 weekly, stocks 0.40 vs 0.57 (E2). use weekly for correlations.
  our covariance scaling uses daily changes -> slightly underestimates risk, worth testing weekly
- bonds and stocks on the same moon move opposite (~ -0.2)
- most commodities fall a lot over 1995-2016

## Macro (D7, D7b + Eemil)
- CPI up -> that moon's bonds down over the next 20-60 days. stable in all 5 moons and 3 periods (Eemil),
  pooled t -2.9 (D7b). best non-trend lead
- GDP: my pooled GDP -> bonds (t 2.2) fails Eemil's stability check. drop it
- ask Lynx: is the macro value on date t known on date t?

## Model / layers
- jump clip: barely matters on real data, keep as safety
- glide 10% / 30% / 100%: about the same score, only holding time changes
- ultrapure_silicon has fat tails (kurtosis ~10, biggest day 16.6 sigma), consider a cap (Eemil)
