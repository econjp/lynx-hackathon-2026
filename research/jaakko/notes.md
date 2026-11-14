# Research notes

### Strategy
- trend: ewmac trend following, speed picked on train data
- sizing: inverse vol on price changes (handles negative prices like oil)
- risk: scale down positions when assets start moving together
- execution: move 30% toward target daily so we don't overtrade (holding ~40-50d, avoids penalty)
- roll clip: cap 1-day gaps at 5x vol so kurtosis doesn't blow up

### Results (Adj. Sharpe)
- example data: train 0.71 / test 0.97 (full ~0.81)
- 1d lag test: 0.84 (extra day delay doesn't hurt signal)
- real futures (29 assets): test +0.20 (team baseline broke at -0.08)
- 2024 data: test 0.64 (team baseline fell to 0.37)

### Dropped / Didn't work
- breakout: 95% same as ewmac, not worth running
- cross-asset mom: lost money on train
- macro data: too slow and lags price, just overfits

### Lynx question
- does test data include past history for warm-up, or just test dates?

### Run
- python research/jaakko/test_model.py
