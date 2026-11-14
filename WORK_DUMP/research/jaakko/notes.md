# Fri notes

### Model (model.py)
- EWMA crossover (fast EMA - slow EMA) / vol, avg |f| = 1, cap +-2
- speed sets fast / mid / slow / all, pick best on train (first 60%). top two within 0.1 -> "all"
- units = forecast / price-change vol (P&L is X * dP)
- scale whole book so x' Cov x = 1 (250d cov of price changes)
- move 30% toward target each day -> holding ~40-65d
- clip one-day jumps at 5x vol before signals (roll gaps). P&L still on raw prices

### Results (adjusted sharpe, train / test)
- example data: ours (fast) 0.71 / 0.97, team baseline 40/80 tanh 0.73 / 1.05, Lynx 0.53 / 0.85
- real futures (29, yfinance): ours (all) 0.36 / 0.20, team baseline 0.02 / -0.08 (kurtosis 216)
- -> team baseline wins on example data, breaks on real data with roll jumps
- leak test PASS, +1 day lag ~ same

### Left out
- breakout: 0.95 corr with ewmac, off (w_breakout=0)
- xs_mom: lost money on train. macro: ~0 on test
- shrinking cov: no difference
- skew on futures test is negative (~-0.6), so no "positive skew" claim

### Ask Lynx
- does the test period come with the train history? model has ~1-2y warm-up

### Run
- python research/jaakko/test_model.py                         (example data)
- python research/jaakko/test_model.py data/futures_panel.csv  (local futures, not in git)

---

# Sat research update

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
- stress test data: test 0.64 (team baseline fell to 0.37)

### Dropped / Didn't work
- breakout: 95% same as ewmac, not worth running
- cross-asset mom: lost money on train
- macro data: too slow and lags price, just overfits

### Lynx question
- does test data include past history for warm-up, or just test dates?
