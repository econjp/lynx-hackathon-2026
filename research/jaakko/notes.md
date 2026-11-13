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
