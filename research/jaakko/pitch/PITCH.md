# Signal idea: multi-speed trend (Jake)

All numbers come from `TABLES.md`, built by `make_tables.py` from the raw Lynx data.
Check them yourself: `python research/jaakko/pitch/make_tables.py` (~10 s).

## 1. The idea (30 s)
- follow trends at a few speeds at once
- per asset: fast moving average - slow moving average, divided by the asset's vol
- positive = uptrend = long, negative = short
- two slow speed pairs (32/128, 64/256 days) + 12-month momentum, averaged 50/50
- continuous, capped at +-2: weak trend = small position, strong trend = full

## 2. Why it fits Lynx's rules (30 s)
- 2-day lag + slippage -> slow signals. holding ~65 days
- P&L = units x price change -> divide by price-change vol, every asset gets the same money risk
- adjusted sharpe punishes fat tails -> cap at +-2, no on/off flips

## 3. Results (1 min) -> T1, T2
- positive in all 3 periods: 1.17 / 0.73 / 0.40, full period 0.80
- team baseline 0.64, Lynx baseline 0.40 on the same data
- what I tried and dropped: on/off signal, fast speeds, simple MA, breakout (all in T1)

## 4. Is it real? (1 min) -> T3, T4
- leak test PASS
- 1 extra day of lag: sharpe 0.87 (unchanged)
- 2x slippage: 0.78
- without the 10 best days: 0.76
- placebo (signal shifted in time): 0.28 on average, best of 10 = 0.49, vs 0.80 real
- 25 of 27 assets make money

## 5. Weak spots (be upfront)
- scores fall over time (1.17 -> 0.40). trend is weaker in the latest period
- negative skew (-0.15), max drawdown ~2 years of vol
- all trend signals are 0.7-0.9 correlated (T5) -> another trend signal adds little

## 6. What the team should add next
- something NOT trend: macro per planet (GDP/CPI per planet vs its bonds, stocks, FX)
- sizing and execution tests on top of this signal (vol estimate, glide speed)
