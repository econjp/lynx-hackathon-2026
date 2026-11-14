# Signal idea: multi-speed trend (Jake)

All numbers come from `TABLES.md`, built by `make_tables.py` from the raw Lynx data.
Check them yourself: `python research/jaakko/pitch/make_tables.py` (~10 s).
Data analysis: `DATA.md`, built by `data_analysis.py`.

## 0. What the data looks like (1 min) -> DATA.md
- 27 assets, 1995-2016: 5 planets x (bonds, stocks, fx vs titan) + 13 commodities
- clean data: no bad OHLC bars, few jumps (0-7 days > 6 sigma per asset)
- most commodities fall a lot (1.0 -> 0.04-0.3): long downtrends, so the short side matters (D1)
- bonds and stocks move opposite (-0.2 to -0.3 on every planet), commodities barely correlate (0.09) (D2, D3)
- trend exists: past 12m direction gets next month right 56-57% for stocks and bonds, 53% fx,
  only 51% commodities (D4)
- macro: 60d GDP/CPI change barely predicts the planet's next 60 days (|corr| <= 0.22, noise level).
  CPI moves WITH stocks the same period (0.2-0.44), which is not a forecast (D7)
- high-low range vol is a steady ~80% of close vol -> Parkinson vol would barely change sizing (D8)

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

- money comes from every class: bonds 24%, stocks 26%, fx 20%, commodities 30% of P&L (D5)
- positive in 16 of 21 years, worst years 2006, 2009, 2012 (D6)

## 5. Weak spots (be upfront)
- scores fall over time (1.17 -> 0.40). trend is weaker in the latest period
- negative skew (-0.15), max drawdown ~2 years of vol
- all trend signals are 0.7-0.9 correlated (T5) -> another trend signal adds little

## 6. What the team should add next
- something NOT trend. macro per planet is the obvious candidate, but D7 shows the simple version
  doesn't predict -> needs a better idea (levels, differences between planets), not just momentum
- sizing and execution tests on top of this signal (vol estimate, glide speed)
