# Backlog

## Ideas to try
- Multi-horizon trend (combine fast and slow lookbacks)
- Continuous signal (tanh instead of sign to cut turnover and slippage)
- Correlated assets (if correlation > 0.85, test spread / relative divergence)
- Reference prices (monthly open levels as dip-buying filter in uptrends)
- Volatility stop (scale down position size when rolling vol spikes)
- Macro filters (test regime filter if real data has macro series)

## Schedule
- Thursday night: Everyone test candidate models in your own research folder
- Friday: Compare scores in EXPERIMENTS.md and merge best model into `src/baseline.py`
- Saturday: Hackathon day (09:00 - 17:30)
