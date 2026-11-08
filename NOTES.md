# Notes & Ideas (Team Call 8.11.2026)

## Questions to ask Lynx
- How many assets in the real dataset, and roughly what asset classes (equities, commodities, bonds, FX)?
- How many additional macro time series are included, and what are they?
- How long is the out-of-sample period, and does it start right after in-sample data ends?
- How much weight goes to the Adjusted Sharpe vs the presentation/explanation?
- Are positions fractional/unbounded or capped?

---

## Strategy Ideas from Call

### 1. Multi-timeframe trend (Jake & Krisse)
- Instead of a single 50-day window, combine 3 horizons: Fast (20d), Medium (40d), and Slow (80d).
- Use `tanh()` instead of `np.sign()`. This makes position changes smooth, cuts turnover, saves on slippage, and lowers excess kurtosis.

### 2. Correlated assets & divergence (Krisse)
- If some assets have >0.85 correlation:
  - Track the spread or relative performance between them.
  - If one takes out a swing low while the other holds, buy the stronger one (divergence).
  - In quant terms: z-score the spread and trade mean reversion / relative momentum. Keeps market exposure near zero.

### 3. Reference levels / Opening prices (Krisse)
- In futures, monthly/quarterly open prices act as institutional anchor levels.
- Buy shallow dips below the monthly open, but only when the higher timeframe trend (e.g. 100d) is positive.
- If it breaks down >2 sigma below monthly open, cut it to avoid catching falling knives during crashes.

### 4. Stop-loss / Risk cut
- Hard stops (`if loss > 5%: exit`) fail with the 2-day lag and incur double slippage.
- Quant way: scale down position size continuously when 10-day volatility spikes above the 100-day baseline.

### 5. Simplicity & validation (Eemil)
- Keep parameter count small so we don't overfit in-sample.
- Test parameters across different sub-periods (walk-forward) to make sure they hold up out-of-sample.
