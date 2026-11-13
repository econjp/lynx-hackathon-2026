# Jaakko's research notes

## Fri notes

### 1. Signals
- sign() too jumpy -> causes fat tails + high kurtosis.
- multi-horizon EWMAC (16/64, 32/128, 64/256): continuous trend, capped at +/-2.
- breakout sleeve (80, 160, 320d): channel breakouts add positive skew (lynx rewards skew).

### 2. Risk & Sizing
- price difference sizing: divide by vol(diff(P)), not pct return vol. PnL is raw dollar moves.
- banana vol is 11.4% vs cucumber 1.45% -> inverse-vol parity mandatory.
- blend vol: fast 35d + slow 500d anchor. Cuts size fast in vol spikes.

### 3. Portfolio & Execution
- ex-ante cov scaling: we cut the correlated bets on day 1 (lynx baseline uses model_risk for dividing the positions within 100d PnL std.dev, which lags).
- execution glide: 30% daily partial adjustment towards target. Cuts turnover by 60%, extends holding to ~55-70d, kills slippage penalty.

### 4. Tests
- 100% causal: truncation leak test max diff = 0.00e+00.
- Adj Sharpe on example data: 0.60 (holding 55d, bias +0.003).
