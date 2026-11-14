# 🛰️ Team NewNordics Shared Research Hub & Tactical Playbook
**Stockholm Lynx Hackathon (Nov 14, 2026)**  
**Team**: Jaakko Pakarinen, Eemil Ylä-Nikkilä, Kristofer Vaiksaar, Erik Joonas Salumäe

---

## 🏆 Current Production Baseline: The GOAT Model
* **Production Code**: [`src/goat_model.py`](./src/goat_model.py)
* **Interactive Walkthrough Notebook**: [`research/jaakko/strategy_walkthrough.ipynb`](./research/jaakko/strategy_walkthrough.ipynb)
* **Submission Runner**: [`submission/run_final.py`](./submission/run_final.py)
* **Official Positions File**: [`submission/positions_insample.csv`](./submission/positions_insample.csv)

### How to Import & Use the GOAT Model in Your Layer:
```python
import polars as pl
from src.goat_model import goat_positions, stretch_stop_positions, get_sector

prices = pl.read_csv("data/close_prices_insample.csv", try_parse_dates=True)
positions = goat_positions(prices)  # Returns pl.DataFrame with AsOfDate + 27 assets
```

### Baseline Performance Metrics (Lynx Official Evaluator `evaluation.py`):
| Metric | GOAT Model Value | Lynx Baseline | Why it Matters |
| :--- | :---: | :---: | :--- |
| **Full Adjusted Sharpe** | **1.137** | 0.595 | 2x higher risk-adjusted return |
| **Nominal Sharpe** | **1.317** | 0.626 | High gross compounding |
| **Average Holding Period**| **61.6 days** | 13.6 days | Completely immune to 2-day lag & slippage |
| **Excess Kurtosis** | **4.33** | 27.76 (Lynx raw) | Eliminates quadratic tail-risk penalty |
| **Rebalance Trades Skipped**| **72.0%** | 0% | 10% deadband retains 98% of gross alpha |
| **Era 1 (1995–2002) AdjSR**| **1.199** | 0.871 | Robust across historical eras |
| **Era 2 (2002–2009) AdjSR**| **1.117** | 0.603 | Survives dot-com crash & GFC |
| **Era 3 (2009–2016) AdjSR**| **1.007** | 0.380 | Survives post-crisis QE stagnation |
| **Causality Test** | **PASSED (0.00e+00)**| PASSED | Bit-level zero lookahead proof |

---

## 👥 The 4-Layer Division of Labor (Who Owns What)

### 1. Layer 1: Alpha Signals (Jake)
* **Scope**: Core signal generation and trend/momentum logic.
* **Current Engine**: Kristofer's EMA 10/100 stretch stop on 10 stocks & bonds + 12m sector-neutral cross-sectional momentum.
* **File to modify**: `src/goat_model.py` and `research/jaakko/`.

### 2. Layer 2: Covariance Matrix & Moon Holidays (Erik)
* **Scope**: Fix the moon holiday desynchronization artifact (Titan has 227 flat days, Ganymede 81; daily correlation falsely depressed to 0.41 vs true 0.54).
* **Target**: Implement a 5-day / weekly rolling covariance matrix with Ledoit-Wolf shrinkage.
* **File to work in**: `src/risk.py` or `research/erik/`.

### 3. Layer 3: Portfolio Construction & Macro Overlay (Eemil)
* **Scope**: Sizing weights and the Planetary CPI Macro Overlay.
* **Target**: Add the CPI acceleration overlay ($t = -2.96$ bond short tilt) into `src/final_model.py`.
* **File to work in**: `src/final_model.py` and `research/eemil/`.

### 4. Layer 4: Execution Shield & Friction Management (Kristofer)
* **Scope**: 10% rebalance deadband, exponential glide smoothing ($\alpha = 0.25$), and OOS validation pipeline.
* **Target**: Test `submission/run_final.py` on sliced insample data to guarantee zero failure when live test CSVs drop.
* **File to work in**: `submission/run_final.py` and `src/portfolio.py`.

---

## ⚡ 15-Minute Quant Instructor Defense Cheat Sheet
*(When Lynx Senior Quants walk in, use these exact bullet points)*

#### Q1: "What is your core strategy?"
> **Answer**: *"We deploy a Dual-Engine architecture: Kristofer's EMA 10/100 with a 3-close ratchet trailing stop on stocks and bonds to ride structural bull drift with an active crash filter, combined with a 12-month sector-neutral cross-sectional momentum engine that systematically shorts collapsing commodities. The two engines share an empirical correlation of only 0.12, giving us a major diversification dividend."*

#### Q2: "How do you handle Lynx's 2-day execution lag and 0.02 slippage?"
> **Answer**: *"We built a Layer 4 Execution Shield: a 10% rebalancing deadband and 25% exponential glide. Any target rebalance smaller than 10% is cancelled. This skips 72% of all daily trades, pushing our average holding period to 61.6 days and preserving 98% of gross alpha."*

#### Q3: "Why is your Adjusted Sharpe so high without being destroyed by kurtosis?"
> **Answer**: *"Lynx's Adjusted Sharpe heavily penalizes excess kurtosis. Fast, binary switching models explode kurtosis to 20+. Because we use smooth inverse-volatility sizing and continuous momentum ranking, our portfolio excess kurtosis is only 4.3 (close to Gaussian), avoiding the quadratic penalty entirely."*

#### Q4: "Is this just holding stocks?"
> **Answer**: *"No. Buy-and-hold has 100% net long exposure (`bias = +1.0`) and zero shorting power. When commodities fell 70–90%, buy-and-hold bled. Our model maintains an active short book on decaying commodities, and during the 2002–2009 crash era, our ratchet stop outperformed buy-and-hold by +0.32 Sharpe (1.117 vs 0.791)."*

#### Q5: "How do you know you have zero lookahead bias?"
> **Answer**: *"We ran truncation invariance causality tests cutting price histories at days 2000, 3000, 4000, and 5000. The maximum numerical difference between truncated and full-run positions is exactly 0.00e+00."*
