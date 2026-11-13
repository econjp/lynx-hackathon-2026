# Experiment Leaderboard

Track model runs using Lynx's official evaluator (`evaluation.calc_key_figures`).

| Run ID | Author | Model Description | Adjusted Sharpe | Sharpe (1d lag) | Av Holding | Excess Kurtosis | Notes |
|---|---|---|---|---|---|---|---|
| `base-01` | Lynx | Default 50d sign baseline | 0.6621 | 0.6062 | 14.9d | +0.28 | Quickstart.ipynb baseline |
| `champ-01` | Team | Multi-horizon (40d/80d) + tanh + vol scaling | **0.8600** | 0.7917 | 14.9d | +0.58 | Current baseline in src/baseline.py |
| `ind-jake` | Jaakko | EWMAC (speed picked on train) + jump clip + vol sizing + ex-ante cov + 30% glide (`research/jaakko/model.py`) | **0.8070** | 0.8360 | 39.9d | +1.00 | full example data, fast. train/test 0.71 / 0.97. real futures test 0.20 (team baseline -0.08) |
| `ind-eemil` | Eemil | *TBD (due Thursday)* | - | - | - | - | In progress |
| `ind-erik` | Erik | *TBD (due Thursday)* | - | - | - | - | In progress |
| `ind-kris` | Kristofer | *TBD (due Thursday)* | - | - | - | - | In progress |
