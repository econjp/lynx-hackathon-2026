"""Draws the white-paper figures from whitepaper/sweeps.json.   python whitepaper/figures.py"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "sweeps.json").read_text())
BASE = S["base"][0]
GREEN = LinearSegmentedColormap.from_list("g", ["#f4f3ee", "#cfe9d9", "#7cc49b", "#1b8a52", "#0b5a33"])
INK, MUTED, BLUE = "#0b0b0b", "#52514e", "#2a78d6"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                     "axes.edgecolor": "#9a9993", "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED})
LO, HI = 0.6, 1.2


def heat(ax, M, xt, yt, xl, yl, title, base_ij, eras=None):
    im = ax.imshow(M, cmap=GREEN, vmin=LO, vmax=HI, origin="lower", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = f"{v:.2f}" + (f"\n{eras[i][j]}" if eras else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5 if eras else 10,
                    color="white" if v > 1.0 else INK, weight="bold" if (i, j) == base_ij else "normal")
    ax.add_patch(Rectangle((base_ij[1] - 0.5, base_ij[0] - 0.5), 1, 1, fill=False, lw=2.5, ec=BLUE))
    ax.set_xticks(range(len(xt)), xt); ax.set_yticks(range(len(yt)), yt)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, loc="left", color=INK, fontsize=11, weight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def era_txt(v):
    return "/".join(f"{x:.1f}" for x in v[1:])


# 1. trend speeds
g = S["trend_grid"]; F, R = g["fasts"], g["ratios"]
M = np.array([[g["cells"][f"{f}/{f * r}"][0] for f in F] for r in R])
E = [[era_txt(g["cells"][f"{f}/{f * r}"]) for f in F] for r in R]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
im = heat(ax, M, [str(f) for f in F], [f"{r}x" for r in R], "fast EMA span (days)", "slow span = fast x",
          "Trend: one EMA pair in place of 32/128 + 64/256", (R.index(4), F.index(32)), E)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="whole-model adjusted Sharpe")
fig.tight_layout(); fig.savefig(HERE / "fig_trend.png", dpi=200); plt.close(fig)

# 2. inflation lookbacks
g = S["infl_grid"]; K = g["k"]
M = np.array([[g["cells"][f"{b}/{s}"][0] for b in K] for s in K])
E = [[era_txt(g["cells"][f"{b}/{s}"]) for b in K] for s in K]
fig, ax = plt.subplots(figsize=(7.5, 4.6))
im = heat(ax, M, [str(k) for k in K], [str(k) for k in K], "bonds: CPI lookback k (days)", "stocks: CPI lookback k (days)",
          "Inflation lookbacks", (K.index(120), K.index(60)), E)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="whole-model adjusted Sharpe")
fig.tight_layout(); fig.savefig(HERE / "fig_inflation.png", dpi=200); plt.close(fig)

# 3. twins
g = S["twin_grid"]; C, SP = g["corr"], g["spread"]
M = np.array([[g["cells"][f"{c}/{s}"][0] for s in SP] for c in C])
E = [[era_txt(g["cells"][f"{c}/{s}"]) for s in SP] for c in C]
fig, ax = plt.subplots(figsize=(7.5, 4.6))
im = heat(ax, M, [str(s) for s in SP], [str(c) for c in C], "spread compared with its trailing mean over (days)",
          "twin threshold (correlation)", "Twins: threshold and spread window", (C.index(0.8), SP.index(250)), E)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="whole-model adjusted Sharpe")
fig.tight_layout(); fig.savefig(HERE / "fig_twins.png", dpi=200); plt.close(fig)

# 4. one-dimensional sweeps
panels = [("trend_mom", "Momentum lookback (days)", "250"), ("infl_normal", "Inflation 'normal' window (days)", "750"),
          ("glide", "Glide: share of the gap closed per day", "0.3"), ("risk_days", "Sleeve risk window (days)", "250"),
          ("costs", "Trading cost (x Lynx's cost)", "1")]
fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
for ax, (key, lab, base) in zip(axes, panels):
    d = S[key]; xs = list(d); ys = [d[x][0] if isinstance(d[x], list) else d[x] for x in xs]
    ax.axhspan(BASE - 0.1, BASE + 0.1, color="#cfe9d9", lw=0)
    ax.plot(range(len(xs)), ys, color=BLUE, lw=2, marker="o", ms=6)
    b = xs.index(base); ax.plot(b, ys[b], marker="o", ms=11, mfc="none", mec=INK, mew=2)
    for i, y in enumerate(ys):
        ax.text(i, y + 0.03, f"{y:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(xs)), xs); ax.set_title(lab, fontsize=9.5, color=INK, loc="left")
    ax.set_ylim(0.3, 1.35); ax.grid(axis="y", color="#e6e5e0"); ax.set_axisbelow(True)
axes[0].set_ylabel("whole-model adjusted Sharpe")
fig.tight_layout(); fig.savefig(HERE / "fig_1d.png", dpi=200); plt.close(fig)
print("figures written")
