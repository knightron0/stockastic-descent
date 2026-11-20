"""Compare price->LR mappings on (1) a real week and (2) a synthetic month built to break them."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import lr_algos as A

here = Path(__file__).parent
ink, muted, grid = "#1d2329", "#7c8591", "#e6e9ed"

def synthetic_month(seed=0):
    rng = np.random.default_rng(seed); days, per = 22, 390
    drift = np.concatenate([np.full(7 * per, np.log(1.6) / (7 * per)),     # +60% rally over 7 days
                            np.zeros(4 * per),                            # flat
                            np.full(6 * per, np.log(0.5) / (6 * per)),    # -50% crash over 6 days
                            np.zeros(5 * per)])                           # flat
    r = drift + 0.0012 * rng.standard_normal(days * per)
    r[14 * per] += np.log(0.8)                                            # -20% overnight gap (earnings)
    return 100 * np.exp(np.cumsum(r)), per

def methods(p):
    return [
        ("Price ÷ starting price, ^5  (current)", np.clip(A.price_ratio(p), 0.5, 2)),
        ("Detrended vs trailing 1-day avg, smooth-capped", A.detrended(p)),
        ("Rank within trailing day", A.rank(p)),
        ("Detrended + each hour re-centred to 1×", A.hour_normalized(A.detrended(p))),
    ]

def panel(p, per, title, out):
    ms = methods(p); x = np.arange(len(p))
    fig, axes = plt.subplots(len(ms) + 1, 1, figsize=(13, 1.9 * (len(ms) + 1)), sharex=True,
                             gridspec_kw={"height_ratios": [1.5] + [1] * len(ms)})
    for ax in axes:
        ax.grid(axis="y", color=grid, lw=0.8); ax.set_axisbelow(True); ax.margins(x=0.005)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted, labelsize=8)
        for d in range(per, len(p), per): ax.axvline(d, color=grid, lw=0.6, zorder=0)
    axes[0].plot(x, p, color="#2a78d6", lw=1.1); axes[0].set_title(title, loc="left", fontsize=11, color=ink, fontweight="bold")
    for ax, (name, m) in zip(axes[1:], ms):
        lr = A.ETA0 * m * 1e3
        pinned = np.mean((m <= 0.501) | (m >= 1.999)) * 100
        ax.axhline(3, color=muted, lw=0.8, ls="--")
        ax.plot(x, lr, color="#eb6834", lw=1.1)
        ax.set_ylim(0, 6.3); ax.set_yticks([0, 1.5, 3, 4.5, 6])
        ax.set_title(name, loc="left", fontsize=9.5, color=ink)
        ax.text(1, 1.02, f"mean {lr.mean():.2f}e-3 · stuck at a limit {pinned:.0f}% of steps", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=muted)
        ax.set_ylabel("LR ×1e-3", color=muted, fontsize=8)
    axes[-1].set_xticks(range(0, len(p), per * (5 if len(p) > 10 * per else 1)))
    axes[-1].set_xticklabels([f"day {i+1}" for i in range(0, len(p) // per, 5 if len(p) > 10 * per else 1)])
    fig.tight_layout(); fig.savefig(out, dpi=120); print(out)

p, per = synthetic_month()
panel(p, per, "STRESS TEST (fake): +60% rally, flat, 20% overnight gap, −50% crash — 22 trading days", here / "stress_synthetic.png")
df = pd.read_csv(here / "data" / "NVDA_1m.csv"); p = df[df.Volume > 0].Close.to_numpy()
panel(p, 390, "NVDA, real week (Nov 13–19), regular session", here / "stress_NVDA.png")
