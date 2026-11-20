import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import lr_algos as A

t = sys.argv[1] if len(sys.argv) > 1 else "SPCX"
here = Path(__file__).parent
df = pd.read_csv(here / "data" / f"{t}_1m.csv", parse_dates=["Datetime"]); df = df[df.Volume > 0].reset_index(drop=True)
p = df.Close.to_numpy(); dates = df.Datetime.dt.date.to_numpy()
day_starts = np.r_[True, dates[1:] != dates[:-1]]
x = np.arange(len(p))

algos = [
    ("Z-score vs trailing day  (α=0.3, clip 0.5–2×)", A.zscore(p)),
    ("Z-score vs whole week  (looks ahead — can't run live)", A.zscore_week_lookahead(p)),
    ("Log-return since day open  (k=50, current lr.py)", A.log_return(p, day_starts=day_starts)),
    ("Momentum  (30-min EMA of returns)", A.momentum(p)),
    ("Calm boost  (low volatility → higher LR)", A.calm_boost(p)),
    ("Drawdown  (LR falls below trailing high)", A.drawdown(p)),
]
fig, axes = plt.subplots(len(algos) + 1, 1, figsize=(13, 2.1 * (len(algos) + 1)), sharex=True,
                         gridspec_kw={"height_ratios": [1.6] + [1] * len(algos)})
ink, muted, grid = "#1d2329", "#7c8591", "#e6e9ed"
def style(ax):
    ax.grid(axis="y", color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted, labelsize=8)
    for d in np.flatnonzero(day_starts)[1:]: ax.axvline(d, color=grid, lw=1, zorder=0)

ax = axes[0]; style(ax)
ax.plot(x, p, color="#2a78d6", lw=1.3)
ax.set_title(f"{t} price, 1-minute bars (Yahoo Finance)", loc="left", fontsize=11, color=ink, fontweight="bold")
ax.set_ylabel("$", color=muted)
for ax, (name, mult) in zip(axes[1:], algos):
    style(ax)
    lr = A.schedule(mult)
    clipped = np.mean((lr <= A.ETA0 * 0.5 + 1e-12) | (lr >= A.ETA0 * 2 - 1e-12)) * 100
    ax.axhline(A.ETA0, color=muted, lw=0.8, ls="--")
    ax.plot(x, lr * 1e3, color="#eb6834", lw=1.1)
    ax.axhline(A.ETA0 * 1e3, color=muted, lw=0.8, ls="--")
    ax.set_ylim(1.2, 6.3); ax.set_yticks([1.5, 3, 6])
    ax.set_title(name, loc="left", fontsize=9.5, color=ink)
    ax.text(1.0, 1.02, f"mean {lr.mean()*1e3:.2f}e-3 · pinned at clip {clipped:.0f}%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=muted)
    ax.set_ylabel("LR ×1e-3", color=muted, fontsize=8)
ticks = np.flatnonzero(day_starts)
axes[-1].set_xticks(ticks + 195); axes[-1].set_xticklabels([pd.Timestamp(dates[i]).strftime("%a %b %d") for i in ticks])
axes[-1].tick_params(axis="x", length=0)
fig.text(0.01, 0.003, "Dashed line = base LR 3e-3. One step per minute here (live job will use 15 s → 4× the steps). "
         "Each trading day = 390 steps; one hourly cron batch = 60 of these minutes.", fontsize=8, color=muted)
fig.tight_layout(rect=(0, 0.015, 1, 1))
out = here / f"lr_algos_{t}.png"; fig.savefig(out, dpi=130); print(out)
