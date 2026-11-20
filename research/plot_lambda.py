"""LR = LR0 + lambda * (P - ref) / ref for several lambda = c * LR0, from research/lr_preview.json (no Modal)."""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).parent
d = json.loads((here / "lr_preview.json").read_text())
ET = ZoneInfo("America/New_York")
LR0 = 3e-3
CS = [1, 3, 5]
SHADE = {1: 0.45, 3: 0.75, 5: 1.0}
COL = {"ANT": "#2a78d6", "SPCX": "#eb6834", "SP500": "#1baf7a", "NVDA": "#6250d6"}
ink, muted, grid = "#111418", "#8a929e", "#eef1f4"

fig, axes = plt.subplots(4, 1, figsize=(14, 12))
print(f"{'run':6s} " + "  ".join(f"{'c=' + str(c):>22s}" for c in CS) + "   (LR ×1e-3: min / mean / max)")
for ax, r in zip(axes, ["ANT", "SPCX", "SP500", "NVDA"]):
    ratio = np.array(d["runs"][r]["s"])           # P / ref per step (ref = first bar of the run)
    x = np.arange(1, len(ratio) + 1)
    ax.grid(axis="y", color=grid); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted, labelsize=8)
    ax.axhline(3.0, color=muted, lw=0.8, ls="--")
    row = []
    for c in CS:
        lr = (LR0 + c * LR0 * (ratio - 1)) * 1e3
        ax.plot(x, lr, color=COL[r], alpha=SHADE[c], lw=1.3 if c == 5 else 1.0, label=f"λ = {c}×LR  ({c}% LR per 1% price)")
        row.append(f"{lr.min():5.2f} / {lr.mean():4.2f} / {lr.max():4.2f}")
        if lr.min() <= 0: row[-1] += " NEG!"
    print(f"{r:6s} " + "  ".join(f"{t:>22s}" for t in row))
    move = (ratio.min() - 1) * 100, (ratio.max() - 1) * 100
    ax.set_title(f"{r}: price moved {move[0]:+.1f}% to {move[1]:+.1f}% vs its starting price", loc="left", fontsize=10.5, color=ink)
    ax.set_ylabel("LR ×1e-3", color=muted, fontsize=9)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=ink, ncol=3)
axes[-1].set_xlabel("training step (7 jobs × 240 steps per day, Nov 13–19)", color=muted, fontsize=9)
fig.suptitle("LR = 3e-3 + λ · (P − ref) / ref,  ref = first price of the run.  λ = LR is the current formula.",
             x=0.01, ha="left", fontsize=12, color=ink, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = here / "lr_lambda.png"; fig.savefig(out, dpi=120); print(out)
