"""Plot research/lr_preview.json: per-run price, signal and LR over a replayed stretch of trading days."""
import json, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).parent
d = json.loads((here / "lr_preview.json").read_text())
ET = ZoneInfo("America/New_York")
SRC = {"ANT": "ANT", "SPCX": "SPCX", "SP500": "SPY", "NVDA": "NVDA"}
COL = {"ANT": "#2a78d6", "SPCX": "#eb6834", "SP500": "#1baf7a", "NVDA": "#6250d6", "BASELINE": "#8a929e"}
ink, muted, grid = "#111418", "#8a929e", "#eef1f4"
runs = d["runs"]

fig, axes = plt.subplots(5, 2, figsize=(15, 13), gridspec_kw={"width_ratios": [1, 1.25]})
for row, r in enumerate(["ANT", "SPCX", "SP500", "NVDA", "BASELINE"]):
    a_p, a_lr = axes[row]
    for ax in (a_p, a_lr):
        ax.grid(axis="y", color=grid); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted, labelsize=8)
    lr = np.array(runs[r]["lr"]) * 1e3; x = np.arange(1, len(lr) + 1)
    if r in SRC:
        p = d["prices"][SRC[r]]
        ts = [datetime.fromisoformat(t).astimezone(ET) for t in p["ts"]]
        days = np.r_[True, [ts[i].date() != ts[i - 1].date() for i in range(1, len(ts))]]
        bx = np.arange(len(ts))
        a_p.plot(bx, p["close"], color=COL[r], lw=1.1)
        for i in np.flatnonzero(days)[1:]: a_p.axvline(i, color=grid, lw=1)
        a_p.set_xticks(np.flatnonzero(days)); a_p.set_xticklabels([ts[i].strftime("%a %d") for i in np.flatnonzero(days)])
        a_p.set_title(f"{r} ({SRC[r]}) price, 1-min bars" if SRC[r] != r else f"{r} price, 1-min bars", loc="left", fontsize=10, color=ink)
    else:
        a_p.axis("off"); a_p.text(0, 0.5, "BASELINE: no price.\nLR = 3e-3 every step.", fontsize=10, color=ink, va="center")
    for k in range(240 * 7, len(lr), 240 * 7): a_lr.axvline(k, color=grid, lw=1)
    a_lr.plot(x, lr, color=COL[r], lw=1.2)
    a_lr.axhline(3.0, color=muted, lw=0.8, ls="--")
    lo, hi = min(lr.min(), 3.0), max(lr.max(), 3.0)          # each run zoomed to its own range (3e-3 kept in view)
    pad = max((hi - lo) * 0.08, 0.005)
    a_lr.set_ylim(lo - pad, hi + pad)
    a_lr.set_title(f"{r} learning rate (×1e-3)", loc="left", fontsize=10, color=ink)
    after = lr
    a_lr.text(1, 1.02, f"mean {after.mean():.2f}  min {after.min():.2f}  max {after.max():.2f}",
              transform=a_lr.transAxes, ha="right", va="bottom", fontsize=8, color=muted)
axes[-1][1].set_xlabel("training step (7 jobs × 240 steps per day)", color=muted, fontsize=9)
fig.suptitle("Replay of Nov 13–19 through the live code: LR = 3e-3 × price / starting price",
             x=0.01, ha="left", fontsize=11.5, color=ink, fontweight="bold")
fig.text(0.01, 0.005, "Dashed: 3e-3 = peak LR = BASELINE. Starting price = first bar of the run (Fri 13 9:30). "
         "Note ANT has only 1 day of history (IPO Nov 19), so it only appears on the last day.", fontsize=8.5, color=muted)
fig.tight_layout(rect=(0, 0.015, 1, 0.97))
out = here / "lr_preview.png"; fig.savefig(out, dpi=120); print(out)

print(f"\n{'run':9s} {'steps':>5s}  {'mean':>5s} {'min':>5s} {'max':>5s}   (×1e-3)")
for r in runs:
    a = np.array(runs[r]["lr"]) * 1e3
    if len(a): print(f"{r:9s} {len(runs[r]['lr']):5d}  {a.mean():5.2f} {a.min():5.2f} {a.max():5.2f}")
print("\nsample (one step per job, NVDA):")
nv = runs["NVDA"]
for i in range(239, len(nv["lr"]), 240 * 2)[:12]:
    t = datetime.fromisoformat(nv["ts"][i]).astimezone(ET)
    print(f"  step {i+1:5d}  {t:%a %H:%M} ET  s={nv['s'][i]:.3f}  s~={nv['s_tilde'][i]:.3f}  lr={nv['lr'][i]:.2e}")
