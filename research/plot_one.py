"""Price (regular session 9:30-16:00 ET) + LR curve. Usage: plot_one.py TICKER [k] [nowsd]"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import lr_algos as A

t = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
here = Path(__file__).parent
df = pd.read_csv(here / "data" / f"{t}_1m.csv", parse_dates=["Datetime"]); INDEX = t in ("SPX",)  # indexes report no real volume, so don't drop zero-volume bars
if not INDEX: df = df[df.Volume > 0]
df = df.reset_index(drop=True)
p = df.Close.to_numpy(); dt = df.Datetime; dates = dt.dt.date.to_numpy()
day_starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
x = np.arange(len(p))
k = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
mult = np.clip(A.price_ratio(p, k=k), 0.5, 2.0)
USE_WSD = "nowsd" not in sys.argv
sched = A.wsd(len(p), warmup=0.01, decay=0.2) if USE_WSD else np.ones(len(p))
lr = A.ETA0 * sched * mult

ink, muted, grid = "#1d2329", "#7c8591", "#e6e9ed"
fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
for ax in (a1, a2):
    ax.grid(axis="y", color=grid, lw=0.8); ax.set_axisbelow(True); ax.margins(x=0.005)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted, labelsize=8.5)
    for d in day_starts[1:]: ax.axvline(d - 0.5, color="#c9ced4", lw=1, zorder=0)

a1.plot(x, p, color="#2a78d6", lw=1.3)
a1.set_title(f"{t} {'(S&P 500 index) level' if INDEX else 'price'} — regular session only, 9:30 AM–4:00 PM ET (1-minute bars)", loc="left", fontsize=11.5, color=ink, fontweight="bold")
a1.set_ylabel("Index level" if INDEX else "Price ($)", color=muted)

if USE_WSD:
    a2.plot(x, A.ETA0 * sched * 1e3, color=muted, lw=1.1, ls="--", label="WSD schedule alone")
    a2.plot(x, lr * 1e3, color="#eb6834", lw=1.4, label="WSD × price")
    a2.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=ink)
else:
    a2.axhline(3, color=muted, lw=0.8, ls="--"); a2.text(len(p) + 5, 3, "base 3e-3", color=muted, fontsize=8, va="center")
    a2.plot(x, lr * 1e3, color="#eb6834", lw=1.4)
top = max(3.0, np.ceil(lr.max() * 1e3 * 2) / 2)
a2.set_ylim(0, top + 0.05); a2.set_yticks(np.arange(0, top + 0.01, 0.5))
a2.set_ylabel("Learning rate (×1e-3)", color=muted)
a2.set_title((f"LR = 3e-3 × WSD(step) × (price / starting price)^{k:g}    WSD: 1% warmup, stable, linear decay to 0 over last 20%" if USE_WSD
  else f"LR = 3e-3 × (price / starting price)^{k:g}    (no schedule, 10-min smoothing, clipped to 0.5–2×)"),
             loc="left", fontsize=10, color=ink)
a2.text(0, -0.30, f"mean LR {lr.mean()*1e3:.2f}e-3 (schedule alone {A.ETA0*sched.mean()*1e3:.2f}e-3) · price multiplier ranges {mult.min():.2f}–{mult.max():.2f}× · one step per minute, {len(p)} steps",
        transform=a2.transAxes, fontsize=8.5, color=muted)

# x ticks: 9:30 / 12:00 / 16:00 inside each day, day name underneath
ticks, labels = [], []
for i, s in enumerate(day_starts):
    e = day_starts[i + 1] if i + 1 < len(day_starts) else len(p)
    for hhmm in ("09:30", "12:00", "14:00") + (("15:59",) if i == len(day_starts) - 1 else ()):
        j = s + np.flatnonzero(dt.iloc[s:e].dt.strftime("%H:%M").to_numpy() == hhmm)
        if len(j): ticks.append(j[0]); labels.append({"09:30": "9:30", "12:00": "12:00", "14:00": "2:00", "15:59": "4:00"}[hhmm])
    a2.text((s + e) / 2, -0.17, pd.Timestamp(dates[s]).strftime("%a %b %d"), transform=a2.get_xaxis_transform(),
            ha="center", fontsize=9, color=ink)
a2.set_xticks(ticks); a2.set_xticklabels(labels, fontsize=7.5)
fig.tight_layout()
out = here / f"lr_{t}_{'wsd_price' if USE_WSD else 'price_only'}.png"; fig.savefig(out, dpi=130); print(out)
