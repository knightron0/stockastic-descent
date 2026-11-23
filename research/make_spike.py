"""Minimal clip of a price spike across the live runs: learning rate | val loss, all runs, drawn in sync,
with a tag on the highlighted run's val-loss dot showing its rank.

    python3 research/make_spike.py STEPS.jsonl EVALS.jsonl [FIRST_STEP] [LAST_STEP] [HIGHLIGHT]
    -> research/video/spike_<HIGHLIGHT>.mp4 (+ .gif)

STEPS.jsonl / EVALS.jsonl: one Supabase row per line (steps: run,step,ts,lr; evals: run,step,val_loss).
"""
import json, subprocess, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FFMpegWriter

here = Path(__file__).parent
out = here / "video"; out.mkdir(exist_ok=True)
load = lambda p: [json.loads(l) for l in open(p)]
arg = lambda i, d: sys.argv[i] if len(sys.argv) > i else d
STEPS, EVALS = load(sys.argv[1]), load(sys.argv[2])
LO = int(arg(3, min(r["step"] for r in STEPS)))
HI = int(arg(4, max(r["step"] for r in STEPS)))
HL = arg(5, "NVDA")
ALL = ["ANT", "SPCX", "SP500", "NVDA"]
RUNS = [r for r in ALL if r != HL] + [HL]                     # highlighted run drawn last, on top

lr, val = {}, {}
for run in RUNS:
    rows = sorted((r for r in STEPS if r["run"] == run and LO <= r["step"] <= HI), key=lambda r: r["step"])
    lr[run] = (np.array([r["step"] for r in rows]), np.array([float(r["lr"]) for r in rows]))
    ev = sorted((r for r in EVALS if r["run"] == run and LO <= r["step"] <= HI), key=lambda r: r["step"])
    val[run] = (np.array([r["step"] for r in ev]), np.array([float(r["val_loss"]) for r in ev]))
hl_rows = sorted((r for r in STEPS if r["run"] == HL and LO <= r["step"] <= HI), key=lambda r: r["step"])
t = [datetime.fromisoformat(r["ts"]).timestamp() for r in hl_rows]
gaps = [hl_rows[i]["step"] for i in range(1, len(t)) if t[i] - t[i - 1] > 6 * 3600]   # overnight / weekend

# ---- web/src/app.css theme
BG, INK, INK2, INK3, RULE = "#fafbfc", "#15181d", "#4a525e", "#737780", "#e7e9ec"
COLOR = {"ANT": "#2a78d6", "SPCX": "#eb6834", "SP500": "#1baf7a", "NVDA": "#6250d6"}
NAME = {"SP500": "S&P 500"}
for f in (here / "fonts").glob("Geist*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": ["Geist", "DejaVu Sans"], "text.color": INK})

FPS, W, H = 30, 16, 9
fig = plt.figure(figsize=(W, H), dpi=120, facecolor=BG)
L, R, TOP, BOT, GAPY = 0.045, 0.91, 0.90, 0.05, 0.07         # right margin leaves room for the rank tag
PH = (TOP - BOT - GAPY) / 2

def upto(xs, ys, x):
    """Series clipped at step x, with the last segment interpolated so lines grow smoothly."""
    k = int(np.searchsorted(xs, x, side="right"))
    if k == 0:
        return xs[:0], ys[:0]
    if k < len(xs) and xs[k] > xs[k - 1]:
        y = ys[k - 1] + (ys[k] - ys[k - 1]) * (x - xs[k - 1]) / (xs[k] - xs[k - 1])
        return np.append(xs[:k], x), np.append(ys[:k], y)
    return xs[:k], ys[:k]

def latest(run, x):
    """Val loss at the tip of the drawn line (interpolated between evals), so the rank matches the chart."""
    ys = upto(*val[run], x)[1]
    return ys[-1] if len(ys) else val[run][1][0]

def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)

ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
POP = 0.3                                                      # seconds the rank tag pops when it changes
tag = {"rank": None, "since": -1e9}

def chart(j, label, data):
    y0 = TOP - (j + 1) * PH - j * GAPY
    ax = fig.add_axes([L, y0, R - L, PH * 0.84]); ax.patch.set_alpha(0)
    for s in ax.spines.values():
        s.set_visible(False)
    allv = np.concatenate([v for _, v in data.values()])
    lo, hi = allv.min(), allv.max(); pad = (hi - lo) * 0.08
    ax.set_xlim(LO, HI); ax.set_ylim(lo - pad, hi + pad); ax.set_xticks([]); ax.set_yticks([])
    ax.axhline(lo - pad, color=RULE, lw=1.2)
    for g in gaps:
        ax.axvline(g, color=INK3, lw=1, ls=(0, (2, 4)), alpha=0.5)
        ax.text(g, hi + pad, "  Mon open", color=INK3, fontsize=14, va="top")
    fig.text(L, y0 + PH * 0.94, label, fontsize=20, weight="medium", color=INK2, va="center")
    return ax

def frame(x, f):
    fig.clf(); fig.patches.clear()
    lx = R                                                     # legend, right-aligned on the top row
    for run in reversed(ALL):
        main = run == HL
        t = fig.text(lx, 0.955, NAME.get(run, run), fontsize=18, weight="semibold" if main else "medium",
                     color=INK if main else INK3, ha="right", va="center")
        lx -= t.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width + 0.008   # measured width
        fig.text(lx, 0.955, "●", color=COLOR[run], fontsize=14, ha="right", va="center")
        lx -= 0.03
    for j, (label, data) in enumerate([("learning rate", lr), ("val loss", val)]):
        ax = chart(j, label, data)
        for run in RUNS:
            xs, ys = upto(*data[run], x)
            if len(xs) == 0:
                continue
            main = run == HL
            ax.plot(xs, ys, color=COLOR[run], lw=3.2 if main else 2, alpha=1 if main else 0.5,
                    solid_joinstyle="round", solid_capstyle="round", zorder=3 if main else 2)
            ax.plot(xs[-1], ys[-1], "o", color=COLOR[run], ms=9 if main else 6, mec=BG, mew=2, zorder=4)
            if j == 1 and main:                                # rank tag rides the highlighted run's val-loss dot
                rank = sorted(ALL, key=lambda r: latest(r, x)).index(HL) + 1
                if rank != tag["rank"]:
                    tag["since"] = f if tag["rank"] is not None else -1e9
                    tag["rank"] = rank
                t = (f - tag["since"]) / (POP * FPS)
                pop = 1 + 0.35 * np.sin(np.pi * t) if 0 <= t < 1 else 1
                ax.annotate(ORD[rank], (xs[-1], ys[-1]), xytext=(16, 0), textcoords="offset points",
                            fontsize=17 * pop, weight="semibold", color="white", va="center", ha="left",
                            annotation_clip=False, zorder=6,
                            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.7", fc=COLOR[run], ec="none"))

DRAW, HOLD = 7.0, 2.5                                    # fixed length, whatever the step range
n_draw, n_hold = int(DRAW * FPS), int(HOLD * FPS)
mp4 = out / f"spike_{HL}.mp4"
writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
with writer.saving(fig, str(mp4), dpi=120):
    for i in range(n_draw + n_hold):
        frame(LO + 1 + (HI - LO - 1) * min(i / n_draw, 1.0), i)
        writer.grab_frame(facecolor=BG)
print("wrote", mp4, f"(steps {LO}..{HI})")

gif = out / f"spike_{HL}.gif"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                "fps=20,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=48[p];[b][p]paletteuse=dither=bayer",
                str(gif)], check=True)
print("wrote", gif)
