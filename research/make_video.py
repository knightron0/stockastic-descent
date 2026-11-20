"""Explainer video: the stock market picks our learning rate.

    python3 research/make_video.py            # -> research/video/stockastic.mp4 (+ .gif)

Uses real NVDA 1-minute bars from research/data and the live formula (config.py):
    lr = 3e-3 * price / starting price
"""
import csv, math, subprocess
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

here = Path(__file__).parent
out = here / "video"; out.mkdir(exist_ok=True)

# ------------------------------------------------------------------ data
rows = list(csv.DictReader(open(here / "data" / "NVDA_1m.csv")))
price = np.array([float(r["Close"]) for r in rows])
day = [r["Datetime"][:10] for r in rows]
tod = [r["Datetime"][11:16] for r in rows]
PEAK = 3e-3
lr = PEAK * price / price[0]
N = len(price)
day_starts = [i for i in range(N) if i == 0 or day[i] != day[i - 1]]
from datetime import date
day_lbl = {i: date.fromisoformat(day[i]).strftime("%a %b %-d") for i in day_starts}

# biggest 15-minute moves -> captions
W = 15
chg = price[W:] / price[:-W] - 1
drop_i, rise_i = int(np.argmin(chg)) + W, int(np.argmax(chg)) + W
drop, rise = chg[drop_i - W], chg[rise_i - W]

# ------------------------------------------------------------------ style
BG, INK, MUT, GRID = "#0b0e11", "#e8edf2", "#7d8794", "#1c2229"
GREEN, RED, AMBER, BLUE = "#2fd07a", "#ff4d5e", "#ffb020", "#5aa9ff"
plt.rcParams.update({"font.family": ["Menlo", "DejaVu Sans Mono"], "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": MUT, "xtick.color": MUT, "ytick.color": MUT})
FPS = 30
fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)

def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)

def style_ax(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(labelsize=10)

def cosine(n):
    s = np.linspace(0, 1, n)
    w = np.clip(s / 0.03, 0, 1)
    return PEAK * w * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * s)))

def pct(x):
    return f"{'+' if x >= 0 else '−'}{abs(x) * 100:.1f}%"

# ------------------------------------------------------------------ scenes
def scene_cosine(f, n):
    """Everyone else: a tasteful, peer-reviewed schedule."""
    fig.text(0.07, 0.86, "How every ML paper picks a learning rate:", fontsize=24, weight="bold")
    ax = fig.add_axes([0.09, 0.16, 0.84, 0.6]); style_ax(ax)
    xs = np.arange(N); c = cosine(N)
    k = max(2, int(N * ease(f / (n * 0.7))))
    ax.plot(xs[:k], c[:k] * 1e3, color=BLUE, lw=3)
    ax.set_xlim(0, N); ax.set_ylim(0, 3.6); ax.set_xticks([])
    ax.set_ylabel("learning rate (×10⁻³)", fontsize=12); ax.set_xlabel("training step", fontsize=12)
    if f > n * 0.55:
        a = ease((f - n * 0.55) / (n * 0.2))
        ax.text(0.55, 0.75, "warmup → cosine decay\ncarefully tuned · smooth · boring", transform=ax.transAxes,
                fontsize=15, color=MUT, alpha=a)
    return ax

def scene_reveal(f, n):
    """Cross it out, show our formula."""
    ax = scene_cosine(n, n)
    a = ease(f / (n * 0.25))
    ax.plot([0, N], [0.2, 3.4], color=RED, lw=8 * a, alpha=a, solid_capstyle="round")
    ax.plot([0, N], [3.4, 0.2], color=RED, lw=8 * a, alpha=a, solid_capstyle="round")
    if f > n * 0.3:
        b = ease((f - n * 0.3) / (n * 0.25))
        fig.patches.append(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color=BG, alpha=0.85 * b, zorder=5))
        fig.text(0.5, 0.62, "How we pick it:", ha="center", fontsize=26, alpha=b, zorder=6)
        fig.text(0.5, 0.45, "lr  =  3e-3  ×  NVDA(now) / NVDA(start)", ha="center", fontsize=28,
                 weight="bold", color=GREEN, alpha=b, zorder=6)
        if f > n * 0.65:
            c = ease((f - n * 0.65) / (n * 0.2))
            fig.text(0.5, 0.3, "live 1-minute bars · every trading hour · no smoothing · no mercy",
                     ha="center", fontsize=15, color=MUT, alpha=c, zorder=6)

CAPTIONS = [  # (bar index, duration in bars, text, color)
    (40, 200, "Nvidia stock: now the learning-rate scheduler", AMBER),
    (drop_i, 260, f"NVDA {pct(drop)} in 15 min  →  model learns {abs(drop) * 100:.1f}% slower", RED),
    (rise_i, 260, f"NVDA {pct(rise)} in 15 min  →  gradient steps get {rise * 100:.1f}% bigger. hooray?", GREEN),
]

def scene_live(f, n, merge=0.0):
    k = max(2, int(N * ease(f / n) if merge == 0 else N))
    k = min(k, N)
    xs = np.arange(k)
    up = price[k - 1] >= price[0]
    col = GREEN if up else RED

    # header ticker
    fig.text(0.05, 0.93, "NVDA", fontsize=22, weight="bold")
    fig.text(0.13, 0.93, f"${price[k - 1]:.2f}", fontsize=22)
    fig.text(0.27, 0.93, ("▲ " if up else "▼ ") + pct(price[k - 1] / price[0] - 1), fontsize=22, color=col)
    fig.text(0.05, 0.885, f"{day_lbl[max(d for d in day_starts if d <= k - 1)]}  {tod[k - 1]} ET", fontsize=12, color=MUT)
    fig.text(0.62, 0.93, "LR", fontsize=22, weight="bold")
    fig.text(0.67, 0.93, f"{lr[k - 1]:.3e}", fontsize=22, color=col)
    fig.text(0.62, 0.885, f"model learning speed: {lr[k - 1] / PEAK * 100:5.1f}%", fontsize=12, color=MUT)

    top = [0.1, 0.5, 0.86, 0.34]; bot = [0.1, 0.1, 0.86, 0.34]
    m = ease(merge)
    pos_b = [b + (t - b) * m for t, b in zip(top, bot)]
    ax1 = fig.add_axes(top); ax2 = fig.add_axes(pos_b)
    for ax in (ax1, ax2):
        style_ax(ax); ax.set_xlim(0, N); ax.set_xticks([])
        for d in day_starts[1:]:
            ax.axvline(d, color=MUT, lw=0.8, ls=(0, (3, 4)), alpha=0.6)
    for d in day_starts:
        ax1.text(d + 6, 1.0, day_lbl[d], transform=ax1.get_xaxis_transform(), fontsize=9, color=MUT, va="bottom")
    ax1.set_ylim(price.min() * 0.985, price.max() * 1.01)
    ax2.set_ylim(lr.min() * 0.985, lr.max() * 1.01)
    ax1.set_ylabel("NVDA price ($)", fontsize=11)
    ax2.set_ylabel("learning rate", fontsize=11)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2e}"))
    ax1.plot(xs, price[:k], color=INK, lw=1.6)
    ax2.plot(xs, lr[:k], color=col, lw=1.6, alpha=1 - 0.4 * m)
    ax1.plot(k - 1, price[k - 1], "o", color=INK, ms=6)
    ax2.plot(k - 1, lr[k - 1], "o", color=col, ms=6)
    if merge > 0:
        ax2.patch.set_alpha(0); ax2.axis("off")
        for ax in (ax1,):
            ax.set_ylabel("")
        return

    # cosine ghost: what a sane schedule would be doing right now
    ax2.plot(np.arange(N), PEAK * np.ones(N), color=BLUE, lw=1.2, ls="--", alpha=0.5)
    ax2.text(N * 0.995, PEAK, "sane constant LR", color=BLUE, alpha=0.7, fontsize=9, ha="right", va="bottom")

    # "pipe" between panels
    ax1.annotate("", xy=(k - 1, ax1.get_ylim()[0]), xytext=(k - 1, ax1.get_ylim()[0]),
                 annotation_clip=False)
    fig.text(0.5, 0.465, "↓   × 3e-3 / $%.2f   ↓" % price[0], ha="center", fontsize=12, color=AMBER)

    # market close flash
    for d in day_starts[1:]:
        if d - 70 <= k - 1 < d + 4:
            fig.text(0.5, 0.67, "4:00 PM — MARKET CLOSED.  training paused.  model sent home.",
                     ha="center", fontsize=17, weight="bold", color=AMBER,
                     bbox=dict(boxstyle="round,pad=0.5", fc=BG, ec=AMBER, lw=1.5))

    for i0, dur, txt, c in CAPTIONS:
        if i0 <= k - 1 < i0 + dur:
            a = min(1.0, (k - 1 - i0) / 25, (i0 + dur - (k - 1)) / 25)
            fig.text(0.5, 0.06, txt, ha="center", fontsize=17, weight="bold", color=c, alpha=a)

def scene_merge(f, n):
    scene_live(n, n, merge=f / (n * 0.6))
    if f > n * 0.55:
        a = ease((f - n * 0.55) / (n * 0.2))
        fig.text(0.5, 0.25, "spot the difference.", ha="center", fontsize=34, weight="bold", alpha=a)
        fig.text(0.5, 0.17, "(there isn't one. it's the same curve.)", ha="center", fontsize=15, color=MUT, alpha=a)

def scene_end(f, n):
    tot = price[-1] / price[0] - 1
    lines = [
        (0.00, 0.68, f"NVDA fell {abs(tot) * 100:.1f}% this week.", 28, INK, "normal"),
        (0.22, 0.56, f"So our LLM learned {abs(tot) * 100:.1f}% slower.", 28, RED, "bold"),
        (0.42, 0.46, "On purpose.", 28, INK, "normal"),
        (0.62, 0.25, "stockastic descent", 40, GREEN, "bold"),
    ]
    for t0, y, txt, fs, c, w in lines:
        if f >= t0 * n:
            fig.text(0.5, y, txt, ha="center", fontsize=fs, color=c, weight=w, alpha=ease((f - t0 * n) / 12))
    if f >= 0.72 * n:
        fig.text(0.5, 0.18, "a small LLM whose learning rate is a stock price, live, every trading hour",
                 ha="center", fontsize=13, color=MUT, alpha=ease((f - 0.72 * n) / 12))

SCENES = [(scene_cosine, 3.0), (scene_reveal, 3.5), (scene_live, 15.0), (scene_merge, 3.5), (scene_end, 5.0)]

# ------------------------------------------------------------------ render
mp4 = out / "stockastic.mp4"
writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
with writer.saving(fig, str(mp4), dpi=100):
    for fn, secs in SCENES:
        n = int(secs * FPS)
        for f in range(n):
            fig.clf(); fig.patches.clear()
            fn(f, n)
            writer.grab_frame(facecolor=BG)
        print(f"{fn.__name__}: {n} frames")
print("wrote", mp4)

gif = out / "stockastic.gif"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                "fps=15,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer",
                str(gif)], check=True)
print("wrote", gif)
