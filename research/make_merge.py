"""Twitter cut: ~7.5s 16:9 loop in the dashboard's theme. Price and LR draw side by side, then slide together.

    python3 research/make_merge.py [TICKER]     # default ANT -> research/video/stockastic_ANT.mp4 (+ .gif)

lr = 3e-3 * price / starting price (config.py). Bars with zero volume (e.g. ANT's pre-IPO placeholder) are dropped,
so the starting price is the first real trade, as in the live run.
"""
import csv, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FFMpegWriter

here = Path(__file__).parent
out = here / "video"; out.mkdir(exist_ok=True)
TICKER = (sys.argv[1] if len(sys.argv) > 1 else "ANT").upper()

rows = [r for r in csv.DictReader(open(here / "data" / f"{TICKER}_1m.csv")) if float(r["Volume"]) > 0]
price = np.array([float(r["Close"]) for r in rows])
PEAK = 3e-3
lr = PEAK * price / price[0]
N = len(price)

# ---- web/src/app.css theme
BG, PANEL, INK, INK2, INK3, RULE, GRID = "#fafbfc", "#ffffff", "#15181d", "#4a525e", "#737780", "#e7e9ec", "#f0f1f3"
OK, BAD = "#18794e", "#c2352f"
SERIES = {"ANT": "#2a78d6", "SPCX": "#eb6834", "SPY": "#1baf7a", "NVDA": "#6250d6"}
ACC = SERIES.get(TICKER, "#2a78d6")
for f in (here / "fonts").glob("Barlow-*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": ["Barlow", "DejaVu Sans"], "text.color": INK})

FPS = 30
W, H = 16, 9
fig = plt.figure(figsize=(W, H), dpi=120, facecolor=BG)

def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)

# panel geometry (figure fractions)
PW, PH, PY = 0.44, 0.74, 0.09
LEFT_X, RIGHT_X, MID_X = 0.04, 0.52, 0.28
PAD_X, HEAD = 0.055, 0.1          # inner plot inset: room for y labels / panel head

def card(x, alpha=1.0):
    fig.patches.append(FancyBboxPatch((x, PY), PW, PH, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fc=PANEL, ec=RULE, lw=1.2, alpha=alpha,
                                      mutation_aspect=W / H, zorder=-5))

def plot(x, y, k, color, lw, side, fmt, alpha=1.0, grid=True):
    ax = fig.add_axes([x + PAD_X, PY + 0.05, PW - 2 * PAD_X, PH - HEAD - 0.05])
    ax.patch.set_alpha(0); ax.set_zorder(5)
    for s in ax.spines.values():
        s.set_visible(False)
    lo, hi = y.min(), y.max(); pad = (hi - lo) * 0.08
    ax.set_xlim(0, N - 1); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks([])
    ticks = np.linspace(lo, hi, 4)
    ax.set_yticks(ticks)
    ax.yaxis.set_ticks_position(side); ax.tick_params(length=0, labelsize=13, labelcolor=INK3, pad=10)
    ax.set_yticklabels([fmt(t) for t in ticks], alpha=alpha)
    if grid:
        ax.grid(axis="y", color=GRID, lw=1.2)
    ax.plot(np.arange(k), y[:k], color=color, lw=lw, alpha=alpha, solid_joinstyle="round")
    ax.plot(k - 1, y[k - 1], "o", color=color, ms=9, alpha=alpha, mec=PANEL, mew=2)
    return ax

def head(x, label, value, delta, color, alpha=1.0):
    ty = PY + PH - 0.065
    fig.text(x + 0.022, ty, label, fontsize=24, weight="semibold", color=color, alpha=alpha)
    fig.text(x + PW - 0.022, ty, value, fontsize=24, weight="medium", color=INK, ha="right", alpha=alpha,
             fontfamily="Barlow")
    fig.text(x + PW - 0.022, ty - 0.045, delta, fontsize=17, weight="medium", ha="right", alpha=alpha,
             color=OK if delta.startswith("+") else BAD)

DRAW, MERGE, HOLD = 4.0, 1.0, 2.5
n_draw, n_merge, n_hold = int(DRAW * FPS), int(MERGE * FPS), int(HOLD * FPS)
START = 0.03

def pct(r):
    return f"{'+' if r >= 0 else '−'}{abs(r) * 100:.1f}%"

def frame(i):
    fig.clf(); fig.patches.clear()
    k = max(2, int(N * (START + (1 - START) * ease(i / n_draw))))
    m = ease((i - n_draw) / n_merge) if i > n_draw else 0.0
    d = pct(price[k - 1] / price[0] - 1)
    px = LEFT_X + (MID_X - LEFT_X) * m
    lx = RIGHT_X + (MID_X - RIGHT_X) * m
    fade = max(0.0, 1 - 2.5 * m)

    card(px)
    if m < 1:
        card(lx, alpha=fade)
    plot(px, price, k, INK2, 2.4, "left", lambda v: f"${v:.0f}")
    plot(lx, lr, k, ACC, 3.0, "right", lambda v: f"{v * 1e3:.2f}e-3", grid=m == 0)
    head(px, f"${TICKER}", f"${price[k - 1]:.2f}", d, INK2)
    head(lx, "learning rate", f"{lr[k - 1]:.2e}", d, ACC, alpha=fade)

    if i > n_draw + n_merge * 0.7:
        a = ease((i - n_draw - n_merge * 0.7) / 9)
        fig.text(0.5, 0.89, "same curve.", ha="center", fontsize=48, weight="semibold", alpha=a)
    fig.text(0.96, 0.045, "stockastic descent", ha="right", fontsize=15, color=INK3)

mp4 = out / f"stockastic_merge_{TICKER}.mp4"
writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
with writer.saving(fig, str(mp4), dpi=120):
    for i in range(n_draw + n_merge + n_hold):
        frame(i); writer.grab_frame(facecolor=BG)
print("wrote", mp4)

gif = out / f"stockastic_merge_{TICKER}.gif"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                "fps=20,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=48[p];[b][p]paletteuse=dither=bayer",
                str(gif)], check=True)
print("wrote", gif)
