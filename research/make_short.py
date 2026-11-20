"""Twitter cut: ~5s 16:9 loop in the dashboard's theme, from a real run's backup.
$TICKER price | learning rate | training loss draw in sync, then freeze midway: "??" on the loss curve,
then fade to an end card: "stockastic descent" / stockasticdescent.live.

    python3 research/make_short.py [RUN] [BACKUP_TAG]     # default ANT thursday-2026-11-19
    -> research/video/stockastic_<RUN>.mp4 (+ .gif)
"""
import csv, subprocess, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FFMpegWriter

here = Path(__file__).parent
out = here / "video"; out.mkdir(exist_ok=True)
RUN = (sys.argv[1] if len(sys.argv) > 1 else "ANT").upper()
TAG = sys.argv[2] if len(sys.argv) > 2 else "thursday-2026-11-19"
SYMBOL = {"SP500": "SPY"}.get(RUN, RUN)
src = here.parent / "backups" / TAG

# ---- data. Loss = the real run's logged train loss. LR = what-if: 3e-3 * price / IPO price, where the IPO price is
# the zero-volume placeholder bar in research/data/<SYMBOL>_1m.csv ($83 for ANT). (The live run held LR at 3e-3
# until trading opened and used the first trade, $111.13, as its starting price, so it never saw the pop.)
steps = [r for r in csv.DictReader(open(src / "steps.csv")) if r["run"] == RUN]
prices = [r for r in csv.DictReader(open(src / "prices.csv")) if r["ticker"] == SYMBOL and float(r["volume"] or 0) > 0]
ts = lambda s: datetime.fromisoformat(s).timestamp()
p_ts = [ts(r["ts"]) for r in prices]; p_close = [float(r["close"]) for r in prices]
ipo = next(csv.DictReader(open(here / "data" / f"{SYMBOL}_1m.csv")))
IPO = float(ipo["Close"]) if float(ipo["Volume"]) == 0 else p_close[0]
if float(ipo["Volume"]) == 0:
    p_ts.insert(0, ts(ipo["Datetime"])); p_close.insert(0, IPO)
p_ts, p_close = np.array(p_ts), np.array(p_close)
s_ts = np.array([ts(r["ts"]) for r in steps])
loss_all = np.array([float(r["train_loss"]) for r in steps])
LEAD = 60                                            # steps of flat IPO price shown before the open
opened = int(np.searchsorted(s_ts, p_ts[-len(prices)]))
first = max(int(np.searchsorted(s_ts, p_ts[0])), opened - LEAD)
idx = np.searchsorted(p_ts, s_ts[first:], side="right") - 1
price, loss = p_close[idx], loss_all[first:]
PEAK_LR = 3e-3
lr = PEAK_LR * price / IPO
step0 = first + 1
ema = np.empty_like(loss); acc = loss[0]
for i, v in enumerate(loss):
    acc = 0.97 * acc + 0.03 * v; ema[i] = acc
N = len(price)
STOP = (opened - first) + int((N - (opened - first)) * 0.5)    # freeze midway through the trading session
PK = int(np.argmax(price[:STOP]))                    # price line: green up to the peak, red after

# ---- web/src/app.css theme
BG, PANEL, INK, INK2, INK3, RULE, GRID = "#fafbfc", "#ffffff", "#15181d", "#4a525e", "#737780", "#e7e9ec", "#f0f1f3"
OK, BAD = "#18794e", "#c2352f"
SERIES = {"ANT": "#2a78d6", "SPCX": "#eb6834", "SP500": "#1baf7a", "NVDA": "#6250d6"}
ACC = SERIES.get(RUN, "#2a78d6")
for f in (here / "fonts").glob("Barlow-*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": ["Barlow", "DejaVu Sans"], "text.color": INK})

FPS, W, H = 30, 21, 9                                 # 21:9, 2520x1080
fig = plt.figure(figsize=(W, H), dpi=120, facecolor=BG)

def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)

from matplotlib.patches import Polygon
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator

PW, PH, PY, GAP = 0.31, 0.84, 0.08, 0.01
XS = [0.5 - 1.5 * PW - GAP + j * (PW + GAP) for j in range(3)]
PAD_L, PAD_R, HEAD = 0.016, 0.016, 0.13
XMAX = STOP / 0.6                                     # data stops 60% of the way across: the run isn't over

def card(x):
    for off, a in ((0.006, 0.035), (0.003, 0.05)):   # soft drop shadow
        fig.patches.append(FancyBboxPatch((x, PY - off), PW, PH, boxstyle="round,pad=0,rounding_size=0.014",
                                          transform=fig.transFigure, fc="#1b2330", ec="none", alpha=a,
                                          mutation_aspect=W / H, zorder=-6))
    fig.patches.append(FancyBboxPatch((x, PY), PW, PH, boxstyle="round,pad=0,rounding_size=0.014",
                                      transform=fig.transFigure, fc=PANEL, ec=RULE, lw=1.2,
                                      mutation_aspect=W / H, zorder=-5))

def gfill(ax, xs, ys, color, top_alpha, floor):
    """Vertical gradient under a curve: top_alpha at the curve's peak fading to 0 at the floor."""
    r, g, b = to_rgb(color)
    grad = np.zeros((256, 1, 4)); grad[..., :3] = (r, g, b); grad[:, 0, 3] = np.linspace(0, top_alpha, 256)
    im = ax.imshow(grad, extent=[xs[0], xs[-1], floor, ys.max()], aspect="auto", origin="lower", zorder=1,
                   interpolation="bicubic")
    im.set_clip_path(Polygon([(xs[0], floor), *zip(xs, ys), (xs[-1], floor)], closed=True,
                             transform=ax.transData, fc="none", ec="none"))

def plot(x, series, k, fmt, color, raw=None, color2=None):
    ax = fig.add_axes([x + PAD_L, PY + 0.045, PW - PAD_L - PAD_R, PH - HEAD - 0.045])
    ax.patch.set_alpha(0); ax.set_zorder(5)
    for s in ax.spines.values():
        s.set_visible(False)
    vis = series[:STOP]
    lo, hi = vis.min(), vis.max(); pad = (hi - lo) * 0.12
    ax.set_xlim(0, XMAX); ax.set_ylim(lo - pad, hi + pad); ax.set_xticks([])
    ax.yaxis.set_major_locator(MaxNLocator(4)); ax.set_yticklabels([]); ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, lw=1.2)
    ax.axhline(lo - pad, color=RULE, lw=1.2)
    xs = np.arange(k)
    now = color if color2 is None or k - 1 <= PK else color2
    gfill(ax, xs, series[:k], now, 0.13 if color2 else 0.22, lo - pad)
    if raw is not None:
        ax.plot(xs, raw[:k], color=color, lw=0.9, alpha=0.18, zorder=2)
    kw = dict(lw=2.8, zorder=3, solid_joinstyle="round", solid_capstyle="round")
    if color2 is None:
        ax.plot(xs, series[:k], color=color, **kw)
    else:
        ax.plot(xs[:PK + 1], series[:min(k, PK + 1)], color=color, **kw)
        if k > PK:
            ax.plot(xs[PK:], series[PK:k], color=color2, **kw)
    color = now
    ax.axvline(k - 1, color=INK3, lw=1, ls=(0, (2, 3)), alpha=0.5, zorder=2)          # "now" guide
    ax.plot(k - 1, series[k - 1], "o", color=color, ms=22, alpha=0.15, mew=0, zorder=4)   # halo
    ax.plot(k - 1, series[k - 1], "o", color=color, ms=9, mec=PANEL, mew=2.2, zorder=5)
    return ax

def head(x, label, value, color):
    ty = PY + PH - 0.075
    fig.text(x + 0.016, ty, label, fontsize=26, weight="semibold", color=color, va="center")
    fig.text(x + PW - 0.016, ty, value, fontsize=30, weight="semibold", color=INK, ha="right", va="center")

def pct(r):
    return f"{'+' if r >= 0 else '−'}{abs(r) * 100:.1f}%"

DRAW, FREEZE, FADE, HOLD = 3.0, 2.2, 0.7, 2.0
n_draw, n_freeze, n_fade, n_hold = (int(d * FPS) for d in (DRAW, FREEZE, FADE, HOLD))
START = (LEAD - 12) / STOP            # frame 0: flat at the IPO price, pop a few frames later

def frame(i):
    fig.clf(); fig.patches.clear()
    t = min(i / n_draw, 1.0)
    k = max(2, int(STOP * (START + (1 - START) * (1 - (1 - t) ** 2))))   # ease-out: the pop lands fast
    for x in XS:
        card(x)

    plot(XS[0], price, k, lambda v: f"${v:.0f}", BAD)
    plot(XS[1], lr, k, lambda v: f"{v * 1e3:.1f}e-3", ACC)
    ax_loss = plot(XS[2], ema, k, lambda v: f"{v:.1f}", ACC, raw=loss)
    head(XS[0], f"${SYMBOL}", f"${price[k - 1]:.2f}", INK2)
    head(XS[1], "learning rate", f"{lr[k - 1]:.2e}".replace("e-03", "e-3"), ACC)
    head(XS[2], "training loss", f"{ema[k - 1]:.2f}", ACC)

    if i >= n_draw:                                    # freeze; "??" pops into the empty 40% of the loss panel
        t = (i - n_draw) / 9
        s = 1 + 0.3 * np.sin(np.pi * t) if t < 1 else 1
        ax_loss.text(0.8, 0.5, "??", transform=ax_loss.transAxes, ha="center", va="center", fontsize=110 * s,
                   weight="semibold", color=INK, alpha=ease(t * 2), zorder=10)

    j = i - n_draw - n_freeze                          # end card: charts fade out, title fades in over them
    if j >= 0:
        a = ease(j / n_fade)
        fig.patches.append(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, fc=BG, ec="none", alpha=a, zorder=20))
        b = ease((j - n_fade * 0.4) / n_fade)
        fig.text(0.5, 0.54, "stockastic descent", ha="center", va="center", fontsize=96, weight="bold",
                 color=INK, alpha=b, zorder=21)
        fig.text(0.5, 0.40, "stockasticdescent.live", ha="center", va="center", fontsize=40, weight="medium",
                 color=ACC, alpha=b, zorder=21)


mp4 = out / f"stockastic_{RUN}.mp4"
writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
with writer.saving(fig, str(mp4), dpi=120):
    for i in range(n_draw + n_freeze + n_fade + n_hold):
        frame(i); writer.grab_frame(facecolor=BG)
print("wrote", mp4, f"(steps {step0}..{step0 + STOP - 1})")

gif = out / f"stockastic_{RUN}.gif"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                "fps=20,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=48[p];[b][p]paletteuse=dither=bayer",
                str(gif)], check=True)
print("wrote", gif)
