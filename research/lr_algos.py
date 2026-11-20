"""Candidate price -> LR mappings. All causal (only use prices up to t), like the live hourly job would.
Each returns a raw multiplier; `schedule` clips it to [lo, hi] and scales by eta0."""
import numpy as np
import pandas as pd

ETA0 = 3e-3


def _trail(x, w):
    s = pd.Series(x)
    return s.rolling(w, min_periods=min(30, w)).mean().to_numpy(), s.rolling(w, min_periods=min(30, w)).std().to_numpy()


def zscore(p, alpha=0.3, win=390):                 # price vs trailing 1-day mean, in std units
    m, s = _trail(p, win)
    return np.exp(alpha * np.nan_to_num((p - m) / s))

def zscore_week_lookahead(p, alpha=0.3):           # the pasted version: whole-week mean/std (uses the future)
    return np.exp(alpha * (p - p.mean()) / p.std())

def log_return(p, k=50.0, day_starts=None):        # current lr.py: amplified log-return since today's open
    anchor = np.maximum.accumulate(np.where(day_starts, np.arange(len(p)), 0))
    return np.exp(k * np.log(p / p[anchor]))

def momentum(p, k=1.0, span=30, win=390):          # EMA of 1-min returns, normalized by trailing return std
    r = np.r_[0, np.diff(np.log(p))]
    ema = pd.Series(r).ewm(span=span).mean().to_numpy()
    _, s = _trail(r, win)
    return np.exp(k * np.nan_to_num(ema / (s / np.sqrt(span))) / 3)

def calm_boost(p, beta=1.0, span=30):              # low realized vol -> higher LR, high vol -> lower
    r = np.r_[0, np.diff(np.log(p))]
    vol = pd.Series(r).rolling(span, min_periods=5).std().bfill().to_numpy()
    ref = pd.Series(vol).expanding().median().to_numpy()
    return (ref / vol) ** beta

def drawdown(p, k=15.0, win=390):                  # LR drops as price falls below its trailing high
    hi = pd.Series(p).rolling(win, min_periods=1).max().to_numpy()
    return np.exp(-k * (1 - p / hi))

def schedule(mult, lo=0.5, hi=2.0, smooth=1):
    m = np.clip(mult, lo, hi)
    if smooth > 1:
        m = pd.Series(m).ewm(span=smooth).mean().to_numpy()
    return ETA0 * m

def price_ratio(p, k=5.0, smooth=10):
    """LR follows the price chart directly: (P / P_start)^k, lightly EMA-smoothed.
    k=5 -> a 15% rise doubles the LR, a 13% fall halves it."""
    ps = pd.Series(p).ewm(span=smooth).mean().to_numpy() if smooth > 1 else p
    return (ps / p[0]) ** k

def wsd(n, warmup=0.01, decay=0.2):
    """Warmup-stable-decay multiplier over n steps: linear 0->1 warmup, flat, linear decay to 0 over the last `decay` fraction."""
    t = np.arange(n)
    w, d0 = max(1, int(n * warmup)), int(n * (1 - decay))
    m = np.ones(n)
    m[:w] = (t[:w] + 1) / w
    m[d0:] = 1 - (t[d0:] - d0) / max(1, n - 1 - d0)
    return m

# ---- bounded, drift-free variants (safe to run indefinitely) ----

def _squash(x, cap=np.log(2)):
    """Smoothly bound a log-multiplier to (-cap, cap): never pinned flat like a hard clip."""
    return cap * np.tanh(x / cap)

def detrended(p, k=5.0, span=390, smooth=10):
    """LR follows price relative to its trailing ~1-day EMA (fixed scale k, no division by volatility)."""
    lp = np.log(pd.Series(p).ewm(span=smooth).mean().to_numpy())
    base = pd.Series(lp).ewm(span=span).mean().to_numpy()
    return np.exp(_squash(k * (lp - base)))

def rank(p, win=390, smooth=10):
    """Percentile of price within the trailing window, mapped to 0.5x..2x in log space."""
    ps = pd.Series(pd.Series(p).ewm(span=smooth).mean())
    q = ps.rolling(win, min_periods=2).rank(pct=True).fillna(0.5).to_numpy()
    return np.exp(np.log(2) * (2 * q - 1))

def hour_normalized(mult, steps_per_hour=60):
    """Rescale each hourly batch so its mean log-multiplier is 0: only the within-hour shape survives."""
    lm = np.log(mult).copy()
    for s in range(0, len(lm), steps_per_hour):
        lm[s:s + steps_per_hour] -= lm[s:s + steps_per_hour].mean()
    return np.exp(lm)
