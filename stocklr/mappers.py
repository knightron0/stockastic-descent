"""Price -> LR multiplier mappers. THIS is the part that's expected to change.

A mapper turns a ticker's price history into one multiplier per bar. The job then applies
    lr(step) = ETA0 * wsd(step) * multiplier
so a mapper only controls the *shape relative to the schedule*, never the schedule itself.

Contract (enforced by tests/test_mappers.py for every registered mapper):
  * input: Bars (oldest first; may include several days of history before the current hour)
  * output: np.ndarray, same length as the bars, one multiplier per bar
  * causal: the value at bar i may only depend on bars[: i + 1]
  * should stay within the mapper's own `bounds` (default [MIN_MULT, MAX_MULT]); `safe_multipliers`
    clips + checks anyway, so a buggy mapper can't blow up training
  * `neutral`: the value it returns for a perfectly flat price (default 1.0)

What the multiplier *means* depends on the schedule in stocklr/lr.py: WSD multiplies the LR by it;
Envelope treats it as the sigmoid signal s in (0, 1).

Adding one:
    @register("my_idea")
    class MyIdea:
        def __init__(self, alpha=1.0): self.alpha = alpha
        def __call__(self, bars: Bars) -> np.ndarray: ...
then set it in stocklr/config.py. Nothing else changes.
"""
from dataclasses import dataclass
import numpy as np

MIN_MULT, MAX_MULT = 0.5, 2.0


@dataclass(frozen=True)
class Bars:
    ts: np.ndarray        # datetime64 / datetimes, oldest first
    close: np.ndarray
    volume: np.ndarray

    def __len__(self):
        return len(self.close)

    def __getitem__(self, sl):
        return Bars(self.ts[sl], self.close[sl], self.volume[sl])

    @classmethod
    def from_rows(cls, rows):
        return cls(np.array([r["ts"] for r in rows]), np.array([r["close"] for r in rows], dtype=float),
                   np.array([r.get("volume", 0.0) for r in rows], dtype=float))


MAPPERS: dict[str, type] = {}


def register(name):
    def deco(cls):
        cls.name = name
        MAPPERS[name] = cls
        return cls
    return deco


def build(spec: dict):
    """spec = {"name": "detrended", **params} -> mapper instance."""
    spec = dict(spec)
    name = spec.pop("name")
    if name not in MAPPERS:
        raise KeyError(f"unknown mapper {name!r}; registered: {sorted(MAPPERS)}")
    return MAPPERS[name](**spec)


def safe_multipliers(mapper, bars: Bars) -> np.ndarray:
    """Run a mapper and guarantee a usable output: right length, finite, clipped to the mapper's bounds."""
    if len(bars) == 0:
        return np.ones(0)
    if not np.all(np.isfinite(bars.close)) or np.any(bars.close <= 0):
        raise ValueError("prices must be finite and positive")
    m = np.asarray(mapper(bars), dtype=float)
    if m.shape != (len(bars),):
        raise ValueError(f"{mapper.name}: returned shape {m.shape}, expected ({len(bars)},)")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"{mapper.name}: returned non-finite multipliers")
    lo, hi = getattr(mapper, "bounds", (MIN_MULT, MAX_MULT))
    return np.clip(m, lo, hi)


# ---------------------------------------------------------------- helpers

def ewm(x, span):
    """Exponential moving average (recursive form, alpha = 2 / (span + 1)). Causal."""
    x = np.asarray(x, dtype=float)
    a = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = a * v + (1 - a) * acc
        out[i] = acc
    return out


# ---------------------------------------------------------------- mappers

@register("constant")
class Constant:
    """Always `value`: the baseline (1.0 under WSD, 0.5 = the sigmoid midpoint under Envelope)."""
    bounds = (0.0, 10.0)

    def __init__(self, value=1.0):
        self.neutral = float(value)
        self.value = value

    def __call__(self, bars):
        return np.full(len(bars), float(self.value))


@register("detrended")
class Detrended:
    """How far the smoothed log price sits from its trailing EMA, scaled by k, tanh-soft-capped to (1/2, 2).
    Measured against a trailing average, so long rallies/crashes can't pin it (research/stress_test.py)."""
    def __init__(self, k=15.0, span=390, smooth=10, cap=float(np.log(2))):
        self.k, self.span, self.smooth, self.cap = k, span, smooth, cap

    def __call__(self, bars):
        lp = np.log(ewm(bars.close, self.smooth))
        dev = lp - ewm(lp, self.span)
        return np.exp(self.cap * np.tanh(self.k * dev / self.cap))


@register("price_ratio")
class PriceRatio:
    """(price / first price in the history) ** k. Follows the chart directly but DRIFTS over long horizons:
    kept for experiments, not recommended for the live run."""
    bounds = (0.0, float("inf"))     # no multiplier clip: the schedule clips the LR itself (min_lr / max_lr)
    def __init__(self, k=5.0, smooth=10):
        self.k, self.smooth = k, smooth

    def __call__(self, bars):
        p = ewm(bars.close, self.smooth)
        return (p / p[0]) ** self.k


@register("ema_sigmoid")
class EmaSigmoid:
    """Running-EMA sigmoid signal s_t in (0, 1).

        x_t = ln P_t,   a = 0.5 ** (1 / (h * B))
        W_t = a W_{t-1} + 1,  S1_t = a S1_{t-1} + x_t,  S2_t = a S2_{t-1} + x_t^2       (W, S1, S2 start at 0)
        mu_t = S1_t / W_t,  sigma_t = sqrt(max(S2_t / W_t - mu_t^2, 0))
        z_t = (x_t - mu_t) / (w sigma_t),  s_t = 1 / (1 + exp(-z_t))   (s_t = 1/2 when sigma_t = 0)

    h = half-life in trading days, B = bars per session (390 for 1-minute bars), w = width.
    Bars must be regular-session only, oldest first. mu/sigma depend on where the history starts, so
    the job always feeds history from a fixed anchor (>= 30 days before the run starts).
    Implementation detail: x is shifted by x_0 before accumulating. mu and sigma are shift-invariant, so
    the result is the same, but it avoids cancellation in S2/W - mu^2 (x ~ 5, variance ~ 1e-5).
    """
    bounds = (0.0, 1.0)
    neutral = 0.5

    def __init__(self, half_life_days=5.0, width=1.0, bars_per_session=390):
        if half_life_days <= 0 or width <= 0 or bars_per_session <= 0:
            raise ValueError("half_life_days, width and bars_per_session must be > 0")
        self.h, self.w, self.B = float(half_life_days), float(width), int(bars_per_session)

    def __call__(self, bars):
        x = np.log(np.asarray(bars.close, dtype=float))
        if len(x) == 0:
            return x
        x = x - x[0]
        a = 0.5 ** (1.0 / (self.h * self.B))
        out = np.empty_like(x)
        W = S1 = S2 = 0.0
        for t, xt in enumerate(x):
            W = a * W + 1.0
            S1 = a * S1 + xt
            S2 = a * S2 + xt * xt
            mu = S1 / W
            sigma = np.sqrt(max(S2 / W - mu * mu, 0.0))
            if sigma <= 1e-12:
                out[t] = 0.5
            else:
                z = (xt - mu) / (self.w * sigma)
                out[t] = 1.0 / (1.0 + np.exp(-z))
        return out
