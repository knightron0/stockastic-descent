"""Price -> LR multiplier mappers. THIS is the part that's expected to change.

A mapper turns a ticker's price history into one multiplier per bar. The job then applies
    lr(step) = ETA0 * wsd(step) * multiplier
so a mapper only controls the *shape relative to the schedule*, never the schedule itself.

Contract (enforced by tests/test_mappers.py for every registered mapper):
  * input: Bars (oldest first; may include several days of history before the current hour)
  * output: np.ndarray, same length as the bars, one multiplier per bar
  * causal: the value at bar i may only depend on bars[: i + 1]
  * should stay within [MIN_MULT, MAX_MULT]; `safe_multipliers` clips + checks anyway, so a buggy
    mapper can't blow up training

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
    """Run a mapper and guarantee a usable output: right length, finite, clipped to [MIN_MULT, MAX_MULT]."""
    if len(bars) == 0:
        return np.ones(0)
    if not np.all(np.isfinite(bars.close)) or np.any(bars.close <= 0):
        raise ValueError("prices must be finite and positive")
    m = np.asarray(mapper(bars), dtype=float)
    if m.shape != (len(bars),):
        raise ValueError(f"{mapper.name}: returned shape {m.shape}, expected ({len(bars)},)")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"{mapper.name}: returned non-finite multipliers")
    return np.clip(m, MIN_MULT, MAX_MULT)


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
    """Always 1.0: the WSD baseline."""
    def __init__(self, value=1.0):
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
    def __init__(self, k=5.0, smooth=10):
        self.k, self.smooth = k, smooth

    def __call__(self, bars):
        p = ewm(bars.close, self.smooth)
        return (p / p[0]) ** self.k
