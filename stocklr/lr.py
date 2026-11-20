"""The fixed part of the LR: base rate x WSD schedule, and how bars map onto the 240 steps of a job.
The price-dependent part lives in stocklr/mappers.py."""
import numpy as np

ETA0 = 3e-3
TOTAL_STEPS = 9_375            # one pass over 600k sequences at batch 64
WARMUP_STEPS = 94              # 1%
DECAY_START = 7_500            # stable until 80%, then linear decay to 0


def wsd(step: int) -> float:
    """Warmup-stable-decay multiplier for a 1-based global step."""
    if step <= WARMUP_STEPS:
        return step / WARMUP_STEPS
    if step <= DECAY_START:
        return 1.0
    return max(0.0, (TOTAL_STEPS - step) / (TOTAL_STEPS - DECAY_START))


def bars_to_steps(n_bars: int, n_steps: int = 240):
    """Index of the bar each step uses: 60 bars -> 4 steps per bar, 30 bars -> 8."""
    if n_bars <= 0:
        raise ValueError("need at least one bar")
    return (np.arange(n_steps) * n_bars // n_steps).astype(int)
