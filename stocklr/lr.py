"""Schedules: turn a step's raw mapper value into the learning rate actually applied.

The mapper (stocklr/mappers.py) decides how price moves; the schedule decides what that means for the LR
over the run. Schedules may keep state across steps (e.g. Envelope's rise cap); that state is stored in
each run's checkpoint so it carries across jobs.

    sched = build_schedule(SCHEDULE)
    lr, logged = sched(step, raw_value, state)      # step is 1-based and global; state is a dict
"""
import numpy as np

TOTAL_STEPS = 9_375            # one pass over 600k sequences at batch 64: the run stops here
STEPS_PER_JOB = 60            # one 15-minute job = 15 one-minute bars x 4 steps per bar

# legacy WSD constants (kept for the "wsd" schedule)
ETA0 = 3e-3
WARMUP_STEPS = 94              # 1%
DECAY_START = 7_500            # stable until 80%, then linear decay to 0


def wsd(step: int) -> float:
    """Warmup-stable-decay multiplier for a 1-based global step."""
    if step <= WARMUP_STEPS:
        return step / WARMUP_STEPS
    if step <= DECAY_START:
        return 1.0
    return max(0.0, (TOTAL_STEPS - step) / (TOTAL_STEPS - DECAY_START))


class WSD:
    """lr = eta0 * wsd(step) * multiplier. Stateless."""
    def __init__(self, eta0=ETA0):
        self.eta0 = eta0

    def __call__(self, step, mult, state):
        return self.eta0 * wsd(step) * mult, mult


class Envelope:
    """Running-EMA sigmoid envelope. With n = step - 1 and N = total_steps:

        s~_n = min(s_n, s~_{n-1} + max_rise / (rise_window * N)),  s~_0 = s_0     (rises capped, falls pass)
        m_n  = min(1, (n + 1) / (warmup * N))                                     (linear warmup, no cooldown)
        lr_n = m_n * eta_max * (floor + (1 - floor) * s~_n)

    `mult` is the mapper's signal s_n in (0, 1). Logged value = s~_n. state["s_tilde"] carries s~ across jobs.
    """
    def __init__(self, eta_max=3e-3, floor=0.1, warmup=0.02, max_rise=0.5, rise_window=0.01, total_steps=TOTAL_STEPS):
        if not 0 <= floor < 1:
            raise ValueError("floor must be in [0, 1)")
        self.eta_max, self.floor, self.warmup = float(eta_max), float(floor), float(warmup)
        self.max_step_rise = float(max_rise) / (float(rise_window) * total_steps)
        self.N = int(total_steps)

    def __call__(self, step, s, state):
        prev = state.get("s_tilde")
        st = float(s) if prev is None else min(float(s), prev + self.max_step_rise)
        state["s_tilde"] = st
        m = min(1.0, step / (self.warmup * self.N)) if self.warmup > 0 else 1.0
        return m * self.eta_max * (self.floor + (1 - self.floor) * st), st


class Constant:
    """Clipped stock LR with optional zero-start linear warmup, then no decay."""
    def __init__(self, peak_lr=3e-3, min_lr=1e-6, max_lr=None, warmup_steps=0):
        self.peak_lr, self.min_lr = float(peak_lr), float(min_lr)
        self.max_lr = None if max_lr is None else float(max_lr)
        self.warmup_steps = warmup_steps

    def __call__(self, step, mult, state):
        lr = max(self.peak_lr * mult, self.min_lr)
        if self.max_lr is not None:
            lr = min(lr, self.max_lr)
        if self.warmup_steps and step <= self.warmup_steps:
            lr *= (step - 1) / self.warmup_steps
        return lr, mult


class Alchemy:
    """Port of deep-learning-alchemy/assignments lr_schedules.py (const / wsd<frac> / linear), with its
    defaults: warmup_steps = int(total * 0.01) = 93, linear from 0; the scheduler counts from 0, so the
    first optimizer step uses lr 0. Here every kind gets the warmup (their ConstantScheduler has none)."""
    def __init__(self, kind, eta0=ETA0, warmup_percent=0.01, decay_fraction=0.2, total_steps=TOTAL_STEPS):
        if kind not in ("const", "wsd", "linear"):
            raise ValueError(f"unknown kind {kind!r}")
        self.kind, self.eta0, self.N = kind, float(eta0), int(total_steps)
        self.W = int(self.N * warmup_percent)
        self.stable_end = self.N - int(self.N * decay_fraction)

    def __call__(self, step, mult, state):
        t = step - 1                                   # their last_epoch at this optimizer step
        if t < self.W:
            f = t / self.W
        elif self.kind == "const":
            f = 1.0
        else:
            start = self.W if self.kind == "linear" else self.stable_end
            f = 1.0 if t < start else 1 - min(1.0, (t - start) / max(1, self.N - start))
        return self.eta0 * f * mult, mult


SCHEDULES = {"wsd": WSD, "envelope": Envelope, "constant": Constant, "alchemy": Alchemy}

# Plain-schedule baselines matching the course repo's recipe (no stock signal, multiplier fixed at 1). Same init seed and batch order as the
# stock runs. Launched with `modal run modal_app.py::baselines`.
BASELINE_SCHEDULES = {k: {"name": "alchemy", "kind": k} for k in ("const", "wsd", "linear")}


def build_schedule(spec: dict):
    spec = dict(spec)
    name = spec.pop("name")
    if name not in SCHEDULES:
        raise KeyError(f"unknown schedule {name!r}; available: {sorted(SCHEDULES)}")
    return SCHEDULES[name](**spec)


def bars_to_steps(n_bars: int, n_steps: int = STEPS_PER_JOB):
    """Index of the bar each step uses: 15 bars -> 4 steps per bar; missing minutes spread the rest."""
    if n_bars <= 0:
        raise ValueError("need at least one bar")
    return (np.arange(n_steps) * n_bars // n_steps).astype(int)
