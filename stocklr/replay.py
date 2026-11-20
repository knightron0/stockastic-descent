"""Build a paced replay from recorded market bars. Only the losses are synthetic."""
import math
import random

from .config import MAPPER_SPECS, PRICE_SOURCE, SCHEDULE, TICKERS
from .lr import TOTAL_STEPS, bars_to_steps, build_schedule
from .mappers import Bars, build, safe_multipliers

REPLAY_RUNS = tuple(r for r in TICKERS if r != "ANT")


def replay_batches(history, start, end):
    """Distribute one full run across each symbol's observed bars in the replay week."""
    runs = {}
    for run in REPLAY_RUNS:
        symbol = PRICE_SOURCE[run]
        bars = history[symbol]
        indices = [i for i, bar in enumerate(bars) if start <= bar["ts"] < end]
        if not indices:
            raise ValueError(f"No real bars for {symbol} during the replay week")
        signals = safe_multipliers(build(MAPPER_SPECS[run]), Bars.from_rows(bars))
        runs[run] = (bars, [indices[i] for i in bars_to_steps(len(indices), TOTAL_STEPS)], signals)

    states = {r: {} for r in REPLAY_RUNS}
    rng = {r: random.Random(r) for r in REPLAY_RUNS}
    seen = set()
    schedule = build_schedule(SCHEDULE)
    first = 1
    while first <= TOTAL_STEPS:
        last = min(first + (479 if first == 1 else 239), TOTAL_STEPS)
        batch = {"steps": [], "prices": [], "evals": [], "jobs": []}
        for run, (bars, indices, signals) in runs.items():
            for step in range(first, last + 1):
                i = indices[step - 1]
                bar = bars[i]
                ts = bar["ts"].isoformat()
                symbol = PRICE_SOURCE[run]
                if (symbol, ts) not in seen:
                    batch["prices"].append(dict(bar, ts=ts, ticker=symbol, job_id=last))
                    seen.add((symbol, ts))
                lr, mult = schedule(step, signals[i], states[run])
                trend = 2.9 + 2.1 * math.exp(-step / 1900)
                batch["steps"].append(dict(run=run, step=step, ts=ts, lr=lr, lr_mult=mult,
                                           train_loss=round(trend + rng[run].gauss(0, .09), 5), job_id=last))
                if step % 480 == 0 or step == TOTAL_STEPS:
                    batch["evals"].append(dict(run=run, step=step, ts=ts, job_id=last,
                                               val_loss=round(trend + .12 + rng[run].gauss(0, .025), 5)))
        yield last, batch
        first = last + 1
