"""Experiment config: runs, the price -> signal mapper, and the schedule that turns the signal into an LR.
Change this (and redeploy) to switch formulas. Each job records MAPPER_SPECS + SCHEDULE in jobs.mappers,
so the dashboard can tell formulas apart."""
TICKERS = ["ANT", "SPCX", "SP500", "NVDA"]   # run names
RUNS = TICKERS                               # no baseline run

# lr = peak_lr * (price / starting price), starting price = first bar of the run (first real cron job).
# k=1: a 10% move changes the LR by 10%. smooth=1: no smoothing. The LR itself is clipped by SCHEDULE.
TICKER_MAPPER = {"name": "price_ratio", "k": 1.0, "smooth": 1}

MAPPER_SPECS = {t: TICKER_MAPPER for t in TICKERS}

# Warm up the clipped stock LR from zero on step 1 to full strength on step 94.
SCHEDULE = {"name": "constant", "peak_lr": 3e-3, "min_lr": 1e-6, "max_lr": None, "warmup_steps": 93}

# History fed to the mapper starts this many days before the run's first job. price_ratio divides by the
# first bar it sees, so 0 makes that the run's starting price. (ema_sigmoid needs 30.)
HISTORY_WARMUP_DAYS = 0

# Previous formula, kept for reference:
#   TICKER_MAPPER = {"name": "ema_sigmoid", "half_life_days": 5.0, "width": 1.0, "bars_per_session": 390}
#   BASELINE mapper {"name": "constant", "value": 0.5}
#   SCHEDULE = {"name": "envelope", "eta_max": 3e-3, "floor": 0.1, "warmup": 0.0, "max_rise": 0.5}
#   HISTORY_WARMUP_DAYS = 30

# which Alpaca symbol each run's prices come from (None = no prices needed).
# SP500 uses SPY, the ETF that tracks the S&P 500 (Alpaca has no index bars).
PRICE_SOURCE = {"ANT": "ANT", "SPCX": "SPCX", "SP500": "SPY", "NVDA": "NVDA"}
