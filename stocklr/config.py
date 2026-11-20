"""Which mapper each run uses. Change this (and redeploy) to switch the price -> LR formula.
Each job records the specs it used in jobs.mappers, so the dashboard can tell formulas apart."""
TICKERS = ["ANT", "SPCX", "SP500", "NVDA"]   # run names
BASELINE = "BASELINE"
RUNS = TICKERS + [BASELINE]

TICKER_MAPPER = {"name": "detrended", "k": 15.0, "span": 390, "smooth": 10}

MAPPER_SPECS = {**{t: TICKER_MAPPER for t in TICKERS}, BASELINE: {"name": "constant"}}

# which Alpaca symbol each run's prices come from (None = no prices needed).
# SP500 uses SPY, the ETF that tracks the S&P 500 (Alpaca has no index bars).
PRICE_SOURCE = {"ANT": "ANT", "SPCX": "SPCX", "SP500": "SPY", "NVDA": "NVDA", BASELINE: None}
