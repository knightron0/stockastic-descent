"""Fetch the last 5 trading days of 1-minute bars (Yahoo Finance, free, no key) -> research/data/<TICKER>_1m.csv"""
import sys
from pathlib import Path
import yfinance as yf

out = Path(__file__).parent / "data"; out.mkdir(exist_ok=True)
for t in sys.argv[1:] or ["SPCX"]:
    df = yf.download(t, period="5d", interval="1m", prepost=False, progress=False, auto_adjust=False)
    if df.empty:
        print(f"{t}: no data"); continue
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.index = df.index.tz_convert("America/New_York")
    df[["Open", "High", "Low", "Close", "Volume"]].to_csv(out / f"{t}_1m.csv")
    days = df.groupby(df.index.date).size()
    print(f"{t}: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")
    print("  per day:", dict((str(k), int(v)) for k, v in days.items()))
    print(f"  close range {df.Close.min():.2f}-{df.Close.max():.2f}")
