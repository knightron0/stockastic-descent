"""research/data/*_1m.csv -> compact JSON {ticker: {t: [unix s], p: [close]}} (drops zero-volume placeholder bars)."""
import csv, json, sys
from datetime import datetime
from pathlib import Path

d = Path(__file__).parent / "data"
out = {}
for t in ["SPCX", "GOOGL", "NVDA", "ANT"]:
    rows = [r for r in csv.DictReader(open(d / f"{t}_1m.csv")) if float(r["Volume"]) > 0]
    out[t] = {"t": [int(datetime.fromisoformat(r["Datetime"]).timestamp()) for r in rows],
              "p": [round(float(r["Close"]), 3) for r in rows]}
    print(t, len(rows), file=sys.stderr)
print(json.dumps(out, separators=(",", ":")))
