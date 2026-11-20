"""backups/<tag>/*.csv -> web/public/backups/<tag>.json in the dashboard's Raw shape (for ?backup=<tag>)."""
import csv, json, sys
from datetime import datetime
from pathlib import Path

tag = sys.argv[1] if len(sys.argv) > 1 else "thursday-2026-11-19"
src = Path("backups") / tag
RUNS = {"ANT", "SPCX", "SP500", "NVDA"}
iso = lambda s: datetime.fromisoformat(s).isoformat() if s else None
num = lambda s: None if s in ("", None) else float(s)
rows = lambda t: list(csv.DictReader(open(src / f"{t}.csv")))

raw = {
    "steps": [{"run": r["run"], "step": int(r["step"]), "ts": iso(r["ts"]), "lr": num(r["lr"]), "lr_mult": num(r["lr_mult"]),
               "train_loss": num(r["train_loss"]), "job_id": int(r["job_id"])} for r in rows("steps") if r["run"] in RUNS],
    "evals": [{"run": r["run"], "step": int(r["step"]), "ts": iso(r["ts"]), "val_loss": num(r["val_loss"])}
              for r in rows("evals") if r["run"] in RUNS],
    "prices": [{"ticker": r["ticker"], "ts": iso(r["ts"]), **{k: num(r[k]) for k in ("open", "high", "low", "close", "volume")},
                "job_id": int(r["job_id"])} for r in rows("prices")],
    "jobs": sorted([{"id": int(r["id"]), "started_at": iso(r["started_at"]), "finished_at": iso(r["finished_at"]),
              "window_start": iso(r["window_start"]), "window_end": iso(r["window_end"]), "status": r["status"],
              "ran_eval": r["ran_eval"] == "True", "n_bars": json.loads(r["n_bars"]) if r["n_bars"] else None,
              "duration_s": num(r["duration_s"]), "error": r["error"] or None,
              "mappers": {k: v for k, v in json.loads(r["mappers"]).items() if k != "BASELINE"} if r["mappers"] else None}
             for r in rows("jobs")], key=lambda j: j["window_end"], reverse=True),
}
out = Path("web/public/backups") / f"{tag}.json"
out.write_text(json.dumps(raw, separators=(",", ":")))
print(out, {k: len(v) for k, v in raw.items()})
