"""stock-lr on Modal: four d8 models whose learning rates follow ANT, SPCX, the S&P 500 (via SPY) and NVDA, plus a WSD baseline.

  uv run modal run modal_app.py::prepare_data     # once: 600k train seqs + val into the Volume (CPU)
  uv run modal run modal_app.py::apply_schema     # create/upgrade Supabase tables (idempotent)
  uv run modal run modal_app.py::check_setup      # secrets, Alpaca, Supabase reachability
  uv run modal run modal_app.py                   # end-to-end test on the last full hour (isolated, cleaned up)
  uv run modal deploy modal_app.py                # turn on the cron (weekdays 10:17-16:17 ET)
"""
import time
import modal

app = modal.App("stock-lr")
vol = modal.Volume.from_name("stock-lr", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "numpy", "psycopg[binary]==3.2.3")
    .add_local_python_source("stocklr", "vendor")
)
VOL = "/vol"
HF = "https://huggingface.co/datasets/kothasuhas/dl_alchemy_seq9p6m_context1024/resolve/main"
TRAIN_BYTES = 600_000 * 1024 * 2   # first 600k sequences, uint16
STEPS_PER_JOB = 240                # runs, tickers and the price -> LR mapper are in stocklr/config.py
HISTORY_DAYS = 8                   # calendar days of bars fetched for the trailing-average multiplier
secrets = modal.Secret.from_name("stock-lr-secrets")


@app.function(image=image, volumes={VOL: vol}, timeout=3600)
def prepare_data():
    import os, urllib.request
    os.makedirs(f"{VOL}/data", exist_ok=True)
    for url, out, rng in [(f"{HF}/val/tokens.bin", "val.bin", None),
                          (f"{HF}/train/tokens.bin", "train_600k.bin", f"bytes=0-{TRAIN_BYTES - 1}")]:
        req = urllib.request.Request(url, headers={"Range": rng} if rng else {})
        n = 0
        with urllib.request.urlopen(req) as r, open(f"{VOL}/data/{out}.tmp", "wb") as f:
            while chunk := r.read(64 << 20):
                f.write(chunk); n += len(chunk)
        os.replace(f"{VOL}/data/{out}.tmp", f"{VOL}/data/{out}")
        print(f"{out}: {n / 1e9:.2f} GB")
    assert os.path.getsize(f"{VOL}/data/train_600k.bin") == TRAIN_BYTES
    vol.commit()




@app.function(image=image, secrets=[secrets], timeout=120)
def check_setup():
    """Verify secrets (names + lengths only, never values), an Alpaca SIP + IEX pull, and Supabase reachability."""
    import json, os, urllib.request, urllib.error
    from datetime import datetime, timedelta, timezone
    for k in ["ALPACA_KEY_ID", "ALPACA_SECRET", "SUPABASE_URL", "SUPABASE_SERVICE_KEY"]:
        v = os.environ.get(k, "")
        print(f"{k:22s} {'set' if v else 'MISSING':8s} len={len(v)}")

    def get(url, headers):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    end = datetime.now(timezone.utc) - timedelta(minutes=16)
    start = end - timedelta(days=4)
    h = {"APCA-API-KEY-ID": os.environ.get("ALPACA_KEY_ID", ""), "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET", "")}
    for feed in ["sip", "iex"]:
        url = (f"https://data.alpaca.markets/v2/stocks/bars?symbols=NVDA,SPCX,SPY,ANT&timeframe=1Min&feed={feed}"
               f"&start={start:%Y-%m-%dT%H:%M:%SZ}&end={end:%Y-%m-%dT%H:%M:%SZ}&limit=10000&sort=desc")
        code, body = get(url, h)
        if code != 200:
            print(f"alpaca {feed}: HTTP {code} {body[:200]!r}"); continue
        bars = json.loads(body).get("bars") or {}
        summary = {s: (len(b), b[0]["t"], b[0]["c"]) for s, b in bars.items()}
        print(f"alpaca {feed}: OK  " + "  ".join(f"{s}: {n} bars, latest {t} close {c}" for s, (n, t, c) in summary.items()))

    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    code, body = get(f"{url}/rest/v1/", {"apikey": key, "Authorization": f"Bearer {key}"})
    print(f"supabase REST: HTTP {code} {'OK' if code == 200 else body[:200]!r}")


db_image = modal.Image.debian_slim(python_version="3.12").pip_install("psycopg[binary]==3.2.3").add_local_file("sql/schema.sql", "/root/schema.sql")


@app.function(image=db_image, secrets=[secrets], timeout=120)
def apply_schema():
    """Create/upgrade the Supabase tables (idempotent) and print what exists."""
    import os, psycopg
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise SystemExit("SUPABASE_DB_URL missing from stock-lr-secrets")
    with psycopg.connect(url, connect_timeout=15) as conn:
        conn.execute(open("/root/schema.sql").read())
        rows = conn.execute("""
            select c.relname, c.relrowsecurity,
                   (select count(*) from information_schema.columns k where k.table_schema='public' and k.table_name=c.relname),
                   (select count(*) from pg_policies p where p.schemaname='public' and p.tablename=c.relname)
            from pg_class c join pg_namespace n on n.oid=c.relnamespace
            where n.nspname='public' and c.relkind='r' order by 1""").fetchall()
    for name, rls, ncol, npol in rows:
        print(f"{name:8s} columns={ncol:2d}  row-level-security={'on' if rls else 'OFF'}  policies={npol}")


def run_job(window=None, test=False, force=False):
    """One hourly job. window=(start_utc, end_utc) overrides the clock (used by tests/backfills).
    test=True writes run names 'test-*' and checkpoints under /vol/ckpt_test so real state is untouched."""
    import csv, os, traceback
    from datetime import datetime, timedelta, timezone
    import numpy as np, psycopg
    from stocklr import db, mappers
    from stocklr.config import MAPPER_SPECS, PRICE_SOURCE, RUNS
    from stocklr.lr import bars_to_steps
    from stocklr.market import ET, fetch_bars, job_window
    from stocklr.train import Tokens, commit_checkpoints, train_hour

    t0 = time.perf_counter()
    win = window or job_window(datetime.now(timezone.utc))
    if win is None:
        print("outside market hours: nothing to do"); return {"status": "no-window"}
    start, end = win
    run_eval = test or end.astimezone(ET).hour % 2 == 0          # 10, 12, 14, 16 ET -> every 2 hours
    ckpt_dir = f"{VOL}/ckpt_test" if test else f"{VOL}/ckpt"
    prefix = "test-" if test else ""
    conn = psycopg.connect(os.environ["SUPABASE_DB_URL"], autocommit=True, connect_timeout=15)
    if not test and not force and db.already_done(conn, end):
        print(f"window ending {end} already trained: skipping"); return {"status": "duplicate"}
    built = {r: mappers.build(MAPPER_SPECS[r]) for r in RUNS}          # fail fast on a bad config
    TICKERS = sorted({PRICE_SOURCE[r] for r in RUNS if PRICE_SOURCE[r]})
    job_id = db.start_job(conn, start, end, run_eval, test, MAPPER_SPECS)
    print(f"job {job_id}: window {start.astimezone(ET):%a %H:%M}-{end.astimezone(ET):%H:%M} ET, eval={run_eval}, test={test}")
    try:
        bars = fetch_bars(TICKERS, end - timedelta(days=HISTORY_DAYS), end,
                          os.environ["ALPACA_KEY_ID"], os.environ["ALPACA_SECRET"])
        window_bars = {s: [b for b in bars[s] if b["ts"] >= start] for s in TICKERS}
        n_bars = {s: len(v) for s, v in window_bars.items()}
        print("bars in window:", n_bars, "| history:", {s: len(v) for s, v in bars.items()})
        if sum(n_bars.values()) == 0:
            db.finish_job(conn, job_id, "skipped", n_bars, time.perf_counter() - t0, "no bars (holiday/halt?)")
            return {"status": "skipped", "job_id": job_id}

        db.insert_prices(conn, window_bars, job_id)
        raw = f"{VOL}/prices/{end.astimezone(ET):%Y-%m-%d}"
        os.makedirs(raw, exist_ok=True)
        with open(f"{raw}/{end.astimezone(ET):%H%M}{'_test' if test else ''}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["ticker", "ts", "open", "high", "low", "close", "volume"])
            for s in TICKERS:
                for b in window_bars[s]:
                    w.writerow([s, b["ts"].isoformat(), b["open"], b["high"], b["low"], b["close"], b["volume"]])

        # price -> per-step multiplier, through whichever mapper config says (clipped + validated)
        even_ts = [start + (end - start) * i / STEPS_PER_JOB for i in range(STEPS_PER_JOB)]
        mults, step_ts = {}, {}
        for r in RUNS:
            src = PRICE_SOURCE[r]
            n = n_bars[src] if src else 0
            if not src:                       # e.g. BASELINE: mapper sees a flat dummy series
                flat = mappers.Bars.from_rows([{"ts": t, "close": 1.0} for t in even_ts])
                mults[r], step_ts[r] = mappers.safe_multipliers(built[r], flat), even_ts
                continue
            if n == 0:                        # ticker didn't trade this hour (halt): hold at 1.0
                print(f"WARNING {r}: no {src} bars this hour, multiplier held at 1.0")
                mults[r], step_ts[r] = np.ones(STEPS_PER_JOB), even_ts
                continue
            m = mappers.safe_multipliers(built[r], mappers.Bars.from_rows(bars[src]))[-n:]
            idx = bars_to_steps(n, STEPS_PER_JOB)
            mults[r], step_ts[r] = m[idx], [window_bars[src][i]["ts"] for i in idx]

        tokens = Tokens(f"{VOL}/data")
        os.makedirs(ckpt_dir, exist_ok=True)
        results = {}
        for r in RUNS:
            res = train_hour(prefix + r, mults[r], tokens, ckpt_dir, run_eval)
            results[r] = res
            if res["done"]:
                print(f"{r}: run complete at step {res['start_step']}, nothing to train"); continue
            print(f"{r}: steps {res['start_step'] + 1}-{res['end_step']}  lr {min(x[1] for x in res['steps']):.2e}-"
                  f"{max(x[1] for x in res['steps']):.2e}  loss {res['steps'][-1][3]:.3f}"
                  + (f"  val {res['val_loss']:.3f}" if res['val_loss'] is not None else "")
                  + f"  ({res['ms_per_step']:.0f} ms/step)")

        step_rows, eval_rows = [], []
        for r, res in results.items():
            ts_list = step_ts[r]
            for i, (step, lr, mult, loss) in enumerate(res["steps"]):
                step_rows.append((prefix + r, step, ts_list[i], lr, mult, loss, job_id))
            if res["val_loss"] is not None:
                eval_rows.append((prefix + r, res["end_step"], res["val_loss"], job_id))
        db.insert_steps(conn, step_rows)
        db.insert_evals(conn, eval_rows)
        commit_checkpoints(ckpt_dir, [prefix + r for r in RUNS])
        vol.commit()
        dur = time.perf_counter() - t0
        db.finish_job(conn, job_id, "ok", n_bars, dur)
        print(f"job {job_id} ok in {dur:.0f}s: {len(step_rows)} step rows, {len(eval_rows)} evals")
        return {"status": "ok", "job_id": job_id, "n_steps": len(step_rows), "n_evals": len(eval_rows), "duration_s": dur}
    except Exception:
        db.finish_job(conn, job_id, "failed", None, time.perf_counter() - t0, traceback.format_exc()[-4000:])
        raise
    finally:
        conn.close()


# The real thing. Schedule only takes effect after `modal deploy`.
@app.function(image=image, gpu="H100", volumes={VOL: vol}, secrets=[secrets], timeout=1800, retries=0,
              schedule=modal.Cron("17 10-16 * * 1-5", timezone="America/New_York"))
def hourly():
    return run_job()


@app.function(image=image, gpu="H100", volumes={VOL: vol}, secrets=[secrets], timeout=1800)
def test_job(start_iso: str, end_iso: str, keep: bool = False):
    """Run the full job on a past window in test mode, check what landed in Supabase, then clean up."""
    import os, shutil
    from datetime import datetime
    import psycopg
    from stocklr import db
    res = run_job(window=(datetime.fromisoformat(start_iso), datetime.fromisoformat(end_iso)), test=True)
    jid = res["job_id"]
    with psycopg.connect(os.environ["SUPABASE_DB_URL"], autocommit=True) as conn:
        q = lambda sql: conn.execute(sql, (jid,)).fetchall()
        status = q("select status, n_bars, round(duration_s) from jobs where id=%s")[0]
        steps = q("select run, count(*), min(step), max(step), min(lr), max(lr), min(lr_mult), max(lr_mult) from steps where job_id=%s group by run order by run")
        evals = q("select run, step, round(val_loss::numeric, 3) from evals where job_id=%s order by run")
        n_prices = q("select ticker, count(*) from prices where job_id=%s group by ticker order by ticker")
        print("\n--- what landed in Supabase ---")
        print("job:", status)
        print("prices:", n_prices)
        for row in steps:
            print(f"steps {row[0]:14s} n={row[1]} steps {row[2]}-{row[3]} lr {row[4]:.2e}-{row[5]:.2e} mult {row[6]:.2f}-{row[7]:.2f}")
        print("evals:", evals)
        checks = {
            "job ok": status[0] == "ok",
            "5 runs x 240 steps": len(steps) == 5 and all(r[1] == STEPS_PER_JOB for r in steps),
            "5 evals": len(evals) == 5,
            "bars pulled": sum((status[1] or {}).values()) > 0,
            "baseline mult == 1": all(r[6] == r[7] == 1.0 for r in steps if r[0].endswith("BASELINE")),
            "mapper specs recorded": q("select mappers is not null from jobs where id=%s")[0][0],
            "multipliers within (0.5, 2)": all(0.5 < r[6] and r[7] < 2.0 for r in steps),
        }
        if not keep:
            db.delete_job(conn, jid)
            shutil.rmtree(f"{VOL}/ckpt_test", ignore_errors=True); vol.commit()
            print("cleaned up test rows and checkpoints")
    for k, v in checks.items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        raise SystemExit("end-to-end test FAILED")
    return checks


@app.local_entrypoint()
def main(start: str = "", end: str = "", keep: bool = False):
    """Default: test on the most recent full regular-session hour that is at least 16 minutes old."""
    from datetime import datetime, timedelta, timezone
    from stocklr.market import ET, job_window
    if not start:
        t = datetime.now(timezone.utc) - timedelta(minutes=16)
        while (w := job_window(t)) is None or (w[1] - w[0]) < timedelta(hours=1):
            t -= timedelta(hours=1)
        start, end = w[0].isoformat(), w[1].isoformat()
    print(f"end-to-end test on {datetime.fromisoformat(start).astimezone(ET):%a %b %d %H:%M}-"
          f"{datetime.fromisoformat(end).astimezone(ET):%H:%M} ET")
    test_job.remote(start, end, keep)
