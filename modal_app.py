"""stock-lr on Modal: four d8 models whose learning rates follow ANT, SPCX, the S&P 500 (via SPY) and NVDA, plus a WSD baseline.

  uv run modal run modal_app.py::prepare_data     # once: 600k train seqs + val into the Volume (CPU)
  uv run modal run modal_app.py::apply_schema     # create/upgrade Supabase tables (idempotent)
  uv run modal run modal_app.py::check_setup      # secrets, Alpaca, Supabase reachability
  uv run modal run modal_app.py                   # end-to-end test on the last full 15-min window (isolated, cleaned up)
  uv run modal deploy modal_app.py                # turn on the cron (weekdays every 15 min, 10:02-16:17 ET)
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
from stocklr.lr import STEPS_PER_JOB  # 60 per 15-min job; runs, tickers and the price -> LR mapper are in stocklr/config.py
EVAL_EVERY = 60                    # val loss at every multiple of this step, plus at each job's end (= once per job)
JOB_TIMEOUT_S = 720                # must stay under the 15-min cron gap so jobs can never overlap
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
    """One 15-minute job. window=(start_utc, end_utc) overrides the clock (used by tests/backfills).
    test=True writes run names 'test-*' and checkpoints under /vol/ckpt_test so real state is untouched."""
    import csv, os, traceback
    from datetime import datetime, timedelta, timezone
    import numpy as np, psycopg
    from stocklr import db, mappers
    from stocklr.config import HISTORY_WARMUP_DAYS, MAPPER_SPECS, PRICE_SOURCE, RUNS, SCHEDULE
    from stocklr.lr import bars_to_steps, build_schedule
    from stocklr.market import ET, fetch_bars, job_window
    from stocklr.train import Tokens, commit_checkpoints, train_window

    t0 = time.perf_counter()
    win = window or job_window(datetime.now(timezone.utc))
    if win is None:
        print("outside market hours: nothing to do"); return {"status": "no-window"}
    start, end = win
    run_eval = True                                               # validation every EVAL_EVERY steps + at job end
    ckpt_dir = f"{VOL}/ckpt_test" if test else f"{VOL}/ckpt"
    prefix = "test-" if test else ""
    conn = psycopg.connect(os.environ["SUPABASE_DB_URL"], autocommit=True, connect_timeout=15)
    if not test and not force and db.already_done(conn, start, end):
        print(f"window ending {end} already trained: skipping"); return {"status": "duplicate"}
    if not test and (busy := db.job_in_flight(conn)):
        print(f"job {busy} is still running: skipping this window"); return {"status": "busy"}
    built = {r: mappers.build(MAPPER_SPECS[r]) for r in RUNS}          # fail fast on a bad config
    schedule = build_schedule(SCHEDULE)
    TICKERS = sorted({PRICE_SOURCE[r] for r in RUNS if PRICE_SOURCE[r]})
    # history always starts at the same anchor (run start - 30 days), so the EMA state is identical to
    # one updated bar by bar since then, no matter which window this job is.
    run_start = None if test else db.run_anchor(conn)
    history_from = (run_start or start) - timedelta(days=HISTORY_WARMUP_DAYS)
    job_id = db.start_job(conn, start, end, run_eval, test, {**MAPPER_SPECS, "_schedule": SCHEDULE})
    print(f"job {job_id}: window {start.astimezone(ET):%a %H:%M}-{end.astimezone(ET):%H:%M} ET, eval={run_eval}, test={test}")
    try:
        bars = fetch_bars(TICKERS, history_from, end,
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
            if not src:                       # run without a ticker: mapper sees a flat dummy series
                flat = mappers.Bars.from_rows([{"ts": t, "close": 1.0} for t in even_ts])
                mults[r], step_ts[r] = mappers.safe_multipliers(built[r], flat), even_ts
                continue
            if n == 0:                        # no bars this window (halt): hold the last value, as when closed
                hist = mappers.safe_multipliers(built[r], mappers.Bars.from_rows(bars[src])) if bars[src] else []
                held = float(hist[-1]) if len(hist) else getattr(built[r], "neutral", 1.0)
                print(f"WARNING {r}: no {src} bars this window, holding signal at {held:.3f}")
                mults[r], step_ts[r] = np.full(STEPS_PER_JOB, held), even_ts
                continue
            m = mappers.safe_multipliers(built[r], mappers.Bars.from_rows(bars[src]))[-n:]
            idx = bars_to_steps(n, STEPS_PER_JOB)
            mults[r], step_ts[r] = m[idx], [window_bars[src][i]["ts"] for i in idx]

        tokens = Tokens(f"{VOL}/data")
        os.makedirs(ckpt_dir, exist_ok=True)
        results = {}
        for r in RUNS:
            res = train_window(prefix + r, mults[r], schedule, tokens, ckpt_dir, run_eval, EVAL_EVERY)
            results[r] = res
            if res["done"]:
                print(f"{r}: run complete at step {res['start_step']}, nothing to train"); continue
            print(f"{r}: steps {res['start_step'] + 1}-{res['end_step']}  lr {min(x[1] for x in res['steps']):.2e}-"
                  f"{max(x[1] for x in res['steps']):.2e}  loss {res['steps'][-1][3]:.3f}"
                  + "".join(f"  val@{s} {v:.3f}" for s, v in res["evals"])
                  + f"  ({res['ms_per_step']:.0f} ms/step)")

        step_rows, eval_rows = [], []
        for r, res in results.items():
            ts_list = step_ts[r]
            for i, (step, lr, mult, loss) in enumerate(res["steps"]):
                step_rows.append((prefix + r, step, ts_list[i], lr, mult, loss, job_id))
            for step, vl in res["evals"]:
                eval_rows.append((prefix + r, step, vl, job_id))
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
# GPU fallbacks: H100 first; if none is free, the next one in the list.
# Fires at :02/:17/:32/:47, training the window that ended 17 min earlier (bars are 15 min delayed):
# 10:02 -> 9:30-9:45 ... 16:17 -> 15:45-16:00. 16:32 and 16:47 find no window and exit at once.
@app.function(image=image, gpu=["H100", "A100-80GB", "L40S"], volumes={VOL: vol}, secrets=[secrets], timeout=JOB_TIMEOUT_S,
              retries=0, schedule=modal.Cron("2,17,32,47 10-16 * * 1-5", timezone="America/New_York"))
def every_15_min():
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
            f"4 runs x {STEPS_PER_JOB} steps": len(steps) == 4 and all(r[1] == STEPS_PER_JOB for r in steps),
            f"4 runs x {STEPS_PER_JOB // EVAL_EVERY} evals": len(evals) == 4 * (STEPS_PER_JOB // EVAL_EVERY),
            "bars pulled": sum((status[1] or {}).values()) > 0,
            "mapper specs recorded": q("select mappers is not null from jobs where id=%s")[0][0],
            "zero LR only on step one": q("select count(*) from steps where job_id=%s and "
                                           "((step=1 and lr<>0) or (step>1 and lr<=0))")[0][0] == 0,
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
    """Default: test on the most recent full 15-minute regular-session window that is at least 16 minutes old."""
    from datetime import datetime, timedelta, timezone
    from stocklr.market import ET, JOB_MINUTES, job_window
    if not start:
        t = datetime.now(timezone.utc)
        while (w := job_window(t)) is None or (w[1] - w[0]) < timedelta(minutes=JOB_MINUTES):
            t -= timedelta(minutes=JOB_MINUTES)
        start, end = w[0].isoformat(), w[1].isoformat()
    print(f"end-to-end test on {datetime.fromisoformat(start).astimezone(ET):%a %b %d %H:%M}-"
          f"{datetime.fromisoformat(end).astimezone(ET):%H:%M} ET")
    test_job.remote(start, end, keep)


@app.function(image=image, secrets=[secrets], timeout=600)
def preview_lrs(first_day: str, last_day: str):
    """CPU only, no training: replay the cron's LR computation over past trading days (same mapper, schedule,
    history anchor and steps-per-job mapping). Returns per-step LRs + per-bar prices/signals."""
    import os
    from datetime import date, datetime, timedelta
    import numpy as np
    from stocklr import mappers
    from stocklr.config import HISTORY_WARMUP_DAYS, MAPPER_SPECS, PRICE_SOURCE, RUNS, SCHEDULE
    from stocklr.lr import bars_to_steps, build_schedule
    from stocklr.market import ET, fetch_bars, job_windows_for_day

    d0, d1 = date.fromisoformat(first_day), date.fromisoformat(last_day)
    windows, d = [], d0
    while d <= d1:
        windows += job_windows_for_day(d)
        d += timedelta(days=1)
    tickers = sorted({PRICE_SOURCE[r] for r in RUNS if PRICE_SOURCE[r]})
    bars = fetch_bars(tickers, windows[0][0] - timedelta(days=HISTORY_WARMUP_DAYS), windows[-1][1],
                      os.environ["ALPACA_KEY_ID"], os.environ["ALPACA_SECRET"])
    built = {r: mappers.build(MAPPER_SPECS[r]) for r in RUNS}
    sigs = {s: mappers.safe_multipliers(built[next(r for r in RUNS if PRICE_SOURCE[r] == s)], mappers.Bars.from_rows(bars[s]))
            for s in tickers}
    out = {r: {"lr": [], "s": [], "s_tilde": [], "ts": []} for r in RUNS}
    state = {r: {} for r in RUNS}
    sched = build_schedule(SCHEDULE)
    step = 0
    for start, end in windows:
        for r in RUNS:
            src = PRICE_SOURCE[r]
            if src:
                idx_all = [i for i, b in enumerate(bars[src]) if start <= b["ts"] < end]
                if not idx_all: continue
                pick = [idx_all[i] for i in bars_to_steps(len(idx_all))]
                vals, tss = sigs[src][pick], [bars[src][i]["ts"].isoformat() for i in pick]
            else:
                vals = np.full(STEPS_PER_JOB, built[r].neutral)
                tss = [(start + (end - start) * i / STEPS_PER_JOB).isoformat() for i in range(STEPS_PER_JOB)]
            for i, v in enumerate(vals):
                lr, st = sched(step + i + 1, float(v), state[r])
                out[r]["lr"].append(lr); out[r]["s"].append(float(v)); out[r]["s_tilde"].append(st); out[r]["ts"].append(tss[i])
        step += STEPS_PER_JOB
    prices = {s: {"ts": [b["ts"].isoformat() for b in bars[s] if b["ts"] >= windows[0][0]],
                  "close": [b["close"] for b in bars[s] if b["ts"] >= windows[0][0]],
                  "s": [float(v) for b, v in zip(bars[s], sigs[s]) if b["ts"] >= windows[0][0]]} for s in tickers}
    return {"runs": out, "prices": prices, "n_steps": step, "history_bars": {s: len(v) for s, v in bars.items()}}


@app.local_entrypoint()
def preview(first: str = "2026-11-13", last: str = "2026-11-19", out: str = "research/lr_preview.json"):
    import json
    res = preview_lrs.remote(first, last)
    open(out, "w").write(json.dumps(res))
    print(f"{res['n_steps']} steps, history bars {res['history_bars']} -> {out}")


@app.function(image=image, secrets=[secrets], timeout=1800)
def replay_week(first_day: str = "2026-11-09"):
    """Publish a new independent training replay to Supabase Storage."""
    import json
    import os
    import uuid
    from datetime import datetime, timedelta, timezone
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen
    from stocklr.config import HISTORY_WARMUP_DAYS, MAPPER_SPECS, PRICE_SOURCE, SCHEDULE
    from stocklr.market import ET, fetch_bars
    from stocklr.lr import TOTAL_STEPS
    from stocklr.replay import REPLAY_RUNS, replay_batches

    start = datetime.strptime(first_day, "%Y-%m-%d").replace(hour=9, minute=30, tzinfo=ET)
    if start.weekday() != 0:
        raise ValueError("first_day must be a Monday")
    end = (start + timedelta(days=4)).replace(hour=16, minute=0)
    symbols = [PRICE_SOURCE[r] for r in REPLAY_RUNS]
    history = fetch_bars(symbols, (start - timedelta(days=HISTORY_WARMUP_DAYS)).astimezone(timezone.utc),
                         end.astimezone(timezone.utc),
                         os.environ["ALPACA_KEY_ID"], os.environ["ALPACA_SECRET"])
    origin = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_KEY"]
    bucket = "stock-lr-replays"
    replay_id = uuid.uuid4().hex
    def storage(path, payload=None):
        headers = {"Authorization": "Bearer " + key, "apikey": key, "Content-Type": "application/json",
                   "Cache-Control": "max-age=0", "x-upsert": "true"}
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode() if payload is not None else None
        for attempt in range(4):
            try:
                with urlopen(Request(origin + "/storage/v1/" + path, data=body, headers=headers), timeout=30) as res:
                    return json.load(res)
            except HTTPError as exc:
                if exc.code < 500 and exc.code != 429 or attempt == 3:
                    raise
            except URLError:
                if attempt == 3:
                    raise
            time.sleep(2 ** attempt)

    try:
        info = storage("bucket/" + bucket)
    except HTTPError as exc:
        detail = exc.read()
        if exc.code != 404 and b"Bucket not found" not in detail:
            raise
        storage("bucket", {"id": bucket, "name": bucket, "public": True})
        info = storage("bucket/" + bucket)
    if not info.get("public"):
        raise ValueError("Replay bucket is private")
    prefix = f"object/{bucket}/{replay_id}/"
    chunks = []
    for step, batch in replay_batches(history, start, end):
        stamps = [row["ts"] for row in batch["steps"]]
        now = datetime.now(timezone.utc).isoformat()
        batch["jobs"].append(dict(id=step, started_at=now, finished_at=now, window_start=min(stamps),
                                  window_end=max(stamps), status="ok", ran_eval=bool(batch["evals"]),
                                  n_bars=None, duration_s=None, error=None, mappers={**MAPPER_SPECS, "_schedule": SCHEDULE}))
        storage(prefix + f"{step}.json", batch)
        chunks.append(step)
        storage(prefix + "manifest.json", {"chunks": chunks, "synthetic_losses": True})
        if step == 480:
            print(f"Replay: ?replay={origin.split('//')[1].split('.')[0]}/{replay_id}", flush=True)
        print(f"Published step {step}", flush=True)
        if step < TOTAL_STEPS:
            time.sleep(5)


@app.function(image=image, gpu="B200", volumes={VOL: vol}, secrets=[secrets], timeout=3600)
def backfill(day: str):
    """Run the real (non-test) 15-minute job for every finished window of a past trading day, in order.
    Already-trained windows are skipped, so it's safe to re-run. Do NOT run while the cron is mid-job."""
    from datetime import date, datetime, timedelta, timezone
    from stocklr.market import ET, job_windows_for_day
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=16)
    wins = [w for w in job_windows_for_day(date.fromisoformat(day)) if w[1] <= cutoff]
    print(f"{day}: {len(wins)} finished windows")
    out = []
    for w in wins:
        res = run_job(window=w)
        print(f"  {w[0].astimezone(ET):%H:%M}-{w[1].astimezone(ET):%H:%M} ET -> {res['status']}")
        out.append(res)
        if res["status"] not in ("ok", "duplicate", "skipped"):   # "busy": the cron is mid-job, stop
            raise RuntimeError(f"stopping: {res}")
    return out


@app.local_entrypoint()
def backfill_day(day: str):
    for r in backfill.remote(day):
        print(r)


@app.function(image=image, gpu="H100", volumes={VOL: vol}, timeout=4 * 3600, retries=2)
def baseline_run(name: str, chunk: int = 600):
    """One full plain-schedule run (stocklr.lr.BASELINE_SCHEDULES[name]), mult fixed at 1, same seed and batch
    order as the stock runs. Checkpoints + log under /vol/baselines/ after every chunk, so a retry resumes.
    Touches neither Supabase nor the cron's checkpoints."""
    import json, os
    from stocklr.lr import BASELINE_SCHEDULES, TOTAL_STEPS, build_schedule
    from stocklr.train import Tokens, commit_checkpoints, train_window
    d = f"{VOL}/baselines"
    os.makedirs(d, exist_ok=True)
    log_path = f"{d}/{name}.json"
    log = json.load(open(log_path)) if os.path.exists(log_path) else {"schedule": BASELINE_SCHEDULES[name], "steps": [], "evals": []}
    sched, tokens = build_schedule(BASELINE_SCHEDULES[name]), Tokens(f"{VOL}/data")
    t0 = time.perf_counter()
    while True:
        res = train_window(name, [1.0] * chunk, sched, tokens, d, True, EVAL_EVERY)
        if res["done"]:
            break
        s0 = res["start_step"]                       # resume: drop log rows past the checkpoint
        log["steps"] = [r for r in log["steps"] if r[0] <= s0]; log["evals"] = [e for e in log["evals"] if e[0] <= s0]
        log["steps"] +=[[s, lr, loss] for s, lr, _, loss in res["steps"]]
        log["evals"] += res["evals"]
        json.dump(log, open(f"{log_path}.tmp", "w")); os.replace(f"{log_path}.tmp", log_path)
        commit_checkpoints(d, [name]); vol.commit()
        print(f"{name}: step {res['end_step']}/{TOTAL_STEPS}  lr {res['steps'][-1][1]:.2e}  val {res['val_loss']:.4f}"
              f"  ({res['ms_per_step']:.0f} ms/step, {time.perf_counter() - t0:.0f}s so far)", flush=True)
    return log


@app.local_entrypoint()
def baselines(names: str = "const,wsd,linear"):
    """uv run modal run --detach modal_app.py::baselines   -> three H100s in parallel, logs to research/baselines/"""
    import json, pathlib
    out = pathlib.Path("research/baselines"); out.mkdir(parents=True, exist_ok=True)
    names = names.split(",")
    for n, log in zip(names, baseline_run.map(names)):
        (out / f"{n}.json").write_text(json.dumps(log))
        print(f"{n}: final val {log['evals'][-1][1]:.4f} -> {out / n}.json")


TABLES = ["jobs", "prices", "steps", "evals"]


@app.function(image=image, volumes={VOL: vol}, secrets=[secrets], timeout=1800)
def backup_and_reset(tag: str, wipe: bool = False):
    """Back up every Supabase row + /vol/ckpt + /vol/prices to /vol/backups/<tag>/, verify it, and (wipe=True)
    then delete those rows and directories so the next cron job starts a fresh run at step 1.
    /vol/data (training sequences) and /vol/backups are never touched."""
    import json, os, shutil
    import psycopg
    out = f"{VOL}/backups/{tag}"
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; pick another tag")
    os.makedirs(out)
    dump = {}
    with psycopg.connect(os.environ["SUPABASE_DB_URL"], autocommit=True) as conn:
        running = conn.execute("select count(*) from jobs where status = 'running'").fetchone()[0]
        if running:
            raise SystemExit(f"{running} job(s) still running; try again after they finish")
        for t in TABLES:
            cur = conn.execute(f"select * from {t}")
            cols = [c.name for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            dump[t] = rows
            with open(f"{out}/{t}.json", "w") as f:
                json.dump(rows, f, default=str)
        for d in ("ckpt", "prices"):
            if os.path.exists(f"{VOL}/{d}"):
                shutil.copytree(f"{VOL}/{d}", f"{out}/{d}")
        vol.commit()
        # verify before deleting anything
        for t in TABLES:
            n = len(json.load(open(f"{out}/{t}.json")))
            assert n == len(dump[t]), f"backup mismatch on {t}"
        ck = sorted(os.listdir(f"{out}/ckpt")) if os.path.exists(f"{out}/ckpt") else []
        for name in ck:
            assert os.path.getsize(f"{out}/ckpt/{name}") == os.path.getsize(f"{VOL}/ckpt/{name}"), name
        summary = {t: len(dump[t]) for t in TABLES} | {"checkpoints": ck}
        print("backed up + verified:", summary, "->", out)
        if wipe:
            for t in ["steps", "evals", "prices", "jobs"]:          # children first (job_id foreign keys)
                conn.execute(f"delete from {t}")
            for d in ("ckpt", "ckpt_test", "prices"):
                shutil.rmtree(f"{VOL}/{d}", ignore_errors=True)
            vol.commit()
            left = {t: conn.execute(f"select count(*) from {t}").fetchone()[0] for t in TABLES}
            print("wiped. rows left:", left, "| volume now:", sorted(os.listdir(VOL)))
    return {"summary": summary, "dump": dump}


@app.local_entrypoint()
def reset(tag: str, wipe: bool = False):
    """uv run modal run modal_app.py::reset --tag <name> [--wipe]   (also saves a local copy under backups/<tag>/)"""
    import csv, json, pathlib
    res = backup_and_reset.remote(tag, wipe)
    local = pathlib.Path("backups") / tag
    local.mkdir(parents=True, exist_ok=True)
    for t, rows in res["dump"].items():
        with open(local / f"{t}.csv", "w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
                for r in rows:
                    w.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()})
    print("local copy:", local.resolve(), res["summary"])
