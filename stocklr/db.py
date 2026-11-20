"""Supabase Postgres writes for the hourly job (psycopg 3)."""
import json


def already_done(conn, window_end):
    row = conn.execute("select 1 from jobs where window_end = %s and status = 'ok' and not is_test limit 1",
                       (window_end,)).fetchone()
    return row is not None


def run_anchor(conn):
    """window_start of the first successful real job: the fixed start of the run (None before it begins)."""
    return conn.execute("select min(window_start) from jobs where status = 'ok' and not is_test").fetchone()[0]


def start_job(conn, window_start, window_end, ran_eval, is_test, mapper_specs):
    return conn.execute(
        "insert into jobs (window_start, window_end, ran_eval, is_test, mappers) values (%s, %s, %s, %s, %s) returning id",
        (window_start, window_end, ran_eval, is_test, json.dumps(mapper_specs))).fetchone()[0]


def finish_job(conn, job_id, status, n_bars=None, duration_s=None, error=None):
    conn.execute("update jobs set status=%s, finished_at=now(), n_bars=%s, duration_s=%s, error=%s where id=%s",
                 (status, json.dumps(n_bars) if n_bars is not None else None, duration_s, error, job_id))


def insert_prices(conn, bars_by_symbol, job_id):
    rows = [(s, b["ts"], b["open"], b["high"], b["low"], b["close"], b["volume"], job_id)
            for s, bars in bars_by_symbol.items() for b in bars]
    with conn.cursor() as cur:
        cur.executemany("insert into prices (ticker, ts, open, high, low, close, volume, job_id) "
                        "values (%s,%s,%s,%s,%s,%s,%s,%s) on conflict (ticker, ts) do nothing", rows)
    return len(rows)


def insert_steps(conn, rows):
    """rows: (run, step, ts, lr, lr_mult, train_loss, job_id)"""
    with conn.cursor() as cur:
        cur.executemany("insert into steps (run, step, ts, lr, lr_mult, train_loss, job_id) values (%s,%s,%s,%s,%s,%s,%s) "
                        "on conflict (run, step) do update set ts=excluded.ts, lr=excluded.lr, lr_mult=excluded.lr_mult, "
                        "train_loss=excluded.train_loss, job_id=excluded.job_id", rows)


def insert_evals(conn, rows):
    """rows: (run, step, val_loss, job_id)"""
    with conn.cursor() as cur:
        cur.executemany("insert into evals (run, step, val_loss, job_id) values (%s,%s,%s,%s) "
                        "on conflict (run, step) do update set val_loss=excluded.val_loss, job_id=excluded.job_id", rows)


def delete_job(conn, job_id):
    for t in ("steps", "evals", "prices"):
        conn.execute(f"delete from {t} where job_id = %s", (job_id,))
    conn.execute("delete from jobs where id = %s", (job_id,))
