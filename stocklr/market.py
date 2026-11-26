"""Alpaca 1-minute bars (SIP feed) and the 15-minute trading window each job covers."""
import json
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
OPEN, CLOSE = time(9, 30), time(16, 0)
JOB_MINUTES = 15               # 9:30 and 16:00 fall on quarter hours, so every window is a full 15 bars
DATA_DELAY = timedelta(minutes=16)   # SIP bars are 15 min delayed: a window is trained 16+ min after it ends
BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"

# NYSE 2026 calendar, hardcoded. Holidays: no jobs at all. Early closes: the session ends at 13:00 ET, so the
# 13:00-16:00 windows (which would otherwise train on after-hours bars) don't exist.
HOLIDAYS = {date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
            date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25)}
EARLY_CLOSES = {date(2026, 11, 27): time(13, 0), date(2026, 12, 24): time(13, 0)}


def session_close(d: date):
    """Close time (ET) of the regular session on date d, or None if the market is shut (weekend/holiday)."""
    if d.weekday() >= 5 or d in HOLIDAYS:
        return None
    return EARLY_CLOSES.get(d, CLOSE)


def job_window(now: datetime):
    """The quarter hour a job running at `now` should train on: the latest [:00|:15|:30|:45, +15 min) ET
    window within 9:30-16:00 that ended at least DATA_DELAY ago (so its delayed bars exist).
    Returns (start, end) as UTC datetimes, or None outside market hours / on weekends and holidays."""
    et = (now - DATA_DELAY).astimezone(ET)
    if (close := session_close(et.date())) is None:
        return None
    end = et.replace(minute=et.minute // JOB_MINUTES * JOB_MINUTES, second=0, microsecond=0)
    day_open = end.replace(hour=OPEN.hour, minute=OPEN.minute)
    day_close = end.replace(hour=close.hour, minute=close.minute)
    if end <= day_open or end > day_close:
        return None
    start = max(end - timedelta(minutes=JOB_MINUTES), day_open)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)



def job_windows_for_day(d):
    """Every window the cron trains on a given date (a datetime.date), in order: 26 x 15 min on a full trading day,
    14 on an early close, none on a holiday."""
    out, t = [], datetime(d.year, d.month, d.day, OPEN.hour, OPEN.minute, tzinfo=ET)
    while (t := t + timedelta(minutes=JOB_MINUTES)).time() <= CLOSE:
        if (w := job_window(t + DATA_DELAY)):
            out.append(w)
    return out


def is_regular_session(ts: datetime) -> bool:
    et = ts.astimezone(ET)
    close = session_close(et.date())
    return close is not None and OPEN <= et.time() < close


def _http_get(url, headers, retries=4):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                _time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(f"Alpaca HTTP {e.code}: {e.read()[:300]!r}") from e
        except urllib.error.URLError:
            if attempt < retries - 1:
                _time.sleep(2 ** attempt * 2)
                continue
            raise


def _utc_z(dt):
    """Alpaca wants UTC 'Z' timestamps; convert first so a window given in ET (or any offset) isn't shifted."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_bars(symbols, start, end, key_id, secret, feed="sip", get=_http_get):
    """Regular-session 1-minute bars with start <= ts < end, oldest first, for each symbol.
    Follows Alpaca's next_page_token. Returns {symbol: [{ts, open, high, low, close, volume}, ...]}."""
    headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
    params = {"symbols": ",".join(symbols), "timeframe": "1Min", "feed": feed, "adjustment": "raw",
              "start": _utc_z(start), "end": _utc_z(end),
              "limit": 10000, "sort": "asc"}
    out = {s: [] for s in symbols}
    token = None
    for _ in range(100):
        q = dict(params, **({"page_token": token} if token else {}))
        body = get(f"{BARS_URL}?{urllib.parse.urlencode(q)}", headers)
        for sym, bars in (body.get("bars") or {}).items():
            for b in bars:
                ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
                if start <= ts < end and is_regular_session(ts):
                    out.setdefault(sym, []).append({"ts": ts, "open": b["o"], "high": b["h"], "low": b["l"],
                                                    "close": b["c"], "volume": b["v"]})
        token = body.get("next_page_token")
        if not token:
            break
    else:
        raise RuntimeError("Alpaca pagination did not terminate")
    for sym in out:
        out[sym].sort(key=lambda b: b["ts"])
    return out
