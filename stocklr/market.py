"""Alpaca 1-minute bars (SIP feed) and the trading-hour window each job covers."""
import json
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
OPEN, CLOSE = time(9, 30), time(16, 0)
BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


def job_window(now: datetime):
    """The hour a job running at `now` should train on: [X-1:00, X:00) ET, clipped to 9:30-16:00.
    Returns (start, end) as UTC datetimes, or None outside market hours / on weekends."""
    et = now.astimezone(ET)
    if et.weekday() >= 5:
        return None
    end = et.replace(minute=0, second=0, microsecond=0)
    day_open = end.replace(hour=OPEN.hour, minute=OPEN.minute)
    day_close = end.replace(hour=CLOSE.hour, minute=0)
    if end <= day_open or end > day_close:
        return None
    start = max(end - timedelta(hours=1), day_open)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def is_regular_session(ts: datetime) -> bool:
    et = ts.astimezone(ET)
    return et.weekday() < 5 and OPEN <= et.time() < CLOSE


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


def fetch_bars(symbols, start, end, key_id, secret, feed="sip", get=_http_get):
    """Regular-session 1-minute bars with start <= ts < end, oldest first, for each symbol.
    Follows Alpaca's next_page_token. Returns {symbol: [{ts, open, high, low, close, volume}, ...]}."""
    headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
    params = {"symbols": ",".join(symbols), "timeframe": "1Min", "feed": feed, "adjustment": "raw",
              "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
