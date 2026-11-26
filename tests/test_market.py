from datetime import date, datetime, timezone
from stocklr.market import ET, fetch_bars, is_regular_session, job_window, job_windows_for_day


def et(*a):
    return datetime(*a, tzinfo=ET)


def test_windows_through_a_trading_day():
    # Thu 2026-11-19; the cron fires at :02/:17/:32/:47 and trains the window that ended 17 min earlier
    def win(h, m):
        s, e = job_window(et(2026, 11, 19, h, m))
        return s.astimezone(ET).strftime("%H:%M"), e.astimezone(ET).strftime("%H:%M")
    assert win(10, 2) == ("09:30", "09:45")
    assert win(10, 17) == ("09:45", "10:00")
    assert win(11, 47) == ("11:15", "11:30")
    assert win(16, 17) == ("15:45", "16:00")
    assert win(10, 15) == ("09:30", "09:45")        # 10:00-10:15's bars aren't all out until 10:16


def test_no_window_outside_hours():
    for h, m in [(9, 47), (10, 0), (16, 32), (16, 47), (17, 17)]:
        assert job_window(et(2026, 11, 19, h, m)) is None, (h, m)   # first window not out yet / after close
    assert job_window(et(2026, 11, 21, 12, 17)) is None    # Saturday
    assert job_window(et(2026, 11, 22, 12, 17)) is None    # Sunday


def test_cron_fires_map_to_26_full_windows():
    fires = [et(2026, 11, 19, h, m) for h in range(10, 17) for m in (2, 17, 32, 47)]
    wins = [w for t in fires if (w := job_window(t))]
    assert len(wins) == 26
    assert all((e - s).total_seconds() == 15 * 60 for s, e in wins)   # every window is a full 15 bars
    assert all(wins[i][1] == wins[i + 1][0] for i in range(len(wins) - 1))
    assert wins == job_windows_for_day(date(2026, 11, 19))
    assert job_windows_for_day(date(2026, 11, 21)) == []


def test_holiday_has_no_windows():
    # Thanksgiving, Thu 2026-11-26: every cron fire finds nothing to do
    assert job_windows_for_day(date(2026, 11, 26)) == []
    assert all(job_window(et(2026, 11, 26, h, m)) is None for h in range(10, 17) for m in (2, 17, 32, 47))
    assert not is_regular_session(et(2026, 11, 26, 12, 0))


def test_early_close_stops_at_1pm():
    # Fri 2026-11-27 closes at 13:00 ET: 14 windows, the last is 12:45-13:00, after-hours bars are dropped
    wins = job_windows_for_day(date(2026, 11, 27))
    assert len(wins) == 14
    assert wins[-1][1].astimezone(ET).strftime("%H:%M") == "13:00"
    assert job_window(et(2026, 11, 27, 13, 17)) is not None      # trains 12:45-13:00
    assert job_window(et(2026, 11, 27, 13, 32)) is None
    assert is_regular_session(et(2026, 11, 27, 12, 59))
    assert not is_regular_session(et(2026, 11, 27, 13, 0))
    assert len(job_windows_for_day(date(2026, 11, 30))) == 26     # Monday is a full day again


def test_dst_is_handled():
    # Mon 2026-03-09 is after US DST starts (Mar 8): 9:30 ET = 13:30 UTC
    s, e = job_window(et(2026, 3, 9, 10, 2))
    assert s == datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc)


def _bar(t, c):
    return {"t": t, "o": c, "h": c, "l": c, "c": c, "v": 100}


def test_fetch_bars_paginates_and_filters():
    start = et(2026, 11, 19, 15, 0).astimezone(timezone.utc)
    end = et(2026, 11, 19, 16, 0).astimezone(timezone.utc)
    pages = [
        {"bars": {"NVDA": [_bar("2026-11-19T19:59:00Z", 1.0),     # 14:59 ET: before window
                           _bar("2026-11-19T20:00:00Z", 2.0)]},   # 15:00 ET: in
         "next_page_token": "p2"},
        {"bars": {"NVDA": [_bar("2026-11-19T20:59:00Z", 3.0),     # 15:59 ET: in
                           _bar("2026-11-19T21:00:00Z", 4.0)],    # 16:00 ET: after-hours, out
                  "ANT": [_bar("2026-11-19T20:30:00Z", 5.0)]},
         "next_page_token": None},
    ]
    calls = []
    def fake_get(url, headers):
        calls.append(url)
        assert headers["APCA-API-KEY-ID"] == "k"
        return pages[len(calls) - 1]
    out = fetch_bars(["NVDA", "ANT", "SPCX"], start, end, "k", "s", get=fake_get)
    assert len(calls) == 2 and "page_token=p2" in calls[1] and "feed=sip" in calls[0]
    assert [b["close"] for b in out["NVDA"]] == [2.0, 3.0]
    assert [b["close"] for b in out["ANT"]] == [5.0]
    assert out["SPCX"] == []


def test_fetch_bars_sends_utc_for_non_utc_window():
    calls = []
    def fake_get(url, headers):
        calls.append(url)
        return {"bars": {"NVDA": [_bar("2026-11-19T19:30:00Z", 1.0)]}, "next_page_token": None}   # 14:30 ET
    out = fetch_bars(["NVDA"], et(2026, 11, 19, 14, 0), et(2026, 11, 19, 15, 0), "k", "s", get=fake_get)
    assert "start=2026-11-19T19%3A00%3A00Z" in calls[0] and "end=2026-11-19T20%3A00%3A00Z" in calls[0]
    assert [b["close"] for b in out["NVDA"]] == [1.0]
