from datetime import datetime, timezone
from stocklr.market import ET, fetch_bars, job_window


def et(*a):
    return datetime(*a, tzinfo=ET)


def test_windows_through_a_trading_day():
    # Thu 2026-11-19
    w = job_window(et(2026, 11, 19, 10, 17))
    assert w[0].astimezone(ET) == et(2026, 11, 19, 9, 30) and w[1].astimezone(ET) == et(2026, 11, 19, 10, 0)
    w = job_window(et(2026, 11, 19, 11, 17))
    assert w[0].astimezone(ET) == et(2026, 11, 19, 10, 0) and w[1].astimezone(ET) == et(2026, 11, 19, 11, 0)
    w = job_window(et(2026, 11, 19, 16, 17))
    assert w[0].astimezone(ET) == et(2026, 11, 19, 15, 0) and w[1].astimezone(ET) == et(2026, 11, 19, 16, 0)


def test_no_window_outside_hours():
    assert job_window(et(2026, 11, 19, 9, 17)) is None     # before the open
    assert job_window(et(2026, 11, 19, 17, 17)) is None    # after the close
    assert job_window(et(2026, 11, 21, 12, 17)) is None    # Saturday
    assert job_window(et(2026, 11, 22, 12, 17)) is None    # Sunday


def test_cron_hours_map_to_seven_windows():
    wins = [job_window(et(2026, 11, 19, h, 17)) for h in range(10, 17)]
    assert all(wins)
    total = sum((e - s).total_seconds() for s, e in wins) / 60
    assert total == 390                                     # exactly one full session, no gaps or overlap
    assert all(wins[i][1] == wins[i + 1][0] for i in range(len(wins) - 1))


def test_dst_is_handled():
    # Mon 2026-03-09 is after US DST starts (Mar 8): 9:30 ET = 13:30 UTC
    s, e = job_window(et(2026, 3, 9, 10, 17))
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
