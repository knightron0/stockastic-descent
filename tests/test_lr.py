import numpy as np
import pytest
from stocklr import lr as L


def test_wsd_shape():
    assert L.wsd(1) == pytest.approx(1 / L.WARMUP_STEPS)
    assert L.wsd(L.WARMUP_STEPS) == 1.0
    assert L.wsd(5000) == 1.0
    assert L.wsd(L.DECAY_START) == 1.0
    assert L.wsd(L.TOTAL_STEPS) == 0.0
    mid = (L.DECAY_START + L.TOTAL_STEPS) // 2
    assert 0.45 < L.wsd(mid) < 0.55
    assert L.wsd(L.TOTAL_STEPS + 100) == 0.0          # never negative


def test_bars_to_steps():
    assert L.STEPS_PER_JOB == 60
    assert list(np.bincount(L.bars_to_steps(15))) == [4] * 15
    assert list(np.bincount(L.bars_to_steps(60, 240))) == [4] * 60
    idx = L.bars_to_steps(13)                # missing minutes still cover all 60 steps
    assert len(idx) == 60 and idx[0] == 0 and idx[-1] == 12
    with pytest.raises(ValueError):
        L.bars_to_steps(0)



# ---- Envelope (live schedule)

def _env(**kw):
    return L.build_schedule({"name": "envelope", "eta_max": 3e-3, "floor": 0.1, "warmup": 0.02, "max_rise": 0.5, **kw})


def test_envelope_midpoint_and_bounds():
    e, st = _env(), {}
    for step in range(1, 400):
        lr, _ = e(step, 0.5, st)
    assert lr == pytest.approx(3e-3 * (0.1 + 0.9 * 0.5))            # eta_max (f + (1-f)/2) at s = 1/2
    assert e(5000, 1.0, {})[0] == pytest.approx(3e-3)                # s -> 1: eta_max
    assert e(5000, 0.0, {})[0] == pytest.approx(3e-4)                # s -> 0: floor * eta_max


def test_envelope_warmup_is_2_percent_linear():
    e = _env()
    N = L.TOTAL_STEPS
    assert e(1, 1.0, {})[0] == pytest.approx(3e-3 / (0.02 * N))
    assert e(int(0.01 * N), 1.0, {})[0] == pytest.approx(3e-3 * int(0.01 * N) / (0.02 * N))
    assert e(int(0.02 * N) + 1, 1.0, {})[0] == pytest.approx(3e-3)   # full after 2%, and no cooldown:
    assert e(N, 1.0, {})[0] == pytest.approx(3e-3)


def test_envelope_caps_rises_but_not_falls():
    e, st = _env(), {}
    N = L.TOTAL_STEPS
    _, s0 = e(1000, 0.1, st)
    _, s1 = e(1001, 0.9, st)
    assert s0 == 0.1 and s1 == pytest.approx(0.1 + 0.5 / (0.01 * N))
    for step in range(1002, 1002 + int(0.016 * N)):                 # 0.8 rise takes 1.6% of the run
        _, s = e(step, 0.9, st)
    assert s == pytest.approx(0.9)
    _, s = e(5000, 0.2, st)                                           # drops pass straight through
    assert s == 0.2


def test_envelope_state_carries_across_jobs():
    sig = np.clip(0.5 + 0.4 * np.sin(np.arange(960) / 25), 0, 1)
    one = _env(); st = {}
    a = [one(i + 1, v, st)[0] for i, v in enumerate(sig)]
    two = _env(); st2 = {}; b = []
    for job in range(4):                                             # 4 jobs, state persisted between
        st2 = dict(st2)
        b += [two(job * 240 + i + 1, v, st2)[0] for i, v in enumerate(sig[job * 240:(job + 1) * 240])]
    assert np.allclose(a, b)


def test_wsd_schedule_matches_legacy():
    w = L.build_schedule({"name": "wsd"})
    assert w(5000, 1.3, {})[0] == pytest.approx(L.ETA0 * 1.3)


def test_unknown_schedule():
    with pytest.raises(KeyError):
        L.build_schedule({"name": "nope"})


def test_envelope_without_warmup_starts_at_full_lr():
    e = _env(warmup=0.0)
    assert e(1, 0.5, {})[0] == pytest.approx(3e-3 * (0.1 + 0.9 * 0.5))


def test_constant_schedule():
    c = L.build_schedule({"name": "constant", "peak_lr": 3e-3})
    assert c(1, 1.0, {})[0] == pytest.approx(3e-3)
    assert c(9000, 0.9, {})[0] == pytest.approx(2.7e-3)


def test_live_formula_is_peak_times_price_over_start():
    import numpy as np
    from stocklr import mappers as M
    from stocklr.config import SCHEDULE, TICKER_MAPPER
    p = np.array([100.0, 110.0, 95.0, 100.0])
    m = M.safe_multipliers(M.build(TICKER_MAPPER), M.Bars(np.arange(4), p, np.ones(4)))
    sched = L.build_schedule(SCHEDULE)
    lrs = [sched(i + 94, v, {})[0] for i, v in enumerate(m)]
    assert np.allclose(lrs, 3e-3 * p / p[0])


def test_live_warmup_matches_trial_schedule():
    from stocklr.config import SCHEDULE
    live = L.build_schedule(SCHEDULE)
    trial = L.Alchemy("const", eta0=1.0)
    for step in (1, 2, 60, 61, 93, 94, 4801):
        for mult in (0.0, 0.9, 2.0):
            expected, _ = trial(step, max(1e-6, 3e-3 * mult), {})
            assert live(step, mult, {}) == (expected, mult)


def test_constant_schedule_clips_lr():
    from stocklr.config import SCHEDULE
    c = L.build_schedule(SCHEDULE)
    assert c(94, 0.0, {})[0] == 1e-6            # price -> 0: floor
    assert c(94, 1e-9, {})[0] == 1e-6
    assert c(94, 5.0, {})[0] == pytest.approx(1.5e-2)   # price x5: no ceiling
    assert c(94, 1.1, {})[0] == pytest.approx(3.3e-3)
    capped = L.build_schedule({**SCHEDULE, "max_lr": 6e-3})
    assert capped(94, 5.0, {})[0] == 6e-3


def test_baseline_schedules():
    s = {k: L.build_schedule(v) for k, v in L.BASELINE_SCHEDULES.items()}
    N, W = L.TOTAL_STEPS, 93                              # int(9375 * 0.01)
    lr = lambda k, step: s[k](step, 1.0, {})[0]
    for k in s:                                           # every baseline warms up from 0 over 93 steps
        assert lr(k, 1) == 0.0
        assert lr(k, 2) == pytest.approx(L.ETA0 / W)
        assert lr(k, W + 1) == pytest.approx(L.ETA0)
    assert lr("const", N) == pytest.approx(L.ETA0)
    assert lr("wsd", 7500) == pytest.approx(L.ETA0) and lr("wsd", 7501) == pytest.approx(L.ETA0)
    assert lr("wsd", 7502) < L.ETA0
    assert lr("linear", N) == pytest.approx(L.ETA0 / (N - W))   # their last step: one notch above 0
    lin = [lr("linear", t) for t in range(W + 1, N + 1)]
    assert all(a > b for a, b in zip(lin, lin[1:]))
