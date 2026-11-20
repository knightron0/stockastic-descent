"""Contract tests run against EVERY registered mapper, so a new formula is checked automatically.
Plus behaviour tests for the mapper currently used in production."""
import numpy as np
import pytest
from stocklr import mappers as M
from stocklr.config import MAPPER_SPECS, RUNS

rng = np.random.default_rng(0)
STRESS = {
    "flat": np.full(800, 123.4),
    "10x rally": np.linspace(100, 1000, 5000),
    "-99% crash": np.linspace(100, 1, 5000),
    "-80% gap": np.r_[np.full(1000, 100.0), np.full(1000, 20.0)],
    "wild vol": 100 * np.exp(np.cumsum(rng.normal(0, 0.05, 5000))),
    "one bar": np.array([50.0]),
}


def bars(p):
    p = np.asarray(p, dtype=float)
    return M.Bars(ts=np.arange(len(p)), close=p, volume=np.ones(len(p)))


ALL = sorted(M.MAPPERS)


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("case", sorted(STRESS))
def test_contract_bounded_finite_right_length(name, case):
    m = M.build({"name": name})
    out = M.safe_multipliers(m, bars(STRESS[case]))
    lo, hi = getattr(m, "bounds", (M.MIN_MULT, M.MAX_MULT))
    assert out.shape == (len(STRESS[case]),)
    assert np.all(np.isfinite(out))
    assert np.all((out >= lo) & (out <= hi))


@pytest.mark.parametrize("name", ALL)
def test_contract_causal(name):
    p = 100 + np.cumsum(rng.normal(0, 0.1, 800))
    m = M.build({"name": name})
    assert np.allclose(m(bars(p[:500])), m(bars(p))[:500]), f"{name} looks at future bars"


@pytest.mark.parametrize("name", ALL)
def test_contract_flat_price_is_neutral(name):
    m = M.build({"name": name})
    assert np.allclose(M.safe_multipliers(m, bars(np.full(300, 42.0))), getattr(m, "neutral", 1.0))


def test_config_is_valid():
    for r in RUNS:
        M.build(MAPPER_SPECS[r])


def test_safe_multipliers_catches_bad_mappers():
    class Broken:
        name = "broken"
        def __init__(self, out): self.out = out
        def __call__(self, b): return self.out(len(b))
    b = bars(np.full(10, 1.0))
    with pytest.raises(ValueError):
        M.safe_multipliers(Broken(lambda n: np.ones(n - 1)), b)        # wrong length
    with pytest.raises(ValueError):
        M.safe_multipliers(Broken(lambda n: np.full(n, np.nan)), b)    # NaN
    assert M.safe_multipliers(Broken(lambda n: np.full(n, 50.0)), b).max() == M.MAX_MULT   # clipped
    with pytest.raises(ValueError):
        M.safe_multipliers(M.build({"name": "constant"}), bars([100.0, 0.0]))  # bad prices


def test_unknown_mapper_is_a_clear_error():
    with pytest.raises(KeyError, match="registered"):
        M.build({"name": "nope"})


# ---- the production mapper (detrended) specifically

def test_detrended_recovers_after_permanent_jump():
    """The bug with price_ratio: a level change pinned the LR forever. Detrended must recover."""
    p = np.r_[np.full(1000, 100.0), np.full(3000, 150.0)]
    m = M.build({"name": "detrended"})(bars(p))
    assert m[1010] > 1.5
    assert abs(m[-1] - 1.0) < 0.02


def test_price_ratio_drifts_which_is_why_its_not_used_live():
    p = np.r_[np.full(1000, 100.0), np.full(3000, 150.0)]
    m = M.safe_multipliers(M.build({"name": "price_ratio"}), bars(p))
    assert m[-1] > 7                                       # (150/100)^5 = 7.6: unbounded, schedule must clip


def test_every_run_has_a_price_source_entry():
    from stocklr.config import PRICE_SOURCE
    assert set(PRICE_SOURCE) == set(RUNS)



# ---- ema_sigmoid (the live mapper)

def _ema_sigmoid_reference(p, h, w, B):
    """Straight from the spec: closed-form weighted sums, no recursion, no shift."""
    x = np.log(np.asarray(p, dtype=float)); a = 0.5 ** (1 / (h * B)); out = []
    for t in range(len(x)):
        wts = a ** (t - np.arange(t + 1))
        mu = (wts * x[: t + 1]).sum() / wts.sum()
        var = max((wts * x[: t + 1] ** 2).sum() / wts.sum() - mu ** 2, 0.0)
        sd = np.sqrt(var)
        out.append(0.5 if sd <= 1e-9 else 1 / (1 + np.exp(-(x[t] - mu) / (w * sd))))
    return np.array(out)


@pytest.mark.parametrize("h,w,B", [(5.0, 1.0, 390), (0.05, 1.0, 390), (1.0, 2.0, 78)])
def test_ema_sigmoid_matches_spec_formula(h, w, B):
    p = 150 * np.exp(np.cumsum(rng.normal(0, 0.002, 600)))
    got = M.build({"name": "ema_sigmoid", "half_life_days": h, "width": w, "bars_per_session": B})(bars(p))
    assert np.allclose(got, _ema_sigmoid_reference(p, h, w, B), atol=1e-6)


def test_ema_sigmoid_decay_constant():
    m = M.build({"name": "ema_sigmoid", "half_life_days": 5.0, "bars_per_session": 390})
    a = 0.5 ** (1 / (m.h * m.B))
    assert a ** (5 * 390) == pytest.approx(0.5)            # weight halves after 5 sessions of 1-minute bars


def test_ema_sigmoid_direction_and_width():
    p = np.r_[100 * np.exp(rng.normal(0, 0.001, 2000)), 103.0]   # noisy flat, then a +3% bar
    s1 = M.build({"name": "ema_sigmoid", "width": 1.0})(bars(p))
    s2 = M.build({"name": "ema_sigmoid", "width": 4.0})(bars(p))
    assert s1[-1] > 0.95                                     # far above the EMA -> near 1
    assert 0.5 < s2[-1] < s1[-1]                             # wider -> gentler
    q = np.r_[p[:-1], 97.0]
    assert M.build({"name": "ema_sigmoid"})(bars(q))[-1] < 0.05


def test_ema_sigmoid_rejects_bad_params():
    for bad in ({"half_life_days": 0}, {"width": -1}, {"bars_per_session": 0}):
        with pytest.raises(ValueError):
            M.build({"name": "ema_sigmoid", **bad})
