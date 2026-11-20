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
    out = M.safe_multipliers(M.build({"name": name}), bars(STRESS[case]))
    assert out.shape == (len(STRESS[case]),)
    assert np.all(np.isfinite(out))
    assert np.all((out >= M.MIN_MULT) & (out <= M.MAX_MULT))


@pytest.mark.parametrize("name", ALL)
def test_contract_causal(name):
    p = 100 + np.cumsum(rng.normal(0, 0.1, 800))
    m = M.build({"name": name})
    assert np.allclose(m(bars(p[:500])), m(bars(p))[:500]), f"{name} looks at future bars"


@pytest.mark.parametrize("name", ALL)
def test_contract_flat_price_is_neutral(name):
    assert np.allclose(M.safe_multipliers(M.build({"name": name}), bars(np.full(300, 42.0))), 1.0)


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
    assert m[-1] == M.MAX_MULT


def test_every_run_has_a_price_source_entry():
    from stocklr.config import PRICE_SOURCE
    assert set(PRICE_SOURCE) == set(RUNS)
