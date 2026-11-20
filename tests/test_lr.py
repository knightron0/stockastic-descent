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
    assert list(np.bincount(L.bars_to_steps(60))) == [4] * 60
    assert list(np.bincount(L.bars_to_steps(30))) == [8] * 30
    idx = L.bars_to_steps(57)                # missing minutes still cover all 240 steps
    assert len(idx) == 240 and idx[0] == 0 and idx[-1] == 56
    with pytest.raises(ValueError):
        L.bars_to_steps(0)
