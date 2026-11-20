from stocklr.train import eval_points


def test_mid_and_end_eval_each_job():
    assert eval_points(0, 240, 120) == [120, 240]
    assert eval_points(240, 240, 120) == [360, 480]


def test_no_interval_is_end_only():
    assert eval_points(480, 240, None) == [720]


def test_end_not_duplicated_and_partial_job():
    assert eval_points(0, 240, 240) == [240]
    assert eval_points(9360, 15, 120) == [9375]          # last job capped at TOTAL_STEPS
    assert eval_points(9240, 135, 120) == [9360, 9375]


def test_empty_job():
    assert eval_points(9375, 0, 120) == []
