import numpy as np

from tda_risk.scoring import (
    acerbi_szekely_z2,
    conditional_coverage,
    duration_geometric_test,
    fz0_score,
    kupiec_uc,
)


def test_fz0_score_is_finite_for_valid_positive_loss_forecasts():
    score = fz0_score(
        np.array([0.01, 0.03, 0.02]),
        np.array([0.02, 0.02, 0.02]),
        np.array([0.04, 0.04, 0.04]),
        0.975,
    )
    assert np.isfinite(score).all()
    assert score[1] > score[0]


def test_fz0_score_marks_nonpositive_es_invalid():
    score = fz0_score(np.array([0.02]), np.array([0.01]), np.array([0.0]), 0.975)
    assert np.isnan(score[0])


def test_coverage_backtests_return_finite_values():
    exceedances = np.array([False] * 98 + [True, False])
    uc_stat, uc_p = kupiec_uc(exceedances, 0.99)
    cc_stat, cc_p = conditional_coverage(exceedances, 0.99)
    assert np.isfinite([uc_stat, uc_p, cc_stat, cc_p]).all()


def test_acerbi_szekely_z2_is_zero_for_calibrated_moment():
    realized = np.array([0.01, 0.02, 0.40, 0.03, 0.50, 0.02])
    var = np.full(realized.size, 0.30)
    es = np.full(realized.size, 0.60)
    statistic, pvalue, n = acerbi_szekely_z2(realized, var, es, 0.975)
    assert n == realized.size
    assert np.isfinite([statistic, pvalue]).all()
    assert 0.0 <= pvalue <= 1.0


def test_duration_geometric_test_has_finite_reference_values():
    statistic, pvalue = duration_geometric_test(
        np.array([False] * 10 + [True] + [False] * 10 + [True]), 0.90
    )
    assert np.isfinite([statistic, pvalue]).all()
    assert 0.0 <= pvalue <= 1.0
