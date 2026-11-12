from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from tda_risk.null_controls import (
    eligible_history,
    topology_history_permutation,
    uniform_random_retrieval,
)
from tda_risk.scoring import christoffersen_independence, conditional_coverage, kupiec_uc


def six_candidate_fixture():
    # Candidate dates 0..5 have distinct features and distinct next-day losses.
    features = np.arange(7, dtype=float)[:, None]
    losses = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 10.0])
    return features, losses


def test_history_only_permutation_selects_dates_using_permuted_features():
    features, losses = six_candidate_fixture()
    candidates = eligible_history(6, 7, 0, features, losses)
    assert candidates.tolist() == [0, 1, 2, 3, 4, 5]
    selection = topology_history_permutation(
        features[6],
        candidates,
        features,
        2,
        np.random.default_rng(1),
        seed=77,
        permutation=np.array([5, 4, 3, 2, 1, 0]),
        forecast_position=6,
    )
    # Candidate dates 0 and 1 receive feature rows 5 and 4, respectively.
    assert selection.selected_positions.tolist() == [0, 1]
    assert selection.feature_origins.tolist() == [5, 4]
    np.testing.assert_array_equal(selection.query_feature, features[6])
    # Scenario losses follow selected dates, not feature origins.
    assert losses[selection.selected_loss_positions].tolist() == [200.0, 300.0]
    assert selection.distances is not None


def test_uniform_retrieval_has_no_topology_distance_path():
    features, losses = six_candidate_fixture()
    candidates = eligible_history(6, 7, 0, features, losses)
    selection = uniform_random_retrieval(candidates, 3, np.random.default_rng(8), seed=88)
    assert selection.control == "uniform_random_retrieval"
    assert selection.distances is None
    assert selection.feature_origins.size == 0
    assert selection.selected_positions.size == 3


def test_radius_boundaries_and_no_future_features():
    features, losses = six_candidate_fixture()
    assert eligible_history(6, 7, 0, features, losses).tolist() == [0, 1, 2, 3, 4, 5]
    assert eligible_history(6, 7, 2, features, losses).tolist() == [0, 1, 2, 3]
    changed = features.copy()
    changed[6, 0] = 99999.0
    assert eligible_history(6, 7, 0, changed, losses).tolist() == [0, 1, 2, 3, 4, 5]


def test_exact_seed_reproduces_each_control():
    features, losses = six_candidate_fixture()
    candidates = eligible_history(6, 7, 0, features, losses)
    first = uniform_random_retrieval(candidates, 4, np.random.default_rng(123), seed=123)
    second = uniform_random_retrieval(candidates, 4, np.random.default_rng(123), seed=123)
    np.testing.assert_array_equal(first.selected_positions, second.selected_positions)
    first = topology_history_permutation(
        features[6], candidates, features, 4, np.random.default_rng(123), seed=123
    )
    second = topology_history_permutation(
        features[6], candidates, features, 4, np.random.default_rng(123), seed=123
    )
    np.testing.assert_array_equal(first.selected_positions, second.selected_positions)
    np.testing.assert_array_equal(first.feature_origins, second.feature_origins)


def test_direct_permutation_tie_break_is_seed_reproducible_and_not_date_ordered():
    features = np.ones((9, 1), dtype=float)
    losses = np.arange(1.0, 10.0)
    candidates = eligible_history(8, 9, 0, features, losses)
    first = topology_history_permutation(
        features[8], candidates, features, 3, np.random.default_rng(321), seed=321
    )
    second = topology_history_permutation(
        features[8], candidates, features, 3, np.random.default_rng(321), seed=321
    )
    np.testing.assert_array_equal(first.selected_positions, second.selected_positions)
    assert first.selected_positions.tolist() != [0, 1, 2]


def test_direct_tied_selection_is_uniform_like_uniform_subset():
    features = np.ones((9, 1), dtype=float)
    losses = np.arange(1.0, 10.0)
    candidates = eligible_history(8, 9, 0, features, losses)
    counts = np.zeros(candidates.size, dtype=int)
    uniform_counts = np.zeros(candidates.size, dtype=int)
    for seed in range(2000):
        direct = topology_history_permutation(
            features[8], candidates, features, 3, np.random.default_rng(seed), seed=seed
        )
        uniform = uniform_random_retrieval(candidates, 3, np.random.default_rng(seed), seed=seed)
        counts[direct.selected_positions] += 1
        uniform_counts[uniform.selected_positions] += 1
    assert counts.min() > 0
    assert np.max(np.abs(counts - uniform_counts)) < 180


def test_conditional_coverage_uses_two_degrees_of_freedom():
    exceedances = np.array([False, False, True, False, True, False, False, True, False, False])
    uc_stat, uc_p = kupiec_uc(exceedances, 0.8)
    ind_stat, ind_p = christoffersen_independence(exceedances)
    cc_stat, cc_p = conditional_coverage(exceedances, 0.8)
    assert np.isclose(cc_stat, uc_stat + ind_stat)
    assert np.isclose(uc_p, chi2.sf(uc_stat, 1))
    assert np.isclose(ind_p, chi2.sf(ind_stat, 1))
    assert np.isclose(cc_p, chi2.sf(cc_stat, 2))
