from __future__ import annotations

import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tda_risk.config import RiskConfig
from tda_risk.features import standardize_window
from tda_risk.methods import (
    EuclideanKNN,
    GapHS,
    RegimeHS,
    _scenario_losses_from_positions,
    _state_knn_inputs,
    _window_euclidean_top_positions,
    clear_method_caches,
    eligible_candidate_positions,
)
from tda_risk.topology import _standardize_topology_window
import tda_risk.topology as topology_module


ROOT = Path(__file__).resolve().parents[1]
STATE_COLUMNS = ("state_a", "state_b")


def synthetic_panel(n: int = 40) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    values = np.arange(n, dtype=float)
    frame = pd.DataFrame(
        {
            "portfolio_return": (values + 1.0) / 1000.0,
            "portfolio_loss": -(values + 1.0) / 1000.0,
            "state_a": values,
            "state_b": values * 2.0 + 1.0,
            "rv20_p": values,
        },
        index=index,
    )
    return frame


class TargetAlignmentTests(unittest.TestCase):
    def test_scenario_positions_map_feature_date_to_next_day_loss(self) -> None:
        frame = synthetic_panel()
        positions = np.array([2, 5, 9])
        np.testing.assert_array_equal(
            _scenario_losses_from_positions(frame, positions),
            frame["portfolio_loss"].to_numpy()[positions + 1],
        )

    def test_regime_hs_uses_following_day_targets(self) -> None:
        frame = synthetic_panel(8)
        frame["rv20_p"] = [0.0, 1.0, 2.0, 3.0, 4.0, 100.0, 5.0, 6.0]
        position = 5
        forecast = RegimeHS(regime_bins=2).forecast(frame, position, RiskConfig())
        # Historical states 3 and 4 are in the current high-volatility bin;
        # their targets are losses at dates 4 and 5, not same-day losses.
        expected = frame["portfolio_loss"].iloc[[4, 5]].to_numpy()
        self.assertEqual(forecast.scenario_count, expected.size)
        self.assertEqual(forecast.var, float(np.quantile(expected, 0.99)))


class LookAheadAndScalingTests(unittest.TestCase):
    def test_state_candidates_and_query_share_one_vintage_scaler(self) -> None:
        frame = synthetic_panel()
        current, history, positions = _state_knn_inputs(frame, 10, STATE_COLUMNS)
        raw = frame.loc[:, list(STATE_COLUMNS)].to_numpy()
        expected_mean = raw[positions].mean(axis=0)
        expected_scale = raw[positions].std(axis=0, ddof=0)
        np.testing.assert_allclose(current, (raw[10] - expected_mean) / expected_scale)
        np.testing.assert_allclose(history, (raw[positions] - expected_mean) / expected_scale)

    def test_state_forecast_does_not_change_when_future_rows_change(self) -> None:
        frame = synthetic_panel()
        position = 20
        method = EuclideanKNN(k=3, state_columns=STATE_COLUMNS, max_k=3)
        clear_method_caches()
        before = method.forecast(frame, position, RiskConfig())
        changed = frame.copy()
        changed.iloc[position + 1 :, changed.columns.get_indexer(list(STATE_COLUMNS))] += 10_000.0
        changed.iloc[position + 1 :, changed.columns.get_loc("portfolio_loss")] = 99.0
        clear_method_caches()
        after = method.forecast(changed, position, RiskConfig())
        self.assertEqual(before.scenario_count, after.scenario_count)
        self.assertEqual(before.var, after.var)
        self.assertEqual(before.es, after.es)

    def test_window_candidates_do_not_use_future_rows(self) -> None:
        frame = synthetic_panel()
        position = 20
        clear_method_caches()
        before = _window_euclidean_top_positions(frame, position, window_length=5, max_k=4)
        changed = frame.copy()
        changed.iloc[position + 1 :, changed.columns.get_loc("portfolio_return")] = 99.0
        clear_method_caches()
        after = _window_euclidean_top_positions(changed, position, window_length=5, max_k=4)
        np.testing.assert_array_equal(before, after)

    def test_regime_forecast_does_not_use_future_rows(self) -> None:
        frame = synthetic_panel(12)
        frame["rv20_p"] = np.arange(12, dtype=float)
        before = RegimeHS(regime_bins=2).forecast(frame, 8, RiskConfig())
        changed = frame.copy()
        changed.iloc[9:, changed.columns.get_loc("rv20_p")] = -10_000.0
        changed.iloc[9:, changed.columns.get_loc("portfolio_loss")] = 99.0
        after = RegimeHS(regime_bins=2).forecast(changed, 8, RiskConfig())
        self.assertEqual(before.var, after.var)
        self.assertEqual(before.es, after.es)

    def test_topology_retrieval_does_not_use_future_rows(self) -> None:
        frame = synthetic_panel(240)
        topology_features = np.column_stack(
            [frame["portfolio_return"].to_numpy(), frame["portfolio_return"].to_numpy() ** 2]
        )
        config = topology_module.TopologyConfig(
            tau_candidates=(1,), embedding_dimension_candidates=(2,), n_jobs=1
        )
        with patch.object(topology_module, "_select_embedding_parameters", return_value=(1, 2)), patch.object(
            topology_module,
            "_get_topology_feature_matrix",
            return_value=topology_features,
        ):
            topology_module.clear_topology_caches()
            before = topology_module._topology_top_positions(
                frame,
                200,
                STATE_COLUMNS,
                5,
                "portfolio_only",
                "landscape1",
                0.0,
                5,
                config,
                "2020-12-31",
                use_null=False,
            )
            changed = frame.copy()
            changed.iloc[201:, changed.columns.get_loc("portfolio_return")] = 99.0
            changed.iloc[201:, changed.columns.get_loc("portfolio_loss")] = 99.0
            topology_module.clear_topology_caches()
            with patch.object(
                topology_module,
                "_get_topology_feature_matrix",
                return_value=np.column_stack(
                    [changed["portfolio_return"].to_numpy(), changed["portfolio_return"].to_numpy() ** 2]
                ),
            ):
                after = topology_module._topology_top_positions(
                    changed,
                    200,
                    STATE_COLUMNS,
                    5,
                    "portfolio_only",
                    "landscape1",
                    0.0,
                    5,
                    config,
                    "2020-12-31",
                    use_null=False,
                )
        np.testing.assert_array_equal(before, after)


class ComparatorAndBoundaryTests(unittest.TestCase):
    def test_shared_window_standardization_matches_topology_convention(self) -> None:
        window = np.array([1.0, 2.0, 4.0, 8.0])
        expected = (window - window.mean()) / window.std(ddof=0)
        np.testing.assert_allclose(standardize_window(window), expected)
        np.testing.assert_allclose(_standardize_topology_window(window), expected)
        np.testing.assert_allclose(standardize_window(np.ones(4)), np.zeros(4))

    def test_exclusion_zone_boundaries(self) -> None:
        self.assertEqual(eligible_candidate_positions(200, 300, 0)[-1], 199)
        self.assertEqual(eligible_candidate_positions(200, 300, 125)[-1], 74)
        for candidate in (76, 75):
            self.assertNotIn(candidate, eligible_candidate_positions(200, 300, 125))

    def test_gap_hs_ends_before_exclusion_gap(self) -> None:
        frame = synthetic_panel(300)
        forecast = GapHS(window=3, exclusion_radius=125).forecast(frame, 200, RiskConfig())
        expected = frame["portfolio_loss"].iloc[72:75].to_numpy()
        self.assertEqual(forecast.scenario_count, 3)
        np.testing.assert_allclose(forecast.es, expected.max())


class RegistryAndDataTests(unittest.TestCase):
    def test_locked_registry_is_explicit_and_unique(self) -> None:
        path = ROOT / "revision_config" / "finalists.csv"
        registry = pd.read_csv(path)
        self.assertEqual(len(registry), 13)
        self.assertEqual(registry["configuration_id"].nunique(), 13)
        self.assertEqual(registry["method"].nunique(), 13)
        topology = registry.loc[registry["method"].eq("topology")]
        self.assertEqual(len(topology), 1)
        self.assertEqual(topology.iloc[0]["configuration_id"], "topology_l250_k100")

    def test_frozen_panel_hash_and_simple_return(self) -> None:
        path = ROOT / "data" / "panel_published.csv.gz"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "a5f7bde9b44696dee70b08da59e21afd342b9ef8a7182db034fff8884659e50e",
        )
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        self.assertEqual(len(frame), 5978)
        self.assertEqual(frame.index.min().date().isoformat(), "2002-07-30")
        self.assertEqual(frame.index.max().date().isoformat(), "2026-05-01")
        first_valid = frame[["SPY_adj_close", "SPY_return"]].dropna().iloc[1]
        previous = frame["SPY_adj_close"].iloc[1]
        self.assertAlmostEqual(first_valid["SPY_return"], first_valid["SPY_adj_close"] / previous - 1.0)


if __name__ == "__main__":
    unittest.main()
