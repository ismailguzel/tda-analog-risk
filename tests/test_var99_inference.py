import numpy as np
import pandas as pd

from tda_risk.inference_var99 import (
    paired_pinball_comparison,
    pinball_loss,
    var99_model_confidence_set,
)


def test_identical_forecasts_have_zero_pinball_difference():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    forecast = np.array([0.0, 1.0, 2.0, 3.0])
    result = paired_pinball_comparison(
        y, forecast, {"same": forecast}, bootstrap_reps=20, block_length=2
    )
    assert result.loc[0, "mean_pinball_difference_topology_minus_comparator"] == 0.0
    assert result.loc[0, "dm_pvalue"] == 1.0


def test_better_topology_forecast_has_negative_loss_difference():
    y = np.linspace(0.0, 1.0, 50)
    topology = y.copy()
    comparator = y + 0.25
    result = paired_pinball_comparison(
        y, topology, {"worse": comparator}, bootstrap_reps=20, block_length=5
    )
    assert result.loc[0, "mean_pinball_difference_topology_minus_comparator"] < 0.0


def test_var99_mcs_preserves_input_dimensions_and_labels():
    losses = pd.DataFrame(
        {
            "topology": pinball_loss(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), 0.99),
            "fhs": np.array([0.1, 0.2, 0.3]),
            "garch": np.array([0.2, 0.1, 0.2]),
        }
    )
    result = var99_model_confidence_set(losses, bootstrap_reps=10, block_length=2)
    assert result.shape[0] == 3
    assert result["model"].tolist() == ["topology", "fhs", "garch"]
    assert set(result["n_dates"]) == {3}
    assert set(result["n_models"]) == {3}
