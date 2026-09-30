from pathlib import Path

import numpy as np
import pandas as pd

from tda_risk.config import TopologyConfig, default_pipeline_config
from tda_risk.methods import GapHS
from tda_risk.registry import load_candidate_registry
from tda_risk.topology import _landscape_topk_block, _window_to_topology_vector


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_registry_is_parseable_complete_and_unique():
    path = ROOT / "revision_config" / "recovery_candidate_registry.csv"
    registry = load_candidate_registry(path)
    assert len(registry) == 123
    assert registry["configuration_id"].is_unique
    assert set(registry.loc[registry.method == "rolling_hs", "window_length"]) == {250, 500, 1000}
    assert set(registry.loc[registry.method == "fhs_ewma", "lambda"]) == {0.94, 0.97}
    assert set(registry.loc[registry.method == "regime_hs", "regime_bins"]) == {2, 4}


def test_existing_validation_checkpoints_have_no_hidden_candidate():
    registry = load_candidate_registry(ROOT / "revision_config" / "recovery_candidate_registry.csv")
    allowed = set(registry["configuration_id"])
    checkpoint_dir = ROOT / "results" / "revision_phase2_validation" / "checkpoints"
    observed = {path.stem for path in checkpoint_dir.glob("*.csv")}
    assert observed <= allowed


def test_validation_script_instantiates_exact_registry_rows():
    from scripts.run_revision_validation import _method_specs

    specs = _method_specs(default_pipeline_config())
    registry = load_candidate_registry(ROOT / "revision_config" / "recovery_candidate_registry.csv")
    assert [config_id for config_id, _ in specs] == registry["configuration_id"].tolist()


def test_gaphs_and_h0_feature_modes_are_explicit_and_deterministic():
    assert GapHS(window=250, exclusion_radius=125).name == "gap_hs"
    config = TopologyConfig(landscape_num_steps=20, landscape_range=(0.0, 1.0))
    diagram = np.array([[0.0, np.inf], [0.1, 0.4], [0.2, 0.3]])
    h0 = _landscape_topk_block(diagram, config, hom_deg=0, top_k=1)
    assert h0.shape == (20,)
    assert np.isfinite(h0).all()
    window = np.sin(np.linspace(0.0, 8.0, 30))
    h0_only = _window_to_topology_vector(window, 1, 2, "landscape_h0_top1", config)
    h0h1 = _window_to_topology_vector(window, 1, 2, "landscape_h0h1_top1_concat", config)
    assert h0_only.shape == (20,)
    assert h0h1.shape == (40,)
    np.testing.assert_array_equal(h0_only, _window_to_topology_vector(window, 1, 2, "landscape_h0_top1", config))
