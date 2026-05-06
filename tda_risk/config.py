from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    start_date: str = "2000-01-01"
    yahoo_tickers: tuple[str, ...] = ("SPY", "IEF", "^VIX", "EURUSD=X", "CL=F")
    fred_series: tuple[str, ...] = ("DGS2", "DGS10", "T10Y2Y", "DFF", "BAMLH0A0HYM2", "BAA10Y")
    spy_weight: float = 0.6
    ief_weight: float = 0.4


@dataclass(frozen=True)
class SplitConfig:
    warmup_end: str = "2011-12-31"
    validation_start: str = "2012-01-01"
    validation_end: str = "2014-12-31"
    test_start: str = "2015-01-01"


@dataclass(frozen=True)
class RiskConfig:
    var_confidence: float = 0.99
    es_confidence: float = 0.975


@dataclass(frozen=True)
class BaselineGrid:
    hs_windows: tuple[int, ...] = (250, 500, 1000)
    fhs_lambdas: tuple[float, ...] = (0.94, 0.97)
    regime_bins: tuple[int, ...] = (2, 4)
    euclidean_k: tuple[int, ...] = (50, 100, 250, 500, 1000)
    analogue_window_lengths: tuple[int, ...] = (60, 125)


@dataclass(frozen=True)
class TopologyConfig:
    window_lengths: tuple[int, ...] = (60,)
    alphas: tuple[float, ...] = (0.5,)
    feature_modes: tuple[str, ...] = ("landscape1",)
    input_modes: tuple[str, ...] = ("portfolio_only",)
    n_jobs: int = 1
    tau_candidates: tuple[int, ...] = (1, 2, 3, 4, 5)
    embedding_dimension_candidates: tuple[int, ...] = (2, 3, 4, 5)
    mi_bins: int = 16
    fnn_ratio_threshold: float = 10.0
    fnn_absolute_threshold: float = 2.0
    fnn_acceptance_rate: float = 0.05
    landscape_num_steps: int = 200
    landscape_range: tuple[float, float] = (0.0, 3.0)
    landscape_topk_weights: tuple[float, ...] = (1.0, 0.5, 0.25)
    null_seed: int = 20260423


@dataclass(frozen=True)
class OutputConfig:
    root_dir: Path = Path("results/baseline_pipeline")
    panel_file: str = "aligned_panel.csv"
    state_file: str = "state_panel.csv"
    forecast_file: str = "forecasts.csv"
    summary_file: str = "summary.csv"
    skip_events_file: str = "skip_events.csv"
    skip_summary_file: str = "skip_summary.csv"


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    grid: BaselineGrid = field(default_factory=BaselineGrid)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    state_columns: tuple[str, ...] = (
        "rv20_p",
        "ret20_spy",
        "VIX",
        "Delta5_DGS10",
        "T10Y2Y",
        "HYOAS",
    )
    min_zscore_history: int = 60


def default_pipeline_config() -> PipelineConfig:
    return PipelineConfig()
