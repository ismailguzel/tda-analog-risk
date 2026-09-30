from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
import platform
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    from ripser import ripser
except ImportError:  # pragma: no cover - handled at runtime
    ripser = None

try:
    from persim import PersLandscapeApprox
except ImportError:  # pragma: no cover - handled at runtime
    PersLandscapeApprox = None

from .config import TopologyConfig
from .features import standardize_window
from .methods import (
    BaseMethod,
    MethodForecast,
    _get_return_window_matrix,
    _scenario_losses_from_positions,
    _select_top_positions,
    _state_knn_inputs,
    empirical_var_es,
)

_TOPOLOGY_CACHE_DATA_ID: tuple[int, int, tuple[str, ...]] | None = None
_TOPOLOGY_ARRAY_CACHE: dict[tuple[object, ...], np.ndarray | tuple[int, int]] = {}
_TOPOLOGY_INPUT_COLUMNS: dict[str, tuple[str, ...]] = {
    "portfolio_only": ("portfolio_return",),
    "portfolio_plus_components": ("portfolio_return", "SPY_return", "IEF_return"),
    "components_only": ("SPY_return", "IEF_return"),
}
_TOPOLOGY_FEATURE_MODE_TOPK: dict[str, tuple[bool, int]] = {
    "landscape1": (False, 1),
    "landscape_h0_top1": (True, 1),
    "landscape_h1_top3_concat": (False, 3),
    "landscape_h1_top3_weighted": (False, 3),
    "landscape_h0h1_top1_concat": (True, 1),
    "landscape_h0h1_top3_concat": (True, 3),
}


def _reset_topology_caches_if_needed(data: pd.DataFrame) -> None:
    global _TOPOLOGY_CACHE_DATA_ID
    data_signature = (id(data), len(data), tuple(str(column) for column in data.columns))
    if _TOPOLOGY_CACHE_DATA_ID != data_signature:
        _TOPOLOGY_CACHE_DATA_ID = data_signature
        _TOPOLOGY_ARRAY_CACHE.clear()


def clear_topology_caches() -> None:
    global _TOPOLOGY_CACHE_DATA_ID
    _TOPOLOGY_CACHE_DATA_ID = None
    _TOPOLOGY_ARRAY_CACHE.clear()


def _require_ripser() -> None:
    if ripser is None:
        raise RuntimeError(
            "Topology dependencies are not installed. Install requirements-topology.txt first."
        )


def _require_landscape_support() -> None:
    if PersLandscapeApprox is None:
        raise RuntimeError(
            "Landscape support is not installed. Install requirements-topology.txt first."
        )


def _delay_embed(window: np.ndarray, tau: int, embedding_dimension: int) -> np.ndarray:
    window_2d = np.asarray(window, dtype=float)
    if window_2d.ndim == 1:
        window_2d = window_2d[:, None]

    usable_length = window_2d.shape[0] - tau * (embedding_dimension - 1)
    if usable_length < 3:
        raise ValueError("Return window is too short for the requested delay embedding.")

    columns = [
        window_2d[offset * tau : offset * tau + usable_length]
        for offset in range(embedding_dimension)
    ]
    return np.concatenate(columns, axis=1)


def _topology_input_columns(input_mode: str) -> tuple[str, ...]:
    try:
        return _TOPOLOGY_INPUT_COLUMNS[input_mode]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported topology input mode: {input_mode}") from exc


def _training_returns(data: pd.DataFrame, training_end: str, input_mode: str) -> np.ndarray:
    _reset_topology_caches_if_needed(data)
    cache_key = ("training_returns", training_end, input_mode)
    if cache_key not in _TOPOLOGY_ARRAY_CACHE:
        training_mask = data.index <= pd.Timestamp(training_end)
        columns = _topology_input_columns(input_mode)
        returns = data.loc[training_mask, list(columns)].to_numpy(dtype=float)
        if returns.ndim == 1 or returns.shape[1] == 1:
            series = returns.reshape(-1)
        else:
            valid_counts = np.isfinite(returns).sum(axis=1)
            series = np.full(returns.shape[0], np.nan, dtype=float)
            valid_rows = valid_counts > 0
            series[valid_rows] = np.nansum(returns[valid_rows], axis=1) / valid_counts[valid_rows]
        _TOPOLOGY_ARRAY_CACHE[cache_key] = series[np.isfinite(series)]
    return _TOPOLOGY_ARRAY_CACHE[cache_key]  # type: ignore[return-value]


def _mutual_information_at_tau(series: np.ndarray, tau: int, bins: int) -> float:
    lead = series[tau:]
    lagged = series[:-tau]
    hist, _, _ = np.histogram2d(lagged, lead, bins=bins)
    joint = hist / np.maximum(hist.sum(), 1.0)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denominator = px @ py
    mask = joint > 0.0
    return float(np.sum(joint[mask] * np.log(joint[mask] / denominator[mask])))


def _select_tau(series: np.ndarray, topology: TopologyConfig) -> int:
    tau_values = list(topology.tau_candidates)
    mi_values = [float("inf")] * len(tau_values)
    for idx, tau in enumerate(tau_values):
        if series.size <= tau + 1:
            continue
        mi_values[idx] = _mutual_information_at_tau(series, tau=tau, bins=topology.mi_bins)

    for idx in range(1, len(mi_values) - 1):
        if mi_values[idx] <= mi_values[idx - 1] and mi_values[idx] <= mi_values[idx + 1]:
            return tau_values[idx]
    return tau_values[int(np.argmin(mi_values))]


def _delay_embed_with_length(
    series: np.ndarray,
    tau: int,
    embedding_dimension: int,
    usable_length: int,
) -> np.ndarray:
    columns = [
        series[offset * tau : offset * tau + usable_length] for offset in range(embedding_dimension)
    ]
    return np.column_stack(columns)


def _false_nearest_neighbor_rate(
    series: np.ndarray,
    tau: int,
    embedding_dimension: int,
    topology: TopologyConfig,
) -> float:
    usable_length = series.size - tau * embedding_dimension
    if usable_length < 10:
        return float("inf")

    embedded_m = _delay_embed_with_length(series, tau, embedding_dimension, usable_length)
    embedded_m1 = _delay_embed_with_length(series, tau, embedding_dimension + 1, usable_length)

    neighbors = NearestNeighbors(n_neighbors=2, algorithm="auto")
    neighbors.fit(embedded_m)
    distances, indices = neighbors.kneighbors(embedded_m)

    nearest_distance = distances[:, 1]
    nearest_indices = indices[:, 1]
    added_coordinate_gap = np.abs(
        embedded_m1[:, -1] - embedded_m1[nearest_indices, -1]
    )
    ratio = added_coordinate_gap / np.maximum(nearest_distance, 1e-8)
    distance_m1 = np.linalg.norm(embedded_m1 - embedded_m1[nearest_indices], axis=1)
    scale = np.std(series, ddof=0) + 1e-8

    false_mask = (
        (ratio > topology.fnn_ratio_threshold)
        | (distance_m1 / scale > topology.fnn_absolute_threshold)
    )
    return float(false_mask.mean())


def _select_embedding_dimension(series: np.ndarray, tau: int, topology: TopologyConfig) -> int:
    dimension_values = list(topology.embedding_dimension_candidates)
    rates: list[float] = []
    for dimension in dimension_values:
        rates.append(_false_nearest_neighbor_rate(series, tau, dimension, topology))

    for dimension, rate in zip(dimension_values, rates):
        if rate <= topology.fnn_acceptance_rate:
            return dimension
    return dimension_values[int(np.argmin(rates))]


def _select_embedding_parameters(
    data: pd.DataFrame,
    topology: TopologyConfig,
    training_end: str,
    input_mode: str,
) -> tuple[int, int]:
    _reset_topology_caches_if_needed(data)
    cache_key = (
        "embedding_parameters",
        training_end,
        input_mode,
        topology.tau_candidates,
        topology.embedding_dimension_candidates,
    )
    if cache_key in _TOPOLOGY_ARRAY_CACHE:
        return _TOPOLOGY_ARRAY_CACHE[cache_key]  # type: ignore[return-value]

    series = _training_returns(data, training_end, input_mode)
    if series.size < 200:
        raise ValueError("Training history is too short to estimate Takens parameters.")

    standardized = (series - series.mean()) / (series.std(ddof=0) + 1e-8)
    tau = _select_tau(standardized, topology)
    embedding_dimension = _select_embedding_dimension(standardized, tau=tau, topology=topology)
    params = (tau, embedding_dimension)
    _TOPOLOGY_ARRAY_CACHE[cache_key] = params
    return params


def _first_landscape_curve(diagram: np.ndarray, topology: TopologyConfig) -> np.ndarray:
    _require_landscape_support()

    finite = diagram[np.isfinite(diagram[:, 1])]
    if finite.size == 0:
        return np.zeros(topology.landscape_num_steps, dtype=float)

    landscape = PersLandscapeApprox(
        start=topology.landscape_range[0],
        stop=topology.landscape_range[1],
        num_steps=topology.landscape_num_steps,
        dgms=[np.empty((0, 2), dtype=float), finite],
        hom_deg=1,
    )
    if landscape.values.size == 0 or landscape.values.shape[0] == 0:
        return np.zeros(topology.landscape_num_steps, dtype=float)
    return np.asarray(landscape.values[0], dtype=float)


def _landscape_topk_block(
    diagram: np.ndarray,
    topology: TopologyConfig,
    *,
    hom_deg: int,
    top_k: int,
) -> np.ndarray:
    _require_landscape_support()
    finite = diagram[np.isfinite(diagram[:, 1])]
    if finite.size == 0:
        return np.zeros(topology.landscape_num_steps * top_k, dtype=float)

    if hom_deg == 0:
        dgms = [finite, np.empty((0, 2), dtype=float)]
    elif hom_deg == 1:
        dgms = [np.empty((0, 2), dtype=float), finite]
    else:
        raise ValueError(f"Unsupported homology degree for landscapes: {hom_deg}")

    try:
        landscape = PersLandscapeApprox(
            start=topology.landscape_range[0],
            stop=topology.landscape_range[1],
            num_steps=topology.landscape_num_steps,
            dgms=dgms,
            hom_deg=hom_deg,
        )
    except Exception:
        return np.zeros(topology.landscape_num_steps * top_k, dtype=float)

    if landscape.values.size == 0:
        return np.zeros(topology.landscape_num_steps * top_k, dtype=float)

    values = np.asarray(landscape.values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    clipped = values[:top_k]
    if clipped.shape[0] < top_k:
        pad = np.zeros((top_k - clipped.shape[0], topology.landscape_num_steps), dtype=float)
        clipped = np.vstack([clipped, pad])
    return clipped.reshape(-1)


def _topology_feature_length(feature_mode: str, topology: TopologyConfig) -> int:
    try:
        use_h0, top_k = _TOPOLOGY_FEATURE_MODE_TOPK[feature_mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported topology feature mode: {feature_mode}") from exc
    block_count = 1 if feature_mode == "landscape_h0_top1" else (2 if use_h0 else 1)
    base_length = block_count * top_k * topology.landscape_num_steps
    if feature_mode == "landscape1":
        return base_length + 1  # keep area statistic for legacy finalist mode
    if feature_mode == "landscape_h1_top3_weighted":
        return topology.landscape_num_steps
    return base_length


def _standardize_topology_window(window: np.ndarray) -> np.ndarray:
    return standardize_window(window)


def _get_topology_input_matrix(data: pd.DataFrame, input_mode: str) -> np.ndarray:
    _reset_topology_caches_if_needed(data)
    columns = _topology_input_columns(input_mode)
    cache_key = ("topology_input_matrix", input_mode, columns)
    if cache_key not in _TOPOLOGY_ARRAY_CACHE:
        _TOPOLOGY_ARRAY_CACHE[cache_key] = data.loc[:, list(columns)].to_numpy(dtype=float)
    return _TOPOLOGY_ARRAY_CACHE[cache_key]  # type: ignore[return-value]


def _get_topology_window_matrix(
    data: pd.DataFrame,
    window_length: int,
    input_mode: str,
) -> np.ndarray:
    _reset_topology_caches_if_needed(data)
    cache_key = ("topology_windows", window_length, input_mode)
    if cache_key not in _TOPOLOGY_ARRAY_CACHE:
        inputs = _get_topology_input_matrix(data, input_mode)
        window_matrix = np.full(
            (len(data), window_length, inputs.shape[1]),
            np.nan,
            dtype=float,
        )
        for position in range(window_length - 1, len(data)):
            window_matrix[position] = inputs[position - window_length + 1 : position + 1]
        _TOPOLOGY_ARRAY_CACHE[cache_key] = window_matrix
    return _TOPOLOGY_ARRAY_CACHE[cache_key]  # type: ignore[return-value]


def _window_to_diagrams(
    window: np.ndarray,
    tau: int,
    embedding_dimension: int,
) -> tuple[np.ndarray, ...]:
    _require_ripser()

    point_cloud = _delay_embed(
        _standardize_topology_window(window),
        tau=tau,
        embedding_dimension=embedding_dimension,
    )
    return tuple(np.asarray(diagram, dtype=float) for diagram in ripser(point_cloud, maxdim=1)["dgms"])


def _window_to_topology_vector(
    window: np.ndarray,
    tau: int,
    embedding_dimension: int,
    feature_mode: str,
    topology: TopologyConfig,
) -> np.ndarray:
    diagrams = _window_to_diagrams(
        window,
        tau=tau,
        embedding_dimension=embedding_dimension,
    )

    if feature_mode == "landscape1":
        landscape = _first_landscape_curve(diagrams[1], topology)
        area = np.trapezoid(
            landscape,
            dx=(topology.landscape_range[1] - topology.landscape_range[0])
            / max(topology.landscape_num_steps - 1, 1),
        )
        return np.concatenate([landscape, np.array([area], dtype=float)])

    use_h0, top_k = _TOPOLOGY_FEATURE_MODE_TOPK.get(feature_mode, (None, None))
    if use_h0 is None or top_k is None:
        raise ValueError(f"Unsupported topology feature mode: {feature_mode}")

    h1_block = _landscape_topk_block(diagrams[1], topology, hom_deg=1, top_k=top_k)
    if feature_mode == "landscape_h1_top3_weighted":
        h1_matrix = h1_block.reshape(top_k, topology.landscape_num_steps)
        raw_weights = np.asarray(topology.landscape_topk_weights[:top_k], dtype=float)
        if raw_weights.size < top_k:
            raw_weights = np.pad(raw_weights, (0, top_k - raw_weights.size), mode="constant")
        raw_weights = np.where(raw_weights > 0.0, raw_weights, 0.0)
        if float(raw_weights.sum()) <= 0.0:
            raw_weights = np.ones(top_k, dtype=float)
        weights = raw_weights / raw_weights.sum()
        return np.average(h1_matrix, axis=0, weights=weights)

    if feature_mode == "landscape_h0_top1":
        return _landscape_topk_block(diagrams[0], topology, hom_deg=0, top_k=top_k)
    if not use_h0:
        return h1_block

    h0_block = _landscape_topk_block(diagrams[0], topology, hom_deg=0, top_k=top_k)
    return np.concatenate([h0_block, h1_block])


def _topology_vector_task(
    task: tuple[int, np.ndarray, int, int, str, TopologyConfig],
) -> tuple[int, np.ndarray | None]:
    position, window, tau, embedding_dimension, feature_mode, topology = task
    try:
        vector = _window_to_topology_vector(
            window,
            tau=tau,
            embedding_dimension=embedding_dimension,
            feature_mode=feature_mode,
            topology=topology,
        )
    except ValueError:
        return position, None
    return position, vector


def _topology_worker_count(topology: TopologyConfig) -> int:
    requested = max(int(topology.n_jobs), 1)
    if requested == 1:
        return 1
    # On macOS, ProcessPoolExecutor(spawn) has ~3-15x overhead vs sequential
    # for short-lived tasks (~5ms/window).  Force single-process to avoid
    # silently degrading runtime.
    if platform.system() == "Darwin":
        print(
            "[topology] WARNING: --n-jobs > 1 is slower on macOS (spawn overhead)."
            " Overriding to n_jobs=1.  Use Linux/HPC for true parallelism.",
            file=sys.stderr,
        )
        return 1
    available = os.cpu_count() or 1
    return min(requested, available)


def _get_topology_feature_matrix(
    data: pd.DataFrame,
    window_length: int,
    input_mode: str,
    feature_mode: str,
    topology: TopologyConfig,
    training_end: str,
    *,
    use_null: bool,
) -> np.ndarray:
    _reset_topology_caches_if_needed(data)
    tau, embedding_dimension = _select_embedding_parameters(
        data,
        topology,
        training_end,
        input_mode,
    )

    feature_key = (
        "topology_feature_matrix",
        window_length,
        input_mode,
        feature_mode,
        tau,
        embedding_dimension,
        topology.landscape_num_steps,
        topology.landscape_range,
        topology.n_jobs,
    )
    if feature_key not in _TOPOLOGY_ARRAY_CACHE:
        windows = _get_topology_window_matrix(data, window_length, input_mode)
        feature_length = _topology_feature_length(feature_mode, topology)
        feature_matrix = np.full((len(data), feature_length), np.nan, dtype=float)
        valid_mask = np.isfinite(windows).all(axis=(1, 2))
        valid_positions = np.flatnonzero(valid_mask)

        worker_count = _topology_worker_count(topology)
        if worker_count == 1 or valid_positions.size < 2:
            for position in valid_positions:
                task_position, vector = _topology_vector_task(
                    (
                        int(position),
                        windows[position],
                        tau,
                        embedding_dimension,
                        feature_mode,
                        topology,
                    )
                )
                if vector is not None:
                    feature_matrix[task_position] = vector
        else:
            tasks = [
                (
                    int(position),
                    windows[position],
                    tau,
                    embedding_dimension,
                    feature_mode,
                    topology,
                )
                for position in valid_positions
            ]
            chunk_size = max(1, len(tasks) // (worker_count * 4))
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                for task_position, vector in executor.map(
                    _topology_vector_task,
                    tasks,
                    chunksize=chunk_size,
                ):
                    if vector is not None:
                        feature_matrix[task_position] = vector

        _TOPOLOGY_ARRAY_CACHE[feature_key] = feature_matrix

    if not use_null:
        return _TOPOLOGY_ARRAY_CACHE[feature_key]  # type: ignore[return-value]

    null_key = ("topology_feature_matrix_null", feature_key, topology.null_seed)
    if null_key not in _TOPOLOGY_ARRAY_CACHE:
        base = _TOPOLOGY_ARRAY_CACHE[feature_key]  # type: ignore[assignment]
        null_matrix = np.full_like(base, np.nan)
        valid_positions = np.flatnonzero(np.isfinite(base).all(axis=1))
        rng = np.random.default_rng(topology.null_seed + 100 * window_length + tau + embedding_dimension)
        permuted_positions = rng.permutation(valid_positions)
        null_matrix[valid_positions] = base[permuted_positions]
        _TOPOLOGY_ARRAY_CACHE[null_key] = null_matrix

    return _TOPOLOGY_ARRAY_CACHE[null_key]  # type: ignore[return-value]


def _standardize_topology_features(
    history_features: np.ndarray,
    current_feature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = history_features.mean(axis=0)
    scale = history_features.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (history_features - mean) / scale, (current_feature - mean) / scale


def _topology_top_positions(
    data: pd.DataFrame,
    position: int,
    state_columns: tuple[str, ...],
    window_length: int,
    input_mode: str,
    feature_mode: str,
    alpha: float,
    max_k: int,
    topology: TopologyConfig,
    training_end: str,
    *,
    use_null: bool,
    exclusion_radius: int = 0,
) -> np.ndarray:
    _reset_topology_caches_if_needed(data)
    tau, embedding_dimension = _select_embedding_parameters(
        data,
        topology,
        training_end,
        input_mode,
    )

    cache_key = (
        "topology_top_positions",
        position,
        state_columns,
        window_length,
        input_mode,
        feature_mode,
        tau,
        embedding_dimension,
        alpha,
        max_k,
        use_null,
        exclusion_radius,
    )
    if cache_key in _TOPOLOGY_ARRAY_CACHE:
        return _TOPOLOGY_ARRAY_CACHE[cache_key]

    current_vector, history_vectors, candidate_positions = _state_knn_inputs(
        data, position, state_columns, exclusion_radius=exclusion_radius
    )
    feature_matrix = _get_topology_feature_matrix(
        data,
        window_length=window_length,
        input_mode=input_mode,
        feature_mode=feature_mode,
        topology=topology,
        training_end=training_end,
        use_null=use_null,
    )
    current_feature = feature_matrix[position]
    if not np.isfinite(current_feature).all():
        raise ValueError("Current topology representation is not available.")

    feature_mask = np.isfinite(feature_matrix[candidate_positions]).all(axis=1)
    candidate_positions = candidate_positions[feature_mask]
    history_vectors = history_vectors[feature_mask]
    if candidate_positions.size == 0:
        raise ValueError("No topology-aware analogue candidates are available.")

    history_features = feature_matrix[candidate_positions]
    history_features, current_feature = _standardize_topology_features(
        history_features,
        current_feature,
    )
    state_distances = np.linalg.norm(history_vectors - current_vector, axis=1)
    topology_distances = np.linalg.norm(history_features - current_feature, axis=1)
    distances = alpha * state_distances + (1.0 - alpha) * topology_distances

    top_positions = _select_top_positions(candidate_positions, distances, max_k)
    _TOPOLOGY_ARRAY_CACHE[cache_key] = top_positions
    return top_positions


@dataclass(frozen=True)
class TopologyKNN(BaseMethod):
    k: int
    state_columns: tuple[str, ...]
    max_k: int
    window_length: int
    input_mode: str
    feature_mode: str
    alpha: float
    topology: TopologyConfig
    training_end: str
    exclusion_radius: int = 0
    name: str = "topology_knn"

    def forecast(self, data: pd.DataFrame, position: int, risk) -> MethodForecast:
        tau, embedding_dimension = _select_embedding_parameters(
            data,
            self.topology,
            self.training_end,
            self.input_mode,
        )
        nearest_positions = _topology_top_positions(
            data,
            position,
            state_columns=self.state_columns,
            window_length=self.window_length,
            input_mode=self.input_mode,
            feature_mode=self.feature_mode,
            alpha=self.alpha,
            max_k=self.max_k,
            topology=self.topology,
            training_end=self.training_end,
            use_null=False,
            exclusion_radius=self.exclusion_radius,
        )
        scenario_losses = _scenario_losses_from_positions(data, nearest_positions[: self.k])
        var, es = empirical_var_es(scenario_losses, risk.var_confidence, risk.es_confidence)
        return MethodForecast(
            self.name,
            {
                "window_length": self.window_length,
                "topology_input_mode": self.input_mode,
                "feature_mode": self.feature_mode,
                "tau": tau,
                "embedding_dimension": embedding_dimension,
                "alpha": self.alpha,
                "k": self.k,
            },
            scenario_losses.size,
            var,
            es,
            scenario_losses,
        )


@dataclass(frozen=True)
class TopologyPlaceboKNN(BaseMethod):
    """Legacy submitted-version placebo; not used in the revised final workflow."""

    k: int
    state_columns: tuple[str, ...]
    max_k: int
    window_length: int
    input_mode: str
    feature_mode: str
    alpha: float
    topology: TopologyConfig
    training_end: str
    exclusion_radius: int = 0
    name: str = "topology_placebo_knn"

    def forecast(self, data: pd.DataFrame, position: int, risk) -> MethodForecast:
        tau, embedding_dimension = _select_embedding_parameters(
            data,
            self.topology,
            self.training_end,
            self.input_mode,
        )
        nearest_positions = _topology_top_positions(
            data,
            position,
            state_columns=self.state_columns,
            window_length=self.window_length,
            input_mode=self.input_mode,
            feature_mode=self.feature_mode,
            alpha=self.alpha,
            max_k=self.max_k,
            topology=self.topology,
            training_end=self.training_end,
            use_null=True,
            exclusion_radius=self.exclusion_radius,
        )
        scenario_losses = _scenario_losses_from_positions(data, nearest_positions[: self.k])
        var, es = empirical_var_es(scenario_losses, risk.var_confidence, risk.es_confidence)
        return MethodForecast(
            self.name,
            {
                "window_length": self.window_length,
                "topology_input_mode": self.input_mode,
                "feature_mode": self.feature_mode,
                "tau": tau,
                "embedding_dimension": embedding_dimension,
                "alpha": self.alpha,
                "k": self.k,
            },
            scenario_losses.size,
            var,
            es,
            scenario_losses,
        )


def build_topology_methods(
    topology: TopologyConfig,
    state_columns: tuple[str, ...],
    k_values: tuple[int, ...],
    training_end: str,
) -> list[BaseMethod]:
    max_k = max(k_values)
    methods: list[BaseMethod] = []
    for window_length in topology.window_lengths:
        for input_mode in topology.input_modes:
            for feature_mode in topology.feature_modes:
                for alpha in topology.alphas:
                    methods.extend(
                        TopologyKNN(
                            k=k,
                            state_columns=state_columns,
                            max_k=max_k,
                            window_length=window_length,
                            input_mode=input_mode,
                            feature_mode=feature_mode,
                            alpha=alpha,
                            topology=topology,
                            training_end=training_end,
                        )
                        for k in k_values
                    )
                    methods.extend(
                        TopologyPlaceboKNN(
                            k=k,
                            state_columns=state_columns,
                            max_k=max_k,
                            window_length=window_length,
                            input_mode=input_mode,
                            feature_mode=feature_mode,
                            alpha=alpha,
                            topology=topology,
                            training_end=training_end,
                        )
                        for k in k_values
                    )
    return methods
