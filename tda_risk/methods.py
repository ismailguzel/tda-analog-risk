from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from dtaidistance import dtw
from numpy.lib.stride_tricks import sliding_window_view
try:
    from arch import arch_model
except ImportError:  # pragma: no cover - handled at runtime
    arch_model = None

from .config import BaselineGrid, RiskConfig


_CACHE_DATA_ID: int | None = None
_METHOD_CACHE: dict[tuple[object, ...], np.ndarray] = {}
_ARRAY_CACHE: dict[tuple[object, ...], np.ndarray] = {}


@dataclass(frozen=True)
class MethodForecast:
    method: str
    params: dict[str, Any]
    scenario_count: int
    var: float
    es: float


def empirical_var_es(
    losses: np.ndarray,
    var_confidence: float,
    es_confidence: float,
) -> tuple[float, float]:
    clean = np.asarray(losses, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("No finite scenario losses available.")

    var_level = float(np.quantile(clean, var_confidence))
    es_threshold = float(np.quantile(clean, es_confidence))
    es_tail = clean[clean >= es_threshold]
    es_value = float(es_tail.mean()) if es_tail.size else es_threshold
    return var_level, es_value


class BaseMethod:
    name: str

    def forecast(self, data: pd.DataFrame, position: int, risk: RiskConfig) -> MethodForecast:
        raise NotImplementedError


def _reset_caches_if_needed(data: pd.DataFrame) -> None:
    global _CACHE_DATA_ID
    data_id = id(data)
    if _CACHE_DATA_ID != data_id:
        _CACHE_DATA_ID = data_id
        _METHOD_CACHE.clear()
        _ARRAY_CACHE.clear()


def clear_method_caches() -> None:
    global _CACHE_DATA_ID
    _CACHE_DATA_ID = None
    _METHOD_CACHE.clear()
    _ARRAY_CACHE.clear()


def _get_loss_array(data: pd.DataFrame) -> np.ndarray:
    _reset_caches_if_needed(data)
    cache_key = ("portfolio_loss",)
    if cache_key not in _ARRAY_CACHE:
        _ARRAY_CACHE[cache_key] = data["portfolio_loss"].to_numpy(dtype=float)
    return _ARRAY_CACHE[cache_key]


def _get_z_matrix(data: pd.DataFrame, state_columns: tuple[str, ...]) -> np.ndarray:
    _reset_caches_if_needed(data)
    cache_key = ("z_matrix", state_columns)
    if cache_key not in _ARRAY_CACHE:
        z_columns = [f"z_{column}" for column in state_columns]
        _ARRAY_CACHE[cache_key] = data.loc[:, z_columns].to_numpy(dtype=float)
    return _ARRAY_CACHE[cache_key]


def _get_return_window_matrix(data: pd.DataFrame, window_length: int) -> np.ndarray:
    _reset_caches_if_needed(data)
    cache_key = ("portfolio_return_windows", window_length)
    if cache_key not in _ARRAY_CACHE:
        returns = data["portfolio_return"].to_numpy(dtype=float)
        windows = np.full((returns.size, window_length), np.nan, dtype=float)
        if returns.size >= window_length:
            windows[window_length - 1 :] = sliding_window_view(returns, window_shape=window_length)
        _ARRAY_CACHE[cache_key] = windows
    return _ARRAY_CACHE[cache_key]


def _select_top_positions(
    candidate_positions: np.ndarray,
    distances: np.ndarray,
    max_k: int,
) -> np.ndarray:
    if candidate_positions.size == 0:
        raise ValueError("No analogue candidates are available.")

    take = min(max_k, candidate_positions.size)
    if take == candidate_positions.size:
        order = np.argsort(distances)
    else:
        partial = np.argpartition(distances, take - 1)[:take]
        order = partial[np.argsort(distances[partial])]
    return candidate_positions[order]


def _state_knn_inputs(
    data: pd.DataFrame,
    position: int,
    state_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_matrix = _get_z_matrix(data, state_columns)
    current_vector = z_matrix[position]
    if not np.isfinite(current_vector).all():
        raise ValueError("Current standardized state is not available.")

    history_matrix = z_matrix[:position]
    candidate_mask = np.isfinite(history_matrix).all(axis=1)
    candidate_positions = np.flatnonzero(candidate_mask)
    candidate_positions = candidate_positions[candidate_positions + 1 <= position]
    if candidate_positions.size == 0:
        raise ValueError("No analogue candidates are available.")

    history_vectors = z_matrix[candidate_positions]
    return current_vector, history_vectors, candidate_positions


def _window_knn_inputs(
    data: pd.DataFrame,
    position: int,
    window_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_matrix = _get_return_window_matrix(data, window_length)
    current_window = window_matrix[position]
    if not np.isfinite(current_window).all():
        raise ValueError("Current return window is not available.")

    history_windows = window_matrix[:position]
    candidate_mask = np.isfinite(history_windows).all(axis=1)
    candidate_positions = np.flatnonzero(candidate_mask)
    candidate_positions = candidate_positions[candidate_positions + 1 <= position]
    if candidate_positions.size == 0:
        raise ValueError("No analogue candidates are available.")

    return current_window, window_matrix[candidate_positions], candidate_positions


def _euclidean_top_positions(
    data: pd.DataFrame,
    position: int,
    state_columns: tuple[str, ...],
    max_k: int,
) -> np.ndarray:
    cache_key = ("euclidean_knn", position, state_columns, max_k)
    if cache_key in _METHOD_CACHE:
        return _METHOD_CACHE[cache_key]

    current_vector, history_vectors, candidate_positions = _state_knn_inputs(
        data, position, state_columns
    )
    distances = np.linalg.norm(history_vectors - current_vector, axis=1)
    top_positions = _select_top_positions(candidate_positions, distances, max_k)
    _METHOD_CACHE[cache_key] = top_positions
    return top_positions


def _mahalanobis_top_positions(
    data: pd.DataFrame,
    position: int,
    state_columns: tuple[str, ...],
    max_k: int,
) -> np.ndarray:
    cache_key = ("mahalanobis_knn", position, state_columns, max_k)
    if cache_key in _METHOD_CACHE:
        return _METHOD_CACHE[cache_key]

    current_vector, history_vectors, candidate_positions = _state_knn_inputs(
        data, position, state_columns
    )
    covariance = np.cov(history_vectors, rowvar=False, ddof=0)
    covariance = np.atleast_2d(covariance)
    covariance = covariance + 1e-6 * np.eye(covariance.shape[0])
    inverse_covariance = np.linalg.pinv(covariance)

    diff = history_vectors - current_vector
    squared = np.einsum("ij,jk,ik->i", diff, inverse_covariance, diff)
    distances = np.sqrt(np.maximum(squared, 0.0))
    top_positions = _select_top_positions(candidate_positions, distances, max_k)
    _METHOD_CACHE[cache_key] = top_positions
    return top_positions


def _dtw_top_positions(
    data: pd.DataFrame,
    position: int,
    window_length: int,
    band: int,
    max_k: int,
) -> np.ndarray:
    cache_key = ("dtw_window", position, window_length, band, max_k)
    if cache_key in _METHOD_CACHE:
        return _METHOD_CACHE[cache_key]

    current_window, history_windows, candidate_positions = _window_knn_inputs(
        data, position, window_length
    )
    stacked = np.vstack([current_window, history_windows]).astype(np.double, copy=False)
    distances = np.asarray(
        dtw.distance_matrix_fast(
            stacked,
            block=((0, 1), (1, stacked.shape[0])),
            compact=True,
            window=band,
            parallel=False,
            use_pruning=True,
        ),
        dtype=float,
    )
    top_positions = _select_top_positions(candidate_positions, distances, max_k)
    _METHOD_CACHE[cache_key] = top_positions
    return top_positions


def _window_euclidean_top_positions(
    data: pd.DataFrame,
    position: int,
    window_length: int,
    max_k: int,
) -> np.ndarray:
    cache_key = ("window_euclidean_knn", position, window_length, max_k)
    if cache_key in _METHOD_CACHE:
        return _METHOD_CACHE[cache_key]

    current_window, history_windows, candidate_positions = _window_knn_inputs(
        data, position, window_length
    )
    distances = np.linalg.norm(history_windows - current_window, axis=1)
    top_positions = _select_top_positions(candidate_positions, distances, max_k)
    _METHOD_CACHE[cache_key] = top_positions
    return top_positions


def _fpca_top_positions(
    data: pd.DataFrame,
    position: int,
    window_length: int,
    n_components: int,
    max_k: int,
) -> np.ndarray:
    cache_key = ("fpca_window", position, window_length, n_components, max_k)
    if cache_key in _METHOD_CACHE:
        return _METHOD_CACHE[cache_key]

    current_window, history_windows, candidate_positions = _window_knn_inputs(
        data, position, window_length
    )
    if history_windows.shape[0] < n_components:
        raise ValueError("Not enough windows are available for FPCA.")

    mean_window = history_windows.mean(axis=0)
    centered_history = history_windows - mean_window
    covariance = centered_history.T @ centered_history / history_windows.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    components = eigenvectors[:, order]

    history_scores = centered_history @ components
    current_scores = (current_window - mean_window) @ components
    distances = np.linalg.norm(history_scores - current_scores, axis=1)
    top_positions = _select_top_positions(candidate_positions, distances, max_k)
    _METHOD_CACHE[cache_key] = top_positions
    return top_positions


def _scenario_losses_from_positions(data: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    losses = _get_loss_array(data)
    scenario_losses = losses[positions + 1]
    return scenario_losses[np.isfinite(scenario_losses)]


@dataclass(frozen=True)
class RollingHS(BaseMethod):
    window: int
    name: str = "rolling_hs"

    def forecast(self, data: pd.DataFrame, position: int, risk: RiskConfig) -> MethodForecast:
        history = data["portfolio_loss"].iloc[: position + 1].dropna().tail(self.window).to_numpy()
        var, es = empirical_var_es(history, risk.var_confidence, risk.es_confidence)
        return MethodForecast(self.name, {"window": self.window}, history.size, var, es)


@dataclass(frozen=True)
class FHSEWMA(BaseMethod):
    lambda_: float
    name: str = "fhs_ewma"

    def forecast(self, data: pd.DataFrame, position: int, risk: RiskConfig) -> MethodForecast:
        returns = data["portfolio_return"].iloc[: position + 1].dropna().to_numpy()
        sigma = ewma_sigma(returns, self.lambda_)
        residuals = returns / sigma
        sigma_next = float(np.sqrt(self.lambda_ * sigma[-1] ** 2 + (1.0 - self.lambda_) * returns[-1] ** 2))
        scenario_losses = -(sigma_next * residuals)
        var, es = empirical_var_es(scenario_losses, risk.var_confidence, risk.es_confidence)
        return MethodForecast(self.name, {"lambda": self.lambda_}, scenario_losses.size, var, es)


@dataclass(frozen=True)
class RegimeHS(BaseMethod):
    regime_bins: int
    name: str = "regime_hs"

    def forecast(self, data: pd.DataFrame, position: int, risk: RiskConfig) -> MethodForecast:
        rv_history = data["rv20_p"].iloc[: position + 1]
        current_value = rv_history.iloc[position]
        historical_values = rv_history.iloc[:position].dropna()
        if historical_values.empty or not np.isfinite(current_value):
            losses = data["portfolio_loss"].iloc[: position + 1].dropna().to_numpy()
        else:
            quantiles = historical_values.quantile(
                np.linspace(0.0, 1.0, self.regime_bins + 1)[1:-1]
            ).to_numpy()
            historical_labels = np.digitize(historical_values.to_numpy(), quantiles, right=True)
            current_label = int(np.digitize([current_value], quantiles, right=True)[0])
            matched_index = historical_values.index[historical_labels == current_label]
            losses = data.loc[matched_index, "portfolio_loss"].dropna().to_numpy()
            if losses.size == 0:
                losses = data["portfolio_loss"].iloc[: position + 1].dropna().to_numpy()

        var, es = empirical_var_es(losses, risk.var_confidence, risk.es_confidence)
        return MethodForecast(self.name, {"regime_bins": self.regime_bins}, losses.size, var, es)


@dataclass(frozen=True)
class EuclideanKNN(BaseMethod):
    k: int
    state_columns: tuple[str, ...]
    max_k: int
    name: str = "euclidean_knn"

    def forecast(self, data: pd.DataFrame, position: int, risk: RiskConfig) -> MethodForecast:
        nearest_positions = _euclidean_top_positions(data, position, self.state_columns, self.max_k)
        scenario_losses = _scenario_losses_from_positions(data, nearest_positions[: self.k])
        var, es = empirical_var_es(scenario_losses, risk.var_confidence, risk.es_confidence)
        return MethodForecast(self.name, {"k": self.k}, scenario_losses.size, var, es)


@dataclass(frozen=True)
class RandomKNN(BaseMethod):
    k: int
    state_columns: tuple[str, ...]
    max_k: int
    random_seed: int = 20260426
    name: str = "random_knn"

    def forecast(self, data: pd.DataFrame, position: int, risk: RiskConfig) -> MethodForecast:
        _, _, candidate_positions = _state_knn_inputs(data, position, self.state_columns)
        if candidate_positions.size == 0:
            raise ValueError("No analogue candidates are available.")
        take = min(self.k, candidate_positions.size)
        rng = np.random.default_rng(self.random_seed + position)
        chosen = rng.choice(candidate_positions, size=take, replace=False)
        scenario_losses = _scenario_losses_from_positions(data, chosen)
        var, es = empirical_var_es(scenario_losses, risk.var_confidence, risk.es_confidence)
        return MethodForecast(self.name, {"k": self.k}, scenario_losses.size, var, es)


@dataclass(frozen=True)
class MahalanobisKNN(BaseMethod):
    k: int
    state_columns: tuple[str, ...]
    max_k: int
    name: str = "mahalanobis_knn"

    def forecast(self, data: pd.DataFrame, position: int, risk: RiskConfig) -> MethodForecast:
        nearest_positions = _mahalanobis_top_positions(data, position, self.state_columns, self.max_k)
        scenario_losses = _scenario_losses_from_positions(data, nearest_positions[: self.k])
        var, es = empirical_var_es(scenario_losses, risk.var_confidence, risk.es_confidence)
        return MethodForecast(self.name, {"k": self.k}, scenario_losses.size, var, es)


@dataclass(frozen=True)
class DTWWindow(BaseMethod):
    k: int
    window_length: int
    max_k: int
    band: int = 10
    name: str = "dtw_window"

    def forecast(self, data: pd.DataFrame, position: int, risk: RiskConfig) -> MethodForecast:
        nearest_positions = _dtw_top_positions(
            data,
            position,
            window_length=self.window_length,
            band=self.band,
            max_k=self.max_k,
        )
        scenario_losses = _scenario_losses_from_positions(data, nearest_positions[: self.k])
        var, es = empirical_var_es(scenario_losses, risk.var_confidence, risk.es_confidence)
        return MethodForecast(
            self.name,
            {"window_length": self.window_length, "k": self.k},
            scenario_losses.size,
            var,
            es,
        )


@dataclass(frozen=True)
class WindowEuclideanKNN(BaseMethod):
    k: int
    window_length: int
    max_k: int
    name: str = "window_euclidean_knn"

    def forecast(self, data: pd.DataFrame, position: int, risk: RiskConfig) -> MethodForecast:
        nearest_positions = _window_euclidean_top_positions(
            data,
            position,
            window_length=self.window_length,
            max_k=self.max_k,
        )
        scenario_losses = _scenario_losses_from_positions(data, nearest_positions[: self.k])
        var, es = empirical_var_es(scenario_losses, risk.var_confidence, risk.es_confidence)
        return MethodForecast(
            self.name,
            {"window_length": self.window_length, "k": self.k},
            scenario_losses.size,
            var,
            es,
        )


@dataclass(frozen=True)
class FPCAWindow(BaseMethod):
    k: int
    window_length: int
    max_k: int
    n_components: int = 5
    name: str = "fpca_window"

    def forecast(self, data: pd.DataFrame, position: int, risk: RiskConfig) -> MethodForecast:
        nearest_positions = _fpca_top_positions(
            data,
            position,
            window_length=self.window_length,
            n_components=self.n_components,
            max_k=self.max_k,
        )
        scenario_losses = _scenario_losses_from_positions(data, nearest_positions[: self.k])
        var, es = empirical_var_es(scenario_losses, risk.var_confidence, risk.es_confidence)
        return MethodForecast(
            self.name,
            {"window_length": self.window_length, "k": self.k},
            scenario_losses.size,
            var,
            es,
        )


@dataclass(frozen=True)
class GARCHStudentT(BaseMethod):
    simulation_draws: int = 20000
    min_history: int = 500
    fit_window: int = 1500
    refit_interval: int = 20
    random_seed: int = 20260426
    name: str = "garch_t"

    def forecast(self, data: pd.DataFrame, position: int, risk: RiskConfig) -> MethodForecast:
        if arch_model is None:
            raise ValueError("arch package is not installed; GARCH baseline unavailable.")

        returns = data["portfolio_return"].iloc[: position + 1].dropna().to_numpy(dtype=float)
        if returns.size < self.min_history:
            raise ValueError("Not enough history for stable GARCH estimation.")
        scaled_returns = returns[-self.fit_window :] * 100.0
        bucket_position = position // self.refit_interval
        cache_key = (
            "garch_t_params",
            bucket_position,
            self.fit_window,
            self.refit_interval,
        )
        if cache_key in _METHOD_CACHE:
            params = _METHOD_CACHE[cache_key]
        else:
            model = arch_model(
                scaled_returns,
                mean="Constant",
                vol="GARCH",
                p=1,
                q=1,
                dist="StudentsT",
                rescale=False,
            )
            fit = model.fit(disp="off", show_warning=False)
            params = np.array(
                [
                    float(fit.params.get("mu", 0.0)),
                    float(fit.params.get("omega", 0.01)),
                    float(fit.params.get("alpha[1]", 0.05)),
                    float(fit.params.get("beta[1]", 0.9)),
                    float(fit.params.get("nu", 8.0)),
                ],
                dtype=float,
            )
            _METHOD_CACHE[cache_key] = params

        mu_next, omega, alpha, beta, nu = params.tolist()
        alpha = float(np.clip(alpha, 0.0, 0.999))
        beta = float(np.clip(beta, 0.0, 0.999))
        if alpha + beta >= 0.999:
            scale_down = 0.999 / (alpha + beta + 1e-12)
            alpha *= scale_down
            beta *= scale_down

        demeaned = scaled_returns - mu_next
        unconditional = omega / max(1.0 - alpha - beta, 1e-6)
        variance_t = max(float(np.var(demeaned, ddof=0)), 1e-6, unconditional)
        for value in demeaned:
            variance_t = omega + alpha * value * value + beta * variance_t
        sigma_next = float(np.sqrt(max(variance_t, 1e-12)))

        nu = max(nu, 2.1)
        scale = np.sqrt((nu - 2.0) / nu)
        rng = np.random.default_rng(self.random_seed + position)
        z = rng.standard_t(nu, size=self.simulation_draws) * scale
        scenario_returns = (mu_next + sigma_next * z) / 100.0
        scenario_losses = -scenario_returns

        var, es = empirical_var_es(scenario_losses, risk.var_confidence, risk.es_confidence)
        return MethodForecast(
            self.name,
            {
                "dist": "students_t",
                "sim_draws": self.simulation_draws,
                "nu": nu,
                "refit_interval": self.refit_interval,
            },
            scenario_losses.size,
            var,
            es,
        )


def ewma_sigma(returns: np.ndarray, lambda_: float) -> np.ndarray:
    clean = np.asarray(returns, dtype=float)
    sigma = np.empty_like(clean)
    initial_window = clean[: min(20, clean.size)]
    initial_sigma = float(np.std(initial_window, ddof=0))
    sigma[0] = initial_sigma if initial_sigma > 0.0 else 1e-8
    for idx in range(1, clean.size):
        sigma[idx] = np.sqrt(lambda_ * sigma[idx - 1] ** 2 + (1.0 - lambda_) * clean[idx - 1] ** 2)
        if sigma[idx] == 0.0:
            sigma[idx] = 1e-8
    return sigma


def build_baseline_methods(grid: BaselineGrid, state_columns: tuple[str, ...]) -> list[BaseMethod]:
    methods: list[BaseMethod] = []
    analogue_max_k = max(grid.euclidean_k)
    methods.extend(RollingHS(window=window) for window in grid.hs_windows)
    methods.extend(FHSEWMA(lambda_=lambda_) for lambda_ in grid.fhs_lambdas)
    methods.extend(RegimeHS(regime_bins=bins) for bins in grid.regime_bins)
    methods.append(GARCHStudentT())
    methods.extend(
        EuclideanKNN(k=k, state_columns=state_columns, max_k=analogue_max_k) for k in grid.euclidean_k
    )
    methods.extend(
        RandomKNN(k=k, state_columns=state_columns, max_k=analogue_max_k) for k in grid.euclidean_k
    )
    methods.extend(
        MahalanobisKNN(k=k, state_columns=state_columns, max_k=analogue_max_k)
        for k in grid.euclidean_k
    )
    for window_length in grid.analogue_window_lengths:
        methods.extend(
            WindowEuclideanKNN(k=k, window_length=window_length, max_k=analogue_max_k)
            for k in grid.euclidean_k
        )
        methods.extend(
            DTWWindow(k=k, window_length=window_length, max_k=analogue_max_k)
            for k in grid.euclidean_k
        )
        methods.extend(
            FPCAWindow(k=k, window_length=window_length, max_k=analogue_max_k)
            for k in grid.euclidean_k
        )
    return methods
