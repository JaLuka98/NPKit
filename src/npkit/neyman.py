from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, cast

import numpy as np
from numpy.typing import NDArray

from ._scan import (
    fill_q_samples_for_true_index,
    prepare_gaussian_grid_cache_1d,
)
from .observables import Params
from .likelihood import GaussianModel, GaussianLikelihood
from .stats import profile_curve_from_grid, q_profile


@dataclass
class Belt:
    """Critical quantiles of the test statistic across a parameter grid."""

    param: str
    grid: np.ndarray  # shape (m,)
    qcrit: np.ndarray  # shape (m,)
    alpha: float  # size (e.g. 0.3173 for 68.27% CI)


@dataclass
class GridBelt:
    """Critical quantiles of the grid-profile statistic for multi-parameter scans."""

    params: tuple[str, ...]
    grid: NDArray[np.float64]  # shape (n_grid, n_params)
    qcrit: NDArray[np.float64]  # shape (n_grid,)
    alpha: float
    confidence_level: float


def _as_parameter_grid(
    params: Sequence[str],
    grid: np.ndarray,
) -> NDArray[np.float64]:
    """Normalize a 1D or 2D parameter grid to shape (n_grid, n_params)."""
    param_names = (params,) if isinstance(params, str) else tuple(params)
    grid_arr = np.asarray(grid, dtype=float)

    if grid_arr.ndim == 1:
        if len(param_names) != 1:
            raise ValueError(
                "a one-dimensional grid can only be used with exactly one parameter"
            )
        grid_arr = grid_arr[:, None]
    elif grid_arr.ndim == 2:
        if len(param_names) != grid_arr.shape[1]:
            raise ValueError(
                "len(params) must match grid.shape[1] for a multi-parameter grid"
            )
    else:
        raise ValueError("grid must be one-dimensional or two-dimensional")

    if grid_arr.shape[0] == 0:
        raise ValueError("grid must contain at least one parameter point")

    return cast(NDArray[np.float64], np.asarray(grid_arr, dtype=float))


def _grid_row_to_params(
    params: Sequence[str],
    row: np.ndarray,
    base: Params | None = None,
) -> Params:
    """Convert one grid row into a parameter dictionary."""
    param_names = (params,) if isinstance(params, str) else tuple(params)
    row_arr = np.asarray(row, dtype=float)
    if row_arr.ndim != 1:
        raise ValueError("row must be one-dimensional")
    if len(param_names) != row_arr.size:
        raise ValueError("row length must match len(params)")

    out = dict(base) if base is not None else {}
    out.update({name: float(value) for name, value in zip(param_names, row_arr)})
    return out


def grid_mask_for_fixed_params(
    params: Sequence[str],
    grid: np.ndarray,
    fixed: Mapping[str, float],
    atol: float = 1e-12,
) -> NDArray[np.bool_]:
    """Return a boolean mask selecting grid rows that match the fixed values."""
    param_names = (params,) if isinstance(params, str) else tuple(params)
    grid_arr = _as_parameter_grid(param_names, grid)

    if not fixed:
        raise ValueError("fixed must contain at least one parameter")

    mask = np.ones(grid_arr.shape[0], dtype=bool)
    for name, value in fixed.items():
        if name not in param_names:
            raise KeyError(f"unknown fixed parameter '{name}'")
        idx = param_names.index(name)
        mask &= np.isclose(grid_arr[:, idx], float(value), atol=atol, rtol=0.0)

    if not mask.any():
        raise ValueError("grid contains no rows matching the fixed parameters")

    return cast(NDArray[np.bool_], mask)


def precompute_predictions(
    model: GaussianModel,
    params: Sequence[str],
    grid: np.ndarray,
    base: Params | None = None,
) -> NDArray[np.float64]:
    """Evaluate the model predictions on every grid point once."""
    param_names = (params,) if isinstance(params, str) else tuple(params)
    grid_arr = _as_parameter_grid(param_names, grid)
    predictions = np.asarray(
        [
            model.obs.predict_vector(_grid_row_to_params(param_names, row, base=base))
            for row in grid_arr
        ],
        dtype=float,
    )
    return cast(NDArray[np.float64], np.asarray(predictions, dtype=float))


def chi2_grid(
    y: np.ndarray,
    predictions: np.ndarray,
    vinv: np.ndarray,
) -> NDArray[np.float64]:
    """Vectorized chi2 on a grid of predictions."""
    y_arr = np.asarray(y, dtype=float)
    predictions_arr = np.asarray(predictions, dtype=float)
    vinv_arr = np.asarray(vinv, dtype=float)

    if predictions_arr.ndim != 2:
        raise ValueError("predictions must have shape (n_grid, n_obs)")
    if vinv_arr.ndim != 2:
        raise ValueError("vinv must be a two-dimensional matrix")
    if vinv_arr.shape != (predictions_arr.shape[1], predictions_arr.shape[1]):
        raise ValueError("vinv shape must match the observable dimension")

    if y_arr.ndim == 1:
        if y_arr.shape[0] != predictions_arr.shape[1]:
            raise ValueError("y must match the observable dimension")
        delta = predictions_arr - y_arr
        chi2 = np.einsum("gi,ij,gj->g", delta, vinv_arr, delta)
        return cast(NDArray[np.float64], np.asarray(chi2, dtype=float))

    if y_arr.ndim == 2:
        if y_arr.shape[1] != predictions_arr.shape[1]:
            raise ValueError("y must have shape (n_toys, n_obs)")
        delta = predictions_arr[:, None, :] - y_arr[None, :, :]
        chi2 = np.einsum("gti,ij,gtj->gt", delta, vinv_arr, delta)
        return cast(NDArray[np.float64], np.asarray(chi2, dtype=float))

    raise ValueError("y must be one-dimensional or two-dimensional")


def q_grid_profile_slice(
    y: np.ndarray,
    predictions: np.ndarray,
    profile_predictions: np.ndarray,
    vinv: np.ndarray,
) -> float | NDArray[np.float64]:
    """Profile statistic using a fixed slice for the tested parameter value."""
    chi2_full = chi2_grid(y=y, predictions=predictions, vinv=vinv)
    chi2_profile = chi2_grid(y=y, predictions=profile_predictions, vinv=vinv)

    if chi2_full.ndim == 1:
        q = float(np.min(chi2_profile) - float(np.min(chi2_full)))
        return float(np.maximum(q, 0.0))

    q = np.min(chi2_profile, axis=0) - np.min(chi2_full, axis=0)
    return cast(NDArray[np.float64], np.maximum(q, 0.0))


def q_grid_profile(
    y: np.ndarray,
    predictions: np.ndarray,
    vinv: np.ndarray,
    test_index: int,
) -> float | NDArray[np.float64]:
    """Grid-profile likelihood-ratio statistic using the discrete best fit."""
    chi2_values = chi2_grid(y=y, predictions=predictions, vinv=vinv)
    if chi2_values.ndim == 1:
        q = float(chi2_values[test_index] - float(np.min(chi2_values)))
        return float(np.maximum(q, 0.0))

    q = chi2_values[test_index, :] - np.min(chi2_values, axis=0)
    return cast(NDArray[np.float64], np.maximum(q, 0.0))


def build_grid_belt(
    params: Sequence[str],
    model: GaussianModel,
    grid: np.ndarray,
    n_toys: int,
    alpha: float,
    rng: np.random.Generator,
    base: Params | None = None,
    batch_size: int | None = None,
) -> GridBelt:
    """Build a Neyman belt for a multi-parameter grid without fitting."""
    param_names = (params,) if isinstance(params, str) else tuple(params)
    if not param_names:
        raise ValueError("params must contain at least one parameter name")
    if n_toys <= 0:
        raise ValueError("n_toys must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")

    grid_arr = _as_parameter_grid(param_names, grid)
    predictions = precompute_predictions(model, param_names, grid_arr, base=base)

    cov = getattr(model, "_cov", None)
    if cov is None:
        cov = model.covariance_matrix
    cov_arr = np.asarray(cov, dtype=float)

    vinv = getattr(model, "_cov_inv", None)
    if vinv is None:
        vinv = model.inverse_covariance
    vinv_arr = np.asarray(vinv, dtype=float)

    chol = getattr(model, "_chol", None)
    if chol is None:
        chol = np.linalg.cholesky(cov_arr)
    chol_arr = np.asarray(chol, dtype=float)

    n_obs = int(predictions.shape[1])
    batch_cap = n_toys if batch_size is None else int(batch_size)
    if batch_cap <= 0:
        raise ValueError("batch_size must be positive when provided")

    q_values = np.empty(n_toys, dtype=float)
    qcrit = np.empty(predictions.shape[0], dtype=float)

    for test_index, mean in enumerate(predictions):
        offset = 0
        while offset < n_toys:
            n_batch = min(batch_cap, n_toys - offset)
            noise = rng.standard_normal(size=(n_batch, n_obs))
            toys = mean + noise @ chol_arr.T
            q_batch = q_grid_profile(
                y=toys,
                predictions=predictions,
                vinv=vinv_arr,
                test_index=test_index,
            )
            q_values[offset : offset + n_batch] = np.asarray(q_batch, dtype=float)
            offset += n_batch

        qcrit[test_index] = float(np.quantile(q_values, 1.0 - alpha))

    return GridBelt(
        params=param_names,
        grid=grid_arr,
        qcrit=qcrit,
        alpha=float(alpha),
        confidence_level=float(1.0 - alpha),
    )


def build_profiled_grid_belt(
    params: Sequence[str],
    poi: str,
    model: GaussianModel,
    grid: np.ndarray,
    n_toys: int,
    alpha: float,
    rng: np.random.Generator,
    start: Params,
    base: Params | None = None,
    batch_size: int | None = None,
    atol: float = 1e-12,
) -> Belt:
    """Build a 1D belt by profiling over all grid parameters except `poi`."""
    param_names = (params,) if isinstance(params, str) else tuple(params)
    if poi not in param_names:
        raise ValueError("poi must be one of params")
    if poi not in start:
        raise ValueError("start must contain the parameter of interest")
    if n_toys <= 0:
        raise ValueError("n_toys must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")

    grid_arr = _as_parameter_grid(param_names, grid)
    predictions = precompute_predictions(model, param_names, grid_arr, base=base)

    cov = getattr(model, "_cov", None)
    if cov is None:
        cov = model.covariance_matrix
    cov_arr = np.asarray(cov, dtype=float)

    vinv = getattr(model, "_cov_inv", None)
    if vinv is None:
        vinv = model.inverse_covariance
    vinv_arr = np.asarray(vinv, dtype=float)

    chol = getattr(model, "_chol", None)
    if chol is None:
        chol = np.linalg.cholesky(cov_arr)
    chol_arr = np.asarray(chol, dtype=float)

    poi_idx = param_names.index(poi)
    poi_values = np.unique(grid_arr[:, poi_idx])
    batch_cap = n_toys if batch_size is None else int(batch_size)
    if batch_cap <= 0:
        raise ValueError("batch_size must be positive when provided")

    n_obs = int(predictions.shape[1])
    q_values = np.empty(n_toys, dtype=float)
    qcrit = np.empty(poi_values.size, dtype=float)

    for i, poi_value in enumerate(poi_values):
        profile_mask = grid_mask_for_fixed_params(
            params=param_names,
            grid=grid_arr,
            fixed={poi: float(poi_value)},
            atol=atol,
        )
        profile_predictions = predictions[profile_mask]
        if profile_predictions.size == 0:
            raise RuntimeError("profile slice unexpectedly empty")

        true_params = dict(base) if base is not None else {}
        true_params.update(start)
        true_params[poi] = float(poi_value)
        mean = np.asarray(model.obs.predict_vector(true_params), dtype=float)

        offset = 0
        while offset < n_toys:
            n_batch = min(batch_cap, n_toys - offset)
            noise = rng.standard_normal(size=(n_batch, n_obs))
            toys = mean + noise @ chol_arr.T
            q_batch = q_grid_profile_slice(
                y=toys,
                predictions=predictions,
                profile_predictions=profile_predictions,
                vinv=vinv_arr,
            )
            q_values[offset : offset + n_batch] = np.asarray(q_batch, dtype=float)
            offset += n_batch

        qcrit[i] = float(np.quantile(q_values, 1.0 - alpha))

    return Belt(param=poi, grid=poi_values, qcrit=qcrit, alpha=float(alpha))


def build_belt(
    param: str,
    model: GaussianModel,
    like_builder: Callable[[], GaussianLikelihood],
    grid: np.ndarray,
    n_toys: int,
    alpha: float,
    rng: np.random.Generator,
    start: Params,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> Belt:
    """
    Build a Neyman belt by generating toys at each true parameter value.

    For each grid point C:
      1) Generate whitened toys z ~ N(0, I) and shift them by the true mean
      2) Compute q(value=C) from the whitened scan curve
      3) Take the (1 - alpha)-quantile over toys as qcrit[C]
    """
    _ = like_builder  # kept for API compatibility; the batched path does not need it.

    cache = prepare_gaussian_grid_cache_1d(
        obs=model.obs,
        covariance=model.covariance_matrix,
        param=param,
        grid=grid,
        start=start,
    )

    q_vals = np.empty(n_toys, dtype=float)
    qcrit = np.empty(cache.grid.size, dtype=float)

    for i in range(cache.grid.size):
        fill_q_samples_for_true_index(
            cache=cache,
            true_index=i,
            n_toys=n_toys,
            rng=rng,
            out=q_vals,
        )
        qcrit[i] = float(np.quantile(q_vals, 1.0 - alpha))

    return Belt(
        param=param, grid=cache.grid, qcrit=qcrit, alpha=alpha
    )


def build_belts_from_grid(
    param: str,
    model: GaussianModel,
    grid: np.ndarray,
    n_toys: int,
    alphas: Sequence[float],
    rng: np.random.Generator,
    start: Params,
) -> tuple[Belt, ...]:
    """
    Build several Neyman belts from the same toy ensemble without fitting.

    For each grid point C:
      1) generate whitened toys at C,
      2) scan the full 1D whitened chi2 curve on the supplied grid,
      3) infer q(C) from the minimum of that curve,
      4) reuse the same q samples to extract multiple critical values.
    """
    alpha_list = [float(alpha) for alpha in alphas]
    if not alpha_list:
        raise ValueError("alphas must contain at least one confidence level.")

    cache = prepare_gaussian_grid_cache_1d(
        obs=model.obs,
        covariance=model.covariance_matrix,
        param=param,
        grid=grid,
        start=start,
    )

    q_samples = np.empty((cache.grid.size, n_toys), dtype=float)

    for i, c in enumerate(cache.grid):
        print(f"Generating toys for {param}={c:.3f} ({i+1}/{cache.grid.size})")
        fill_q_samples_for_true_index(
            cache=cache,
            true_index=i,
            n_toys=n_toys,
            rng=rng,
            out=q_samples[i, :],
        )

    belts = []
    for alpha in alpha_list:
        qcrit = np.quantile(q_samples, 1.0 - alpha, axis=1)
        belts.append(
            Belt(
                param=param,
                grid=cache.grid,
                qcrit=np.asarray(qcrit, dtype=float),
                alpha=alpha,
            )
        )
    return tuple(belts)


def invert_belt_from_curve(belt: Belt, q_obs: np.ndarray) -> tuple[float, float]:
    """
    Invert a belt using a precomputed observed q curve.

    This avoids any likelihood refits and is the right companion to
    `profile_curve_from_grid`.
    """
    q_obs_arr = np.asarray(q_obs, dtype=float)
    if q_obs_arr.shape != belt.grid.shape:
        raise ValueError("q_obs must have the same shape as belt.grid.")

    mask = q_obs_arr <= belt.qcrit
    if not mask.any():
        return (np.nan, np.nan)

    idx = np.where(mask)[0]
    return (float(belt.grid[idx[0]]), float(belt.grid[idx[-1]]))


def invert_belt(
    belt: Belt,
    like_builder: Callable[[], GaussianLikelihood],
    start: Params,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """
    Given the observed data (implicit in like_builder), compute q_obs(C) across the grid
    and return the smallest contiguous interval of C where q_obs(C) <= qcrit(C).
    """
    if len(start) == 1 and belt.param in start:
        _, _, q_obs = profile_curve_from_grid(
            param=belt.param,
            grid=belt.grid,
            like_builder=like_builder,
            start=start,
        )
    else:
        q_obs = np.array(
            [
                q_profile(
                    param=belt.param,
                    value=float(c),
                    like_builder=like_builder,
                    start=start,
                    bounds=bounds,
                )
                for c in belt.grid
            ],
            dtype=float,
        )

    mask = q_obs <= belt.qcrit
    if not mask.any():
        return (np.nan, np.nan)

    # Find the longest (or first) contiguous segment; here we choose the minimal covering segment
    idx = np.where(mask)[0]
    return (float(belt.grid[idx[0]]), float(belt.grid[idx[-1]]))


def check_coverage(
    true_value: float,
    belt: Belt,
    model: GaussianModel,
    start: Params,
    bounds: dict[str, tuple[float, float]] | None,
    n_experiments: int,
    rng: np.random.Generator,
) -> float:
    """
    Empirical coverage at `true_value`:
      fraction of experiments whose CI (from belt inversion) contains true_value.
    """
    hits = 0
    for _ in range(n_experiments):
        y = model.simulate({**start, belt.param: float(true_value)}, rng=rng)

        def fresh_like() -> GaussianLikelihood:
            return GaussianLikelihood(model.obs, y, model.covariance)

        lo, hi = invert_belt(belt, like_builder=fresh_like, start=start, bounds=bounds)
        if lo <= true_value <= hi:
            hits += 1
    return hits / n_experiments
