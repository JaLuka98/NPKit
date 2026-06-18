from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import numpy as np

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
