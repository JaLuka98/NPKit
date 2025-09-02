from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .observables import Params
from .likelihood import GaussianModel, GaussianLikelihood
from .stats import q_profile


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
      1) Generate y_toy ~ N(μ(C), V)
      2) Compute q(value=C) on y_toy via profile-likelihood
      3) Take the (1 - alpha)-quantile over toys as qcrit[C]
    """
    m = grid.size
    qcrit = np.empty(m, dtype=float)

    for i, c in enumerate(grid):
        q_vals = np.empty(n_toys, dtype=float)
        for t in range(n_toys):
            y = model.simulate({**start, param: float(c)}, rng=rng)

            def fresh_like() -> GaussianLikelihood:
                # Bind a fresh likelihood to this toy data
                return GaussianLikelihood(model.obs, y, model.covariance)

            q_vals[t] = q_profile(
                param=param,
                value=float(c),
                like_builder=fresh_like,
                start=start,
                bounds=bounds,
            )
        qcrit[i] = float(np.quantile(q_vals, 1.0 - alpha))

    return Belt(
        param=param, grid=np.asarray(grid, dtype=float), qcrit=qcrit, alpha=alpha
    )


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
