from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .observables import ObservableSet, Params

_DEFAULT_CHUNK_SIZE = 100_000


@dataclass(frozen=True)
class GaussianGridCache1D:
    """Whitened 1D scan cache for a fixed covariance Gaussian model."""

    param: str
    grid: NDArray[np.float64]
    means_w: NDArray[np.float64]
    mu_norm2: NDArray[np.float64]
    chol: NDArray[np.float64]

    def curve_from_yw(self, y_w: np.ndarray) -> NDArray[np.float64]:
        """
        Return the whitened chi2 curve without the toy-dependent constant ||y_w||^2.

        For 1D inputs the result has shape (m,); for batched inputs it has shape
        (n_toys, m), where m is the scan-grid length.
        """
        y_w_arr = np.asarray(y_w, dtype=float)
        if y_w_arr.ndim == 1:
            return cast(
                NDArray[np.float64], -2.0 * (self.means_w @ y_w_arr) + self.mu_norm2
            )
        if y_w_arr.ndim != 2:
            raise ValueError("y_w must be one-dimensional or two-dimensional")
        curve = -2.0 * (y_w_arr @ self.means_w.T) + self.mu_norm2[None, :]
        return cast(NDArray[np.float64], np.asarray(curve, dtype=float))

    def whiten(self, values: np.ndarray) -> NDArray[np.float64]:
        """Transform values to whitened coordinates using the cached Cholesky factor."""
        values_arr = np.asarray(values, dtype=float)
        return cast(NDArray[np.float64], np.linalg.solve(self.chol, values_arr))


def prepare_gaussian_grid_cache_1d(
    *,
    obs: ObservableSet,
    covariance: np.ndarray,
    param: str,
    grid: np.ndarray,
    start: Params,
) -> GaussianGridCache1D:
    grid_arr = np.asarray(grid, dtype=float)
    if grid_arr.ndim != 1:
        raise ValueError("grid must be one-dimensional")

    means = np.asarray(
        [obs.predict_vector({**start, param: float(c)}) for c in grid_arr],
        dtype=float,
    )
    if means.ndim == 1:
        means = means[:, None]

    chol = np.linalg.cholesky(np.asarray(covariance, dtype=float))
    means_w = np.linalg.solve(chol, means.T).T
    mu_norm2 = np.einsum("ij,ij->i", means_w, means_w)

    return GaussianGridCache1D(
        param=param,
        grid=grid_arr,
        means_w=cast(NDArray[np.float64], np.asarray(means_w, dtype=float)),
        mu_norm2=cast(NDArray[np.float64], np.asarray(mu_norm2, dtype=float)),
        chol=cast(NDArray[np.float64], np.asarray(chol, dtype=float)),
    )


def fill_q_samples_for_true_index(
    cache: GaussianGridCache1D,
    true_index: int,
    n_toys: int,
    rng: np.random.Generator,
    out: NDArray[np.float64],
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> NDArray[np.float64]:
    """
    Fill `out` with q-values for toys generated at a single true grid point.

    The computation is chunked to keep temporary allocations bounded while still
    using a batched matrix multiply on the whitened scan curve.
    """
    if out.shape != (n_toys,):
        raise ValueError("out must have shape (n_toys,)")

    n_obs = int(cache.means_w.shape[1])
    mu_true_w = cache.means_w[true_index]

    offset = 0
    while offset < n_toys:
        n_chunk = min(chunk_size, n_toys - offset)
        y_w = rng.standard_normal(size=(n_chunk, n_obs))
        y_w += mu_true_w[None, :]
        curve = cache.curve_from_yw(y_w)
        out[offset : offset + n_chunk] = curve[:, true_index] - np.min(curve, axis=1)
        offset += n_chunk

    return out
