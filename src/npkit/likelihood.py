from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observables import ObservableSet, Params
from .measurements import Combination


def _coerce_cov(cov: object, n: int) -> np.ndarray:
    """
    Accept covariance in several convenient forms and produce a (n,n) matrix:

    - None            -> identity(n)
    - scalar          -> scalar * identity(n)
    - 1D shape (n,)   -> diag(vector)
    - 2D shape (n,n)  -> as-is

    Raises if shape is incompatible or not positive-definite.
    """
    if cov is None:
        M = np.eye(n, dtype=float)
    else:
        arr = np.asarray(cov, dtype=float)
        if arr.ndim == 0:  # scalar
            M = float(arr) * np.eye(n, dtype=float)
        elif arr.ndim == 1:
            if arr.size != n:
                raise ValueError(f"1D covariance length {arr.size} != n={n}")
            M = np.diag(arr)
        elif arr.ndim == 2:
            if arr.shape != (n, n):
                raise ValueError(f"2D covariance shape {arr.shape} != (n,n)={(n, n)}")
            M = arr
        else:
            raise ValueError("covariance must be None, scalar, (n,), or (n,n)")

    # Basic PD check (and symmetrise tiny asymmetries)
    M = 0.5 * (M + M.T)
    try:
        np.linalg.cholesky(M)
    except np.linalg.LinAlgError as e:
        raise ValueError("covariance must be positive-definite") from e
    return M


@dataclass
class GaussianModel:
    """
    Gaussian data model:
    y ~ N(μ(params), V), with μ(params) = obs.predict_vector(params)

    Use this class to:
      - simulate pseudo-data (toys) at given params
      - construct a GaussianLikelihood for observed data
    """

    obs: ObservableSet
    covariance: object  # np.ndarray | float | int | None | (n,) | (n,n)

    def __post_init__(self) -> None:
        n = len(self.obs.observables)
        self.covariance = _coerce_cov(self.covariance, n)
        self._cov_inv = np.linalg.inv(self.covariance)
        # keep logdet around if you want (not used in ratios)
        sign, logdet = np.linalg.slogdet(self.covariance)
        self._logdet = logdet

    def simulate(self, params: Params, rng: np.random.Generator) -> np.ndarray:
        """
        Draw one pseudo-experiment vector y ~ N(μ(params), V).
        """
        mean = self.obs.predict_vector(params)
        return rng.multivariate_normal(mean=mean, cov=self.covariance)

    def likelihood(self, data: Combination) -> "GaussianLikelihood":
        """
        Bind observed data (values must match obs.names order) to a Likelihood.
        """
        if list(data.names) != self.obs.names:
            raise ValueError("data.names must match ObservableSet.names")
        if data.covariance is not None:
            # Allow overriding covariance via data, else use model's V.
            return GaussianLikelihood(self.obs, data.values, data.covariance)
        return GaussianLikelihood(self.obs, data.values, self.covariance)


class GaussianLikelihood:
    """
    -2 log L for Gaussian model with known covariance:
    nll(params) = (y - mu(params))^T V^{-1} (y - mu(params)) + const

    The additive constant is irrelevant for likelihood ratios and is omitted.
    """

    def __init__(
        self, obs: ObservableSet, values: np.ndarray, covariance: object
    ) -> None:
        self.obs = obs
        self.y = np.asarray(values, dtype=float)
        n = self.y.size
        self.V = _coerce_cov(covariance, n)
        self._Vinv = np.linalg.inv(self.V)

    def nll(self, params: Params) -> float:
        r = self.y - self.obs.predict_vector(params)
        return float(r.T @ self._Vinv @ r)
