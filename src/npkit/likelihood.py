from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union, cast
import numpy as np
from numpy.typing import NDArray

from .observables import ObservableSet, Params
from .measurements import Combination

# Flexible input types the user may pass for a covariance
CovInput = Union[
    None, float, int, np.ndarray, Sequence[float], Sequence[Sequence[float]]
]


def _coerce_cov(cov: object, n: int) -> NDArray[np.float64]:
    """
    Accept covariance in several convenient forms and produce a (n,n) float64 matrix:

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
    except np.linalg.LinAlgError as e:  # pragma: no cover
        raise ValueError("covariance must be positive-definite") from e
    return cast(NDArray[np.float64], M.astype(float, copy=False))


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
    covariance: CovInput  # flexible on input; coerced to ndarray in __post_init__
    _cov: NDArray[np.float64] | None = None  # internal, set in __post_init__

    def __post_init__(self) -> None:
        n = len(self.obs.observables)
        self._cov = _coerce_cov(self.covariance, n)
        # Precompute inverse and logdet (kept for potential future use)
        self._cov_inv: NDArray[np.float64] = cast(
            NDArray[np.float64], np.linalg.inv(self._cov)
        )
        sign, logdet = np.linalg.slogdet(self._cov)
        if sign <= 0:
            raise ValueError("covariance must be positive-definite")
        self._logdet = float(logdet)

    def simulate(self, params: Params, rng: np.random.Generator) -> NDArray[np.float64]:
        """
        Draw one pseudo-experiment vector y ~ N(μ(params), V).
        """
        mean = self.obs.predict_vector(params)
        assert self._cov is not None  # for type-checkers
        y = rng.multivariate_normal(mean=mean, cov=self._cov)
        return cast(NDArray[np.float64], np.asarray(y, dtype=float))

    def likelihood(self, data: Combination) -> "GaussianLikelihood":
        """
        Bind observed data (values must match obs.names order) to a Likelihood.
        """
        if list(data.names) != self.obs.names:
            raise ValueError("data.names must match ObservableSet.names")
        if data.covariance is not None:
            # Allow overriding covariance via data, else use model's V.
            return GaussianLikelihood(self.obs, data.values, data.covariance)
        assert self._cov is not None
        return GaussianLikelihood(self.obs, data.values, self._cov)


class GaussianLikelihood:
    """
    -2 log L for Gaussian model with known covariance:
    nll(params) = (y - mu(params))^T V^{-1} (y - mu(params)) + const

    The additive constant is irrelevant for likelihood ratios and is omitted.
    """

    def __init__(
        self, obs: ObservableSet, values: np.ndarray, covariance: CovInput
    ) -> None:
        self.obs = obs
        self.y: NDArray[np.float64] = cast(
            NDArray[np.float64], np.asarray(values, dtype=float)
        )
        n = int(self.y.size)
        self.V: NDArray[np.float64] = _coerce_cov(covariance, n)
        self._Vinv: NDArray[np.float64] = cast(
            NDArray[np.float64], np.linalg.inv(self.V)
        )

    def nll(self, params: Params) -> float:
        r = self.y - self.obs.predict_vector(params)
        return float(r.T @ self._Vinv @ r)
