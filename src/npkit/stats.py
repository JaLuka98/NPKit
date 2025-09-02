from __future__ import annotations

from typing import Mapping, Tuple, Optional

import numpy as np

try:
    from scipy.optimize import minimize
except Exception as _exc:  # pragma: no cover
    minimize = None  # type: ignore[assignment]

from .likelihood import GaussianLikelihood
from .observables import Params


def _require_scipy() -> None:
    if minimize is None:
        raise RuntimeError(
            "SciPy is required for optimisation (scipy.optimize.minimize). "
            "Install with `pip install scipy`."
        )


def _params_to_vector(param_names: list[str], params: Params) -> np.ndarray:
    return np.asarray([params[name] for name in param_names], dtype=float)


def _vector_to_params(param_names: list[str], x: np.ndarray) -> dict[str, float]:
    return {name: float(val) for name, val in zip(param_names, x)}


def fit_mle(
    like: GaussianLikelihood,
    start: Params,
    bounds: Optional[dict[str, Tuple[float, float]]] = None,
) -> tuple[dict[str, float], float]:
    """
    Minimise nll(params) to obtain MLE and nll_min.

    Parameters
    ----------
    like : GaussianLikelihood
    start : dict[str, float]
        Starting values for all free parameters.
    bounds : dict[str, (low, high)] | None
        Optional box constraints.

    Returns
    -------
    (best_params, nll_min)
    """
    _require_scipy()

    names = list(start.keys())
    x0 = _params_to_vector(names, start)
    opt_bounds = None
    if bounds:
        opt_bounds = [bounds.get(n, (-np.inf, np.inf)) for n in names]

    def fun(x: np.ndarray) -> float:
        return like.nll(_vector_to_params(names, x))

    res = minimize(fun, x0, bounds=opt_bounds, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"MLE optimisation failed: {res.message}")
    best = _vector_to_params(names, res.x)
    return best, float(res.fun)


def q_profile(
    param: str,
    value: float,
    like_builder: Callable[[], GaussianLikelihood],
    start: Params,
    bounds: Optional[dict[str, Tuple[float, float]]] = None,
) -> float:
    """
    Profile-likelihood ratio test statistic for one parameter:

        q(value) = nll(params_hat_hat(value)) - nll(params_hat)

    where params_hat_hat(value) fixes `param` = value and minimises over the others.
    """
    # Unconstrained fit
    like = like_builder()
    _, nll_min = fit_mle(like, start=start, bounds=bounds)

    # Names of nuisance/free parameters (everything except `param`)
    fixed_names = [n for n in start.keys() if n != param]

    # If there are no nuisance parameters, just evaluate NLL at the fixed value
    if not fixed_names:
        like = like_builder()
        nll_fixed = like.nll({**start, param: float(value)})
        q = float(nll_fixed - nll_min)
        return max(0.0, q)

    # Otherwise, do the constrained optimisation over nuisance parameters
    like = like_builder()
    fixed_start = {n: start[n] for n in fixed_names}

    def constrained_nll(x: np.ndarray) -> float:
        p = {**{param: float(value)}, **{n: float(v) for n, v in zip(fixed_names, x)}}
        return like.nll(p)

    _require_scipy()
    x0 = np.asarray([fixed_start[n] for n in fixed_names], dtype=float)
    opt_bounds = None
    if bounds:
        opts = [bounds.get(n, (-np.inf, np.inf)) for n in fixed_names]
        # SciPy doesn't like [] for 0-dim problems; keep None in that case
        if len(opts) > 0:
            opt_bounds = opts

    res = minimize(constrained_nll, x0, bounds=opt_bounds, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"Constrained optimisation failed: {res.message}")

    q = float(res.fun - nll_min)
    return max(0.0, q)
