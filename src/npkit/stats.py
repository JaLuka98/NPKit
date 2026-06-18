from __future__ import annotations

from typing import Tuple, Optional, Callable

import numpy as np

from scipy.optimize import minimize, minimize_scalar

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

    For 1 parameter, use a scalar line search (robust if the gradient is zero at start).
    For >=2 parameters, use L-BFGS-B with a Powell fallback.
    """
    names = list(start.keys())

    # --- 1D robust path -----------------------------------------------------
    if len(names) == 1:
        pname = names[0]

        def f1(x: float) -> float:
            return like.nll({pname: float(x)})

        if bounds and pname in bounds:
            lo, hi = bounds[pname]
            res = minimize_scalar(f1, bounds=(lo, hi), method="bounded")
        else:
            # Auto-bracket: expand until downhill is found
            a, b = 0.0, 1.0
            fa, fb = f1(a), f1(b)
            k = 0
            while fb >= fa and k < 12:
                b *= 2.0
                fb = f1(b)
                k += 1
            # If we never found downhill (pathological), still proceed
            res = minimize_scalar(f1, bracket=(a, b))

        if not res.success:
            raise RuntimeError(f"MLE 1D optimisation failed: {res.message}")
        return {pname: float(res.x)}, float(res.fun)

    # --- >=2D path (as before) ---------------------------------------------
    names = list(start.keys())
    x0 = np.asarray([start[n] for n in names], dtype=float)

    opt_bounds = None
    if bounds:
        opt_bounds = [bounds.get(n, (-np.inf, np.inf)) for n in names]

    def fun(x: np.ndarray) -> float:
        return like.nll({n: float(v) for n, v in zip(names, x)})

    res = minimize(fun, x0, bounds=opt_bounds, method="L-BFGS-B")
    # Fallback if stuck near start (can happen on flat/ill-conditioned surfaces)
    if (not res.success) or np.allclose(res.x, x0):
        res = minimize(fun, x0, bounds=opt_bounds, method="Powell")

    if not res.success:
        raise RuntimeError(f"MLE optimisation failed: {res.message}")
    best = {n: float(v) for n, v in zip(names, res.x)}
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


def profile_curve_from_likelihood(
    param: str,
    grid: np.ndarray,
    like: GaussianLikelihood,
    start: Params,
) -> tuple[float, float, np.ndarray]:
    """
    Evaluate the NLL on a 1D parameter grid without any minimization.

    This is the no-nuisance scan path:
      - compute NLL at every grid point,
      - infer the best fit from the minimum of that curve,
      - shift the curve so q_min = 0.
    """
    grid_arr = np.asarray(grid, dtype=float)
    curve = np.asarray(
        [like.nll({**start, param: float(value)}) for value in grid_arr],
        dtype=float,
    )
    best_idx = int(np.argmin(curve))
    nll_min = float(curve[best_idx])
    q_curve = np.maximum(0.0, curve - nll_min)
    return float(grid_arr[best_idx]), nll_min, q_curve


def profile_curve_from_grid(
    param: str,
    grid: np.ndarray,
    like_builder: Callable[[], GaussianLikelihood],
    start: Params,
) -> tuple[float, float, np.ndarray]:
    """
    Convenience wrapper around `profile_curve_from_likelihood`.

    A fresh likelihood is built once, then scanned on the supplied grid.
    """
    return profile_curve_from_likelihood(
        param=param,
        grid=grid,
        like=like_builder(),
        start=start,
    )
