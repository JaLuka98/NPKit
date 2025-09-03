# tests/test_chernoff_boundary.py
import numpy as np
from scipy.stats import norm
import pytest

from npkit import Observable, ObservableSet, GaussianModel, GaussianLikelihood
from npkit.neyman import build_belt


def _q_chernoff_quantile(conf: float) -> float:
    """Quantile of 0.5*delta0 + 0.5*chi2_1 at confidence 'conf' (e.g. 0.6827, 0.9545)."""
    # For mixture: F(q) = Phi(sqrt(q))  ->  sqrt(q) = Phi^{-1}(conf)

    z = float(norm.ppf(conf))
    return z * z


@pytest.mark.parametrize(
    "conf, expected, n_toys, tol",
    [
        (0.6827, _q_chernoff_quantile(0.6827), 20000, 0.08),  # ~0.22
        (0.9545, _q_chernoff_quantile(0.9545), 25000, 0.15),  # ~2.89
    ],
)
def test_boundary_chernoff_quantiles_at_zero(conf, expected, n_toys, tol):
    """
    Model: y ~ N(100 + 10*C^2, sigma^2) with C >= 0, sigma^2 = 100.
    At true C=0, the LR statistic follows 0.5*delta0 + 0.5*chi2_1.
    The Neyman critical value at C=0 must match the mixture quantile.
    """
    rng = np.random.default_rng(12345)

    # Observable with even dependence in C (non-identifiable without boundary)
    obs = ObservableSet([Observable("x", lambda p: 100.0 + 10.0 * (p["C"] ** 2))])

    V = 100.0  # variance sigma^2
    model = GaussianModel(obs=obs, covariance=V)

    # Dummy like_builder; build_belt binds toy data internally
    y_obs = np.array([100.0])

    def like_builder():
        return GaussianLikelihood(obs, y_obs, V)

    alpha = 1.0 - conf
    grid = np.array([0.0])  # only test C = 0 (the boundary point)
    start = {"C": 0.0}
    bounds = {"C": (0.0, 1e9)}  # enforce physical boundary C >= 0

    belt = build_belt(
        param="C",
        model=model,
        like_builder=like_builder,
        grid=grid,
        n_toys=n_toys,
        alpha=alpha,
        rng=rng,
        start=start,
        bounds=bounds,
    )

    q0 = float(belt.qcrit[0])
    assert (
        abs(q0 - expected) <= tol
    ), f"C=0: q_crit={q0:.3f}, expected {expected:.3f} ± {tol}"
