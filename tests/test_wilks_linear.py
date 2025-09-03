# tests/test_wilks_linear.py
import numpy as np

from npkit import (
    Observable,
    ObservableSet,
    GaussianModel,
    GaussianLikelihood,
)
from npkit.neyman import build_belt, invert_belt


def test_linear_model_wilks_68pct():
    """
    y sim N(100 + 10*C, sigma^2), sigma=10, identifiable & linear.
    For toys generated at true C over the grid, q_crit(C) must equal the
    chi^2_1 68.27% quantile (=1) for *all* C (up to MC error).
    Also, with y_obs=100 (C_hat=0), the 68% CI should be approx [-1, 1].
    """
    rng = np.random.default_rng(123)

    # One observable, linear in C
    obs = ObservableSet([Observable("x", lambda p: 100.0 + 10.0 * p["C"])])

    # Variance sigma^2 = 100 (sigma = 10)
    V = 100.0
    model = GaussianModel(obs=obs, covariance=V)

    # Observed data exactly at C=0 mean
    y_obs = np.array([100.0])

    def like_builder():
        return GaussianLikelihood(obs, y_obs, V)

    alpha = 1.0 - 0.6827  # 68.27% CL
    grid = np.linspace(-2.0, 2.0, 41)  # step 0.1
    start = {"C": 0.0}
    bounds = None
    n_toys = 2500  # modest but stable for CI

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

    # --- q_crit should be roughly 1 everywhere (within MC noise) ---------------------
    # mean close to 1
    q_mean = float(np.mean(belt.qcrit))
    assert 0.95 <= q_mean <= 1.05, f"mean q_crit {q_mean:.3f} not ~ 1"

    # every point within a reasonable band (occasionally a few % jitter)
    assert np.all((belt.qcrit >= 0.8) & (belt.qcrit <= 1.2)), (
        f"some q_crit outside [0.8,1.2]: {belt.qcrit.min():.3f}..{belt.qcrit.max():.3f}"
    )

    # --- Invert for observed interval: expect ~[-1, 1] ------------------------
    lo, hi = invert_belt(belt, like_builder=like_builder, start=start, bounds=bounds)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert -1.05 <= lo <= -0.95, f"lo {lo:.3f} not near -1"
    assert 0.95 <= hi <= 1.05, f"hi {hi:.3f} not near +1"
