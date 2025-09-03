import numpy as np
from npkit import (
    Observable,
    ObservableSet,
    Combination,
    GaussianModel,
    GaussianLikelihood,
)
from npkit import build_belt, invert_belt

# Model: two observables
obs = ObservableSet(
    [
        Observable("xsec1", lambda p: 1.0 + 0.4 * p["C"]),
        Observable("xsec2", lambda p: 0.8 - 0.2 * p["C"]),
    ]
)

V = np.array([[0.04**2, 0.5 * 0.04 * 0.05], [0.5 * 0.04 * 0.05, 0.05**2]])

model = GaussianModel(obs=obs, covariance=V)

# Observed data (example)
y_obs = np.array([1.05, 0.78])
data = Combination(names=obs.names, values=y_obs, covariance=V)


def like_builder():
    return GaussianLikelihood(obs, y_obs, V)


start = {"C": 0.0}
bounds = {"C": (-2.0, 2.0)}
rng = np.random.default_rng(123)

# Build a 68% belt
grid = np.linspace(-2, 2, 161)
belt = build_belt(
    param="C",
    model=model,
    like_builder=like_builder,
    grid=grid,
    n_toys=1000,
    alpha=1 - 0.6827,
    rng=rng,
    start=start,
    bounds=bounds,
)

# Invert for observed interval
interval = invert_belt(belt, like_builder=like_builder, start=start, bounds=bounds)
print("Neyman CI:", interval)
