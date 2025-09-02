import numpy as np
from npkit import (
    Observable,
    ObservableSet,
    Combination,
    GaussianModel,
    GaussianLikelihood,
)
from npkit import build_belt, invert_belt, q_profile
import matplotlib.pyplot as plt
from npkit.plot import plot_belt

# Model: two observables
obs = ObservableSet(
    [
        Observable("xsec1", lambda p: 100 + 10 * p["C"] ** 2),
    ]
)

V = np.array([100])

model = GaussianModel(obs=obs, covariance=V)

# Observed data (example)
y_obs = np.array([100])
# data = Combination(names=obs.names, values=y_obs, covariance=V)


def like_builder():
    return GaussianLikelihood(obs, y_obs, V)


start = {"C": 0.0}
bounds = {"C": (-1e6, 1e6)}
rng = np.random.default_rng(123)

# Build a 68% belt
grid = np.linspace(-2, 2, 101)
belt = build_belt(
    param="C",
    model=model,
    like_builder=like_builder,
    grid=grid,
    n_toys=1000,
    alpha=1 - 0.95,
    rng=rng,
    start=start,
    bounds=bounds,
)

print(belt)

# Invert for observed interval
interval = invert_belt(belt, like_builder=like_builder, start=start, bounds=bounds)
print("Neyman CI:", interval)

# Visualize the Neyman belt
ax = plot_belt(belt)
ax.set_title("Neyman Belt Visualization")
plt.show()
