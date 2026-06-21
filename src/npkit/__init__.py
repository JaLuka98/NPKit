"""
NPKit — Frequentist inference utilities:
- Observables and measurements
- Gaussian model (mean = prediction(params), covariance = V)
- Likelihood & profile-likelihood ratio test statistic
- Neyman belt construction and inversion
"""

from .observables import Observable, ObservableSet, Params
from .measurements import Combination
from .likelihood import GaussianModel, GaussianLikelihood
from .stats import (
    fit_mle,
    profile_curve_from_grid,
    profile_curve_from_likelihood,
    q_profile,
)
from .neyman import (
    Belt,
    GridBelt,
    build_belt,
    build_belts_from_grid,
    build_grid_belt,
    build_profiled_grid_belt,
    check_coverage,
    chi2_grid,
    grid_mask_for_fixed_params,
    invert_belt,
    invert_belt_from_curve,
    precompute_predictions,
    q_grid_profile,
    q_grid_profile_slice,
)

__all__ = [
    "Observable",
    "ObservableSet",
    "Params",
    "Combination",
    "GaussianModel",
    "GaussianLikelihood",
    "fit_mle",
    "profile_curve_from_grid",
    "profile_curve_from_likelihood",
    "q_profile",
    "Belt",
    "GridBelt",
    "build_belt",
    "build_belts_from_grid",
    "build_grid_belt",
    "build_profiled_grid_belt",
    "chi2_grid",
    "grid_mask_for_fixed_params",
    "precompute_predictions",
    "q_grid_profile",
    "q_grid_profile_slice",
    "invert_belt_from_curve",
    "invert_belt",
    "check_coverage",
]

__version__ = "2025.09.0"
