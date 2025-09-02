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
from .stats import fit_mle, q_profile
from .neyman import Belt, build_belt, invert_belt, check_coverage

__all__ = [
    "Observable",
    "ObservableSet",
    "Params",
    "Combination",
    "GaussianModel",
    "GaussianLikelihood",
    "fit_mle",
    "q_profile",
    "Belt",
    "build_belt",
    "invert_belt",
    "check_coverage",
]

__version__ = "2025.09.0"
