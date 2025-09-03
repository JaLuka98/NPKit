from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional
import numpy as np


@dataclass
class Combination:
    """
    A set of measured values with (optional) covariance.

    Attributes
    ----------
    names : list[str]
        Order of observables (must match ObservableSet order for likelihood).
    values : np.ndarray, shape (n,)
        Measured values.
    covariance : np.ndarray | None, shape (n, n)
        Total covariance matrix. If None, treat as diagonal with zeros (no errors).
    """

    names: Sequence[str]
    values: np.ndarray
    covariance: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance, dtype=float)
            if self.covariance.shape != (self.values.size, self.values.size):
                raise ValueError("covariance must be (n,n) matching len(values)")
