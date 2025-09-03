from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, Protocol
import numpy as np

Params = Mapping[str, float]
PredictionFunction = Callable[[Params], float]


class VectorPredictor(Protocol):
    """Protocol for objects that can produce a vector prediction for given params."""

    def predict_vector(self, params: Params) -> np.ndarray: ...


@dataclass(frozen=True)
class Observable:
    """
    A single observable defined by a prediction function.

    The function takes a parameter dict (name -> value) and returns a scalar prediction.
    """

    name: str
    predict: PredictionFunction


@dataclass
class ObservableSet:
    """
    A collection of Observables with a stable order.

    Use `predict_vector(params)` to obtain the prediction vector aligned to `names`.
    """

    observables: Sequence[Observable]

    @property
    def names(self) -> list[str]:
        return [o.name for o in self.observables]

    def predict_vector(self, params: Params) -> np.ndarray:
        """Return prediction vector y_th(params) with shape (n,)."""
        return np.asarray([o.predict(params) for o in self.observables], dtype=float)
