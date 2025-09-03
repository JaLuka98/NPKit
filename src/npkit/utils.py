from __future__ import annotations
import numpy as np


def make_rng(seed: int | None) -> np.random.Generator:
    """Create a PCG64-based Generator, or numpy default if seed is None."""
    return np.random.default_rng(None if seed is None else np.random.PCG64(seed))
