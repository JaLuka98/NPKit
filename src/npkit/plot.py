from __future__ import annotations

from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from .neyman import Belt


def plot_belt(
    belt: Belt, q_obs: np.ndarray | None = None, ax: Any | None = None
) -> Any:
    """
    Plot q_crit(C) across the grid; optionally overlay q_obs(C).
    """
    if ax is None:
        fig, ax = plt.subplots()
        _ = fig  # silence linters if unused
    ax.step(
        belt.grid, belt.qcrit, where="mid", label=f"q_crit (alpha={belt.alpha:.3f})"
    )
    if q_obs is not None:
        ax.step(belt.grid, q_obs, where="mid", linestyle="--", label="q_obs")
    ax.set_xlabel(belt.param)
    ax.set_ylabel("q")
    ax.legend()
    return ax
