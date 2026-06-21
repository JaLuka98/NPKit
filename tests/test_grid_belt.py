import numpy as np
import pytest
from scipy.stats import norm

from npkit import GaussianModel, Observable, ObservableSet
from npkit.neyman import (
    GridBelt,
    _as_parameter_grid,
    _grid_row_to_params,
    build_grid_belt,
    chi2_grid,
    precompute_predictions,
    q_grid_profile,
)


def _make_two_parameter_model() -> GaussianModel:
    obs = ObservableSet(
        [
            Observable("x", lambda p: 10.0 + 2.0 * p["ku"] - 1.0 * p["kd"]),
            Observable("y", lambda p: -5.0 + 0.5 * p["ku"] + 3.0 * p["kd"]),
        ]
    )
    cov = np.array([[4.0, 1.2], [1.2, 3.0]], dtype=float)
    return GaussianModel(obs=obs, covariance=cov)


def test_parameter_grid_helpers_and_grid_chi2():
    model = _make_two_parameter_model()
    params = ("ku", "kd")

    grid = np.array([[-1.0, 0.0], [0.0, 0.5], [1.0, -1.0]], dtype=float)
    grid_arr = _as_parameter_grid(params, grid)
    assert grid_arr.shape == (3, 2)
    assert grid_arr.dtype == float

    one_d = _as_parameter_grid(("C",), np.array([-1.0, 0.0, 1.0], dtype=float))
    assert one_d.shape == (3, 1)

    with pytest.raises(ValueError):
        _as_parameter_grid(("C", "D"), np.array([-1.0, 0.0, 1.0], dtype=float))
    with pytest.raises(ValueError):
        _as_parameter_grid(("C",), np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float))

    row_params = _grid_row_to_params(params, np.array([0.25, -0.75], dtype=float), base={"zs": 1.5})
    assert row_params == {"zs": 1.5, "ku": 0.25, "kd": -0.75}

    predictions = precompute_predictions(model, params, grid_arr)
    vinv = model.inverse_covariance

    y = np.array([9.1, -3.7], dtype=float)
    manual = np.array(
        [float((pred - y) @ vinv @ (pred - y)) for pred in predictions],
        dtype=float,
    )
    chi2_single = chi2_grid(y, predictions, vinv)
    assert chi2_single.shape == (grid_arr.shape[0],)
    assert np.allclose(chi2_single, manual)

    toys = np.array([[9.1, -3.7], [10.5, -4.2], [8.8, -2.9]], dtype=float)
    manual_batch = np.column_stack(
        [
            np.array(
                [float((pred - toy) @ vinv @ (pred - toy)) for pred in predictions],
                dtype=float,
            )
            for toy in toys
        ]
    )
    chi2_batch = chi2_grid(toys, predictions, vinv)
    assert chi2_batch.shape == (grid_arr.shape[0], toys.shape[0])
    assert np.allclose(chi2_batch, manual_batch)

    q_single = q_grid_profile(y, predictions, vinv, test_index=1)
    expected_q_single = max(manual[1] - float(manual.min()), 0.0)
    assert np.isclose(q_single, expected_q_single)

    q_batch = q_grid_profile(toys, predictions, vinv, test_index=1)
    expected_q_batch = np.maximum(manual_batch[1] - manual_batch.min(axis=0), 0.0)
    assert q_batch.shape == (toys.shape[0],)
    assert np.allclose(q_batch, expected_q_batch)


def test_build_grid_belt_matches_manual_quantiles():
    model = _make_two_parameter_model()
    params = ("ku", "kd")
    grid = np.array(
        [
            [-1.0, 0.0],
            [0.0, 0.5],
            [1.0, -1.0],
            [0.5, 1.0],
        ],
        dtype=float,
    )
    alpha = 0.2
    n_toys = 41
    batch_size = 13
    seed = 20240621

    normalized_grid = _as_parameter_grid(params, grid)
    predictions = precompute_predictions(model, params, normalized_grid)
    vinv = model.inverse_covariance
    chol = model._chol

    ref_rng = np.random.default_rng(seed)
    qcrit_ref = np.empty(normalized_grid.shape[0], dtype=float)
    q_values = np.empty(n_toys, dtype=float)

    for test_index, mean in enumerate(predictions):
        offset = 0
        while offset < n_toys:
            n_batch = min(batch_size, n_toys - offset)
            toys = mean + ref_rng.standard_normal(size=(n_batch, predictions.shape[1])) @ chol.T
            q_batch = q_grid_profile(
                toys,
                predictions,
                vinv,
                test_index=test_index,
            )
            q_values[offset : offset + n_batch] = np.asarray(q_batch, dtype=float)
            offset += n_batch
        qcrit_ref[test_index] = float(np.quantile(q_values, 1.0 - alpha))

    belt = build_grid_belt(
        params=params,
        model=model,
        grid=grid,
        n_toys=n_toys,
        alpha=alpha,
        rng=np.random.default_rng(seed),
        batch_size=batch_size,
    )

    assert isinstance(belt, GridBelt)
    assert belt.params == params
    assert belt.grid.shape == normalized_grid.shape
    assert belt.grid.dtype == float
    assert np.allclose(belt.grid, normalized_grid)
    assert belt.qcrit.shape == (normalized_grid.shape[0],)
    assert np.allclose(belt.qcrit, qcrit_ref)
    assert belt.alpha == alpha
    assert belt.confidence_level == 1.0 - alpha


def test_build_grid_belt_accepts_one_dimensional_grid():
    obs = ObservableSet([Observable("x", lambda p: 3.0 + 2.0 * p["C"])])
    model = GaussianModel(obs=obs, covariance=1.0)

    belt = build_grid_belt(
        params=("C",),
        model=model,
        grid=np.array([-1.0, 0.0, 1.0], dtype=float),
        n_toys=5,
        alpha=0.5,
        rng=np.random.default_rng(7),
    )

    assert isinstance(belt, GridBelt)
    assert belt.grid.shape == (3, 1)
    assert belt.params == ("C",)


def _central_confidence_from_sigma(nsigma: float) -> float:
    return float(2.0 * norm.cdf(nsigma) - 1.0)


@pytest.mark.parametrize(
    "nsigma,tol",
    [
        (1.0, 0.06),
        (2.0, 0.12),
    ],
)
def test_grid_belt_quadratic_boundary_matches_chernoff_mixture(
    nsigma: float, tol: float
):
    """
    The new grid-based fast path should reproduce the known Chernoff-mixture
    critical value at the boundary for a quadratic model.
    """
    rng = np.random.default_rng(24680)

    obs = ObservableSet([Observable("x", lambda p: 100.0 + 10.0 * (p["C"] ** 2))])
    model = GaussianModel(obs=obs, covariance=100.0)

    conf = _central_confidence_from_sigma(nsigma)
    alpha = 1.0 - conf
    grid = np.linspace(0.0, 4.0, 41, dtype=float)

    belt = build_grid_belt(
        params=("C",),
        model=model,
        grid=grid,
        n_toys=5_000,
        alpha=alpha,
        rng=rng,
    )

    expected = float(norm.ppf(conf) ** 2)
    q0 = float(belt.qcrit[0])
    assert abs(q0 - expected) <= tol, (
        f"boundary qcrit {q0:.6f} not close to Chernoff-mixture value "
        f"{expected:.6f} for nsigma={nsigma}"
    )
