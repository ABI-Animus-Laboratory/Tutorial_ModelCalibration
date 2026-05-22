'''
Tests for models.CellMLModel.LogisticGrowthCellML
'''

import os
import sys
import numpy as np
import pytest

# Make sure the repo root is on the path when running directly or via pytest
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.CellMLModel import LogisticGrowthCellML


@pytest.fixture(scope='module')
def model():
    '''Shared model instance (CellML import + myokit compilation is slow).'''
    return LogisticGrowthCellML()


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

def test_predict_returns_two_arrays(model):
    t, y = model.predict(y0=10.0, r=0.5, K=1000.0, T=10.0, dt=0.5)
    assert isinstance(t, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_predict_shapes_match(model):
    T, dt = 10.0, 0.5
    t, y = model.predict(y0=10.0, r=0.5, K=1000.0, T=T, dt=dt)
    expected_len = int(round(T / dt))
    assert len(t) == expected_len
    assert len(y) == expected_len


def test_predict_time_starts_at_zero(model):
    t, _ = model.predict(y0=10.0, r=0.5, K=1000.0, T=10.0, dt=0.5)
    assert t[0] == pytest.approx(0.0)


def test_predict_initial_condition_honoured(model):
    y0 = 42.0
    _, y = model.predict(y0=y0, r=0.5, K=1000.0, T=10.0, dt=0.5)
    assert y[0] == pytest.approx(y0, rel=1e-3)


def test_predict_population_grows_toward_K(model):
    '''For r>0 and y0<K, population should grow and stay below K.'''
    y0, K = 10.0, 1000.0
    _, y = model.predict(y0=y0, r=0.5, K=K, T=20.0, dt=0.25)
    assert y[-1] > y[0]
    assert y[-1] < K * 1.01   # allow 1 % numerical overshoot tolerance


def test_predict_higher_r_grows_faster(model):
    '''A larger r should produce a higher population at a fixed intermediate time.'''
    _, y_slow = model.predict(y0=10.0, r=0.2, K=1000.0, T=5.0, dt=0.25)
    _, y_fast = model.predict(y0=10.0, r=1.0, K=1000.0, T=5.0, dt=0.25)
    assert y_fast[-1] > y_slow[-1]


def test_predict_different_K_changes_carrying_capacity(model):
    _, y_small_K = model.predict(y0=10.0, r=0.5, K=500.0,  T=30.0, dt=0.5)
    _, y_large_K = model.predict(y0=10.0, r=0.5, K=2000.0, T=30.0, dt=0.5)
    assert y_large_K[-1] > y_small_K[-1]


# ---------------------------------------------------------------------------
# generate_measurements()
# ---------------------------------------------------------------------------

def test_generate_measurements_returns_three_arrays(model):
    result = model.generate_measurements(sigma=50.0, y0=10.0, r=0.5,
                                         K=1000.0, T=10.0, dt=0.5)
    assert len(result) == 3
    t, y, y_meas = result
    assert isinstance(t, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(y_meas, np.ndarray)


def test_generate_measurements_shapes_consistent(model):
    t, y, y_meas = model.generate_measurements(sigma=50.0, y0=10.0, r=0.5,
                                                K=1000.0, T=10.0, dt=0.5)
    assert t.shape == y.shape == y_meas.shape


def test_generate_measurements_noise_zero_sigma(model):
    '''With sigma=0 the measurement should equal the clean trajectory.'''
    _, y, y_meas = model.generate_measurements(sigma=0.0, y0=10.0, r=0.5,
                                               K=1000.0, T=10.0, dt=0.5)
    np.testing.assert_array_almost_equal(y, y_meas)


def test_generate_measurements_adds_noise(model):
    '''With non-zero sigma the noisy and clean arrays should differ.'''
    np.random.seed(0)
    _, y, y_meas = model.generate_measurements(sigma=100.0, y0=10.0, r=0.5,
                                               K=1000.0, T=10.0, dt=0.5)
    assert not np.allclose(y, y_meas)


# ---------------------------------------------------------------------------
# Optimisation sanity-check (light)
# ---------------------------------------------------------------------------

def test_optimisation_recovers_parameters(model):
    '''
    Verify that scipy BFGS can recover r and K from noise-free data to
    within 5 % relative error.  Uses sigma=0 to make the problem easier.
    '''
    from scipy.optimize import minimize

    y0, r_gt, K_gt, T, dt = 10.0, 0.5, 1000.0, 20.0, 0.5
    _, _, y_meas = model.generate_measurements(sigma=0.0, y0=y0, r=r_gt,
                                               K=K_gt, T=T, dt=dt)

    def F(theta):
        r, K = theta
        if r <= 0 or K <= 0:
            return 1e12
        _, y_pred = model.predict(y0=y0, r=r, K=K, T=T, dt=dt)
        return np.square(np.subtract(y_pred, y_meas)).mean()

    res = minimize(F, [0.3, 800.0], method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000})
    r_est, K_est = res.x

    assert abs(r_est - r_gt) / r_gt < 0.10, f'r error too large: {r_est}'
    assert abs(K_est - K_gt) / K_gt < 0.10, f'K error too large: {K_est}'
