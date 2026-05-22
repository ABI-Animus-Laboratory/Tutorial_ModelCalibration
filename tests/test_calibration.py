'''
Tests for models/ThreeCompartmentModel.py (ThreeCompartmentModel).
'''

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.ThreeCompartmentModel import ThreeCompartmentModel

PARAM_NAMES  = ['parameters_global.q_lv_init', 'parameters.C_venous_svc']
PARAM_VALS   = [1.8e-3, 1.1e-6]
OUTPUT_NAMES = ['aortic_root_module.u']


@pytest.fixture(scope='module')
def model():
    '''Shared model instance (CellML compile is slow).'''
    return ThreeCompartmentModel(dt=0.05, sim_time=1.0, pre_time=5.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_model_loads():
    m = ThreeCompartmentModel(dt=0.05, sim_time=1.0, pre_time=5.0)
    assert m.dt == 0.05
    assert m.sim_time == 1.0
    assert m.pre_time == 5.0


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

def test_predict_returns_t_and_outputs(model):
    t, outputs = model.predict(PARAM_NAMES, PARAM_VALS, OUTPUT_NAMES)
    assert isinstance(t, np.ndarray)
    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], np.ndarray)


def test_predict_shapes_consistent(model):
    t, outputs = model.predict(PARAM_NAMES, PARAM_VALS, OUTPUT_NAMES)
    assert len(t) == model.n_steps + 1
    assert len(outputs[0]) == len(t)


def test_predict_time_starts_at_pre_time(model):
    t, _ = model.predict(PARAM_NAMES, PARAM_VALS, OUTPUT_NAMES)
    assert t[0] == pytest.approx(model.pre_time)


def test_predict_params_change_output(model):
    _, o1 = model.predict(PARAM_NAMES, [1.8e-3, 1.1e-6], OUTPUT_NAMES)
    _, o2 = model.predict(PARAM_NAMES, [1.0e-3, 2.0e-6], OUTPUT_NAMES)
    assert not np.allclose(np.mean(o1[0]), np.mean(o2[0]))


# ---------------------------------------------------------------------------
# generate_measurements()
# ---------------------------------------------------------------------------

def test_generate_measurements_shapes(model):
    t, outputs, measurements = model.generate_measurements(
        PARAM_NAMES, PARAM_VALS, OUTPUT_NAMES
    )
    assert t.shape == outputs[0].shape == measurements[0].shape


def test_generate_measurements_zero_sigma(model):
    _, outputs, measurements = model.generate_measurements(
        PARAM_NAMES, PARAM_VALS, OUTPUT_NAMES, sigma=0.0
    )
    np.testing.assert_array_almost_equal(outputs[0], measurements[0])


def test_generate_measurements_adds_noise(model):
    np.random.seed(0)
    _, outputs, measurements = model.generate_measurements(
        PARAM_NAMES, PARAM_VALS, OUTPUT_NAMES, sigma=500.0
    )
    assert not np.allclose(outputs[0], measurements[0])


# ---------------------------------------------------------------------------
# Calibration loop
# ---------------------------------------------------------------------------

def test_calibration_loop_improves_cost(model):
    '''A short random-walk loop should reduce the cost from its starting value.'''
    np.random.seed(0)

    param_vals_current = [1.8e-3, 1.1e-6]
    param_vals_mins = [0.5e-3, 1e-6]
    param_vals_maxs = [2e-3, 3e-6]
    param_vals_history = []
    ground_truth = 12000.0
    max_iter = 15

    def squared_cost(y):
        return np.sum(((np.mean(y) - ground_truth) / ground_truth) ** 2)

    def random_walk_take_step(param_vals, step_weighting=0.1):
        new_vals = np.zeros(len(param_vals))
        for ii in range(len(param_vals)):
            new_vals[ii] = (
                param_vals[ii]
                + step_weighting * np.random.randn() * (param_vals_maxs[ii] - param_vals_mins[ii])
            )
            new_vals[ii] = np.clip(new_vals[ii], param_vals_mins[ii], param_vals_maxs[ii])
        return new_vals

    _, o0 = model.predict(PARAM_NAMES, param_vals_current, OUTPUT_NAMES)
    best_cost = squared_cost(o0[0])
    initial_cost = best_cost

    for _ in range(max_iter):
        _, y = model.predict(PARAM_NAMES, param_vals_current, OUTPUT_NAMES)
        cost = squared_cost(y[0])
        if cost < best_cost:
            best_cost = cost
            param_vals_history.append(list(param_vals_current))
        elif param_vals_history:
            param_vals_current = list(param_vals_history[-1])
        param_vals_current = random_walk_take_step(param_vals_current)

    assert best_cost < initial_cost
