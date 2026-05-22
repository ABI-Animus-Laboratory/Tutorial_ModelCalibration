'''
Created on 22/05/2026

@summary: Single-objective calibration of a logistic growth CellML model.

         Ground-truth trajectory is generated with known (r, K), then
         Gaussian noise is added.  scipy.optimize.minimize (BFGS) recovers
         r and K from the noisy observations.

@author: Finbar Argus
'''

from models.CellMLModel import LogisticGrowthCellML

import numpy as np
from scipy.optimize import minimize

if __name__ == '__main__':

    #    Simulation settings
    y0 = 10.0        # initial population
    T  = 20.0        # total time
    dt = 0.25        # output interval

    #    Ground truth model parameters
    r_gt = 0.5
    K_gt = 1000.0

    #    Measurement noise
    sigma = 20.0

    np.random.seed(42)    # fix seed for reproducibility

    #    Generate synthetic measurements
    model = LogisticGrowthCellML()
    t, y, y_measurement = model.generate_measurements(
        sigma=sigma, y0=y0, r=r_gt, K=K_gt, T=T, dt=dt)

    #    Objective: mean squared error between prediction and measurements
    def F(theta, args):
        r, K = theta
        y0, T, dt, model, y_measurement = args
        _, y_pred = model.predict(y0=y0, r=r, K=K, T=T, dt=dt)
        return np.square(np.subtract(y_pred, y_measurement)).mean()

    #    Optimise with Nelder-Mead (gradient-free, robust for correlated params)
    theta_0 = [0.3, 800.0]    # initial guess

    res = minimize(
        F,
        theta_0,
        args=[y0, T, dt, model, y_measurement],
        method='Nelder-Mead',
        options={'disp': True, 'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 5000},
    )

    print('\n--- Calibration result ---')
    print(f'  r  : ground truth={r_gt:.4f},  estimated={res.x[0]:.4f}')
    print(f'  K  : ground truth={K_gt:.4f},  estimated={res.x[1]:.4f}')
    print(res)
