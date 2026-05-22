'''
Created on 22/05/2026

@summary: Wrapper around a CellML logistic-growth model executed via myokit.

@author: Finbar Argus
'''

import os
import numpy as np
import myokit
import matplotlib.pyplot as plt


class LogisticGrowthCellML:
    '''
    Runs the logistic growth CellML model using myokit.

    Model ODE:
        dy/dt = r * y * (1 - y / K)

    Parameters
    ----------
    cellml_path : str
        Absolute or relative path to the logistic_growth.cellml file.
    '''

    def __init__(self, cellml_path=None):
        if cellml_path is None:
            # Default to cellml_models/logistic_growth.cellml relative to repo root
            here = os.path.dirname(os.path.abspath(__file__))
            cellml_path = os.path.join(here, '..', 'cellml_models', 'logistic_growth.cellml')

        cellml_path = os.path.abspath(cellml_path)
        importer = myokit.formats.importer('cellml')
        self._model = importer.model(cellml_path)
        self._model.validate()

        # Compile the simulation once; reuse across predict() calls
        self._sim = myokit.Simulation(self._model)

    # ------------------------------------------------------------------
    # Public API (mirrors LotkaVolterra interface)
    # ------------------------------------------------------------------

    def predict(self, y0=10.0, r=0.5, K=1000.0, T=20.0, dt=0.25):
        '''
        Simulate the logistic growth model and return (t, y).

        Parameters
        ----------
        y0 : float
            Initial population size.
        r  : float
            Intrinsic growth rate.
        K  : float
            Carrying capacity.
        T  : float
            Total simulation time.
        dt : float
            Time-step between output points.

        Returns
        -------
        t : ndarray, shape (N,)
        y : ndarray, shape (N,)
        '''
        t_eval = np.linspace(0, T, int(np.round(T / dt)))

        self._sim.reset()
        self._sim.set_state([y0])
        self._sim.set_constant('logistic.r', r)
        self._sim.set_constant('logistic.K', K)

        # Run slightly past T so that myokit includes the final log_time point
        d = self._sim.run(T + 1e-9, log=['environment.time', 'logistic.y'],
                          log_times=t_eval)

        return np.array(d['environment.time']), np.array(d['logistic.y'])

    def generate_measurements(self, sigma=50.0, y0=10.0, r=0.5, K=1000.0,
                               T=20.0, dt=0.25):
        '''
        Simulate the model and add Gaussian noise to produce synthetic
        observations.

        Parameters
        ----------
        sigma : float
            Standard deviation of measurement noise.
        y0, r, K, T, dt : see predict()

        Returns
        -------
        t             : ndarray, shape (N,)
        y             : ndarray, shape (N,)  – noise-free trajectory
        y_measurement : ndarray, shape (N,)  – noisy observations
        '''
        t, y = self.predict(y0=y0, r=r, K=K, T=T, dt=dt)
        y_measurement = y + np.random.normal(0.0, sigma, size=y.shape)
        return t, y, y_measurement

    def plot(self, t, y, y_measurement=None):
        '''Plot the simulated trajectory and optional noisy measurements.'''
        plt.figure(figsize=(7, 4))
        plt.plot(t, y, label='Population y(t)')
        if y_measurement is not None:
            plt.scatter(t, y_measurement, color='red', marker='o', s=12,
                        label='Measurements', zorder=3)
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.legend()
        plt.tight_layout()
        plt.show()
