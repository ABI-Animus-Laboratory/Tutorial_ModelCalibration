'''
Created on 22/05/2026

@summary: Myokit wrapper for the flat 3-compartment cardiovascular CellML model.

@author: Finbar Argus
'''

import os
import numpy as np
import myokit


class ThreeCompartmentModel:
    '''
    Runs the flat 3-compartment CellML cardiovascular model with myokit.

    Parameters are referenced by their myokit qualified names, e.g.:
        parameters_global.q_lv_init
        parameters.C_venous_svc
        aortic_root_module.u
    '''

    def __init__(self, cellml_path=None, dt=0.01, sim_time=2.0, pre_time=20.0):
        if cellml_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            cellml_path = os.path.join(here, '..', 'cellml_models', '3compartment.cellml')

        cellml_path = os.path.abspath(cellml_path)
        self._model = myokit.formats.importer('cellml').model(cellml_path)
        self._model.validate()

        self.dt = dt
        self.pre_time = pre_time
        self.sim_time = sim_time
        self.stop_time = pre_time + sim_time
        self.n_steps = int(sim_time / dt)
        self.t_sim = np.linspace(pre_time, self.stop_time, self.n_steps + 1)

        self._sim = myokit.Simulation(self._model)

    def predict(self, param_names, param_vals, output_names):
        '''
        Set parameters, run the simulation, and return (t, outputs).

        Parameters
        ----------
        param_names : list of str
            Myokit qualified names of constants to set, e.g.
            ['parameters_global.q_lv_init', 'parameters.C_venous_svc']
        param_vals : list of float
            Values corresponding to param_names.
        output_names : list of str
            Myokit qualified names of variables to return, e.g.
            ['aortic_root_module.u']

        Returns
        -------
        t       : ndarray, shape (N,)
        outputs : list of ndarray, one per entry in output_names
        '''
        self._sim.reset()
        for name, value in zip(param_names, param_vals):
            self._sim.set_constant(name, value)

        log = self._sim.run(
            self.stop_time + 1e-9,
            log=output_names,
            log_times=self.t_sim,
        )

        outputs = [np.array(log[name]) for name in output_names]
        return self.t_sim, outputs

    def generate_measurements(self, param_names, param_vals, output_names, sigma=500.0):
        '''
        Simulate the model and add Gaussian noise to each output.

        Returns
        -------
        t            : ndarray
        outputs      : list of ndarray  – noise-free trajectories
        measurements : list of ndarray  – noisy observations
        '''
        t, outputs = self.predict(param_names, param_vals, output_names)
        measurements = [y + np.random.normal(0.0, sigma, size=y.shape) for y in outputs]
        return t, outputs, measurements
