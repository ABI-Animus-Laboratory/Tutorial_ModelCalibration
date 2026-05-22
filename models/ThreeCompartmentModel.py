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
        importer = myokit.formats.importer('cellml')
        self._model = importer.model(cellml_path)
        self._model.validate()

        self.dt = dt
        self.pre_time = pre_time
        self.sim_time = sim_time
        self.stop_time = pre_time + sim_time
        self.n_steps = int(sim_time / dt)
        self.t_sim = np.linspace(pre_time, self.stop_time, self.n_steps + 1)

        self._sim = myokit.Simulation(self._model)
        self._default_state = self._sim.state()
        self._last_log = None

    def predict(self, param_names, param_vals, output_names, T=None, dt=None):
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
        T : float, optional
            Override sim_time for this call.
        dt : float, optional
            Override dt for this call.

        Returns
        -------
        t       : ndarray, shape (N,)
        outputs : list of ndarray, one per entry in output_names
        '''
        if T is not None or dt is not None:
            sim_time = T if T is not None else self.sim_time
            step = dt if dt is not None else self.dt
            n = int(sim_time / step)
            t_log = np.linspace(self.pre_time, self.pre_time + sim_time, n + 1)
            stop = self.pre_time + sim_time
        else:
            t_log = self.t_sim
            stop = self.stop_time

        for name, value in zip(param_names, param_vals):
            var = self._model.get(name)
            if var.is_constant():
                self._sim.set_constant(name, value)
            elif var.is_state():
                state = list(self._sim.state())
                state[var.index()] = value
                self._sim.set_state(state)

        self._sim.reset()
        self._sim.set_state(self._default_state)

        self._last_log = self._sim.run(
            stop + 1e-9,
            log=output_names,
            log_times=t_log,
        )

        t = t_log
        outputs = [np.array(self._last_log[name]) for name in output_names]
        return t, outputs

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
