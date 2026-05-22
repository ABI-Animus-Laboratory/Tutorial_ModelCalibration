'''
Created on 22/05/2026

@summary: Random-walk parameter calibration tutorial for the 3-compartment
         cardiovascular model (myokit port of OpenCOR_python_tutorial).

@author: Finbar Argus
'''

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.ThreeCompartmentModel import ThreeCompartmentModel

working_dir = os.path.dirname(os.path.abspath(__file__))

# where you want your outputs to be saved
output_file_path = os.path.join(working_dir, '..', 'outputs')

if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

# time to run the simulation to reach steady state (to reach the time where you want to evaluate outputs)
pre_time = 20

# Amount of simulation that you care about
sim_time = 2

# the output time step
dt = 0.01

# Choose parameter names that you want to vary (myokit qualified names)
param_names = ['parameters_global.q_lv_init', 'parameters.C_venous_svc']

# choose the values of the initial guess for parameter identification
param_vals_current = [1.8e-3, 1.1e-6]
param_vals_mins = [0.5e-3, 1e-6]
param_vals_maxs = [2e-3, 3e-6]
param_vals_history = []

output_names = ['aortic_root_module.u']
ground_truth = 12000

# run param_id until your cost decreases below this value
cost_tolerance = 0.001
# maximum number of iterations
max_iter = 100

# this sets up the simulation object for your cellml model
model = ThreeCompartmentModel(dt=dt, sim_time=sim_time, pre_time=pre_time)


def run_and_get_results(param_vals):
    t, outputs = model.predict(param_names, param_vals, output_names)
    return outputs[0], t


def squared_cost(y, ground_truth):
    # simple squared percentage error cost function
    cost = np.sum(((np.mean(y) - ground_truth) / ground_truth) ** 2)
    return cost

##!!!!! CREATE A NEW COST FUNCTION HERE



##!!!!!


def random_walk_take_step(param_vals, step_weighting=0.1):
    new_param_vals = np.zeros(len(param_vals))
    for II in range(len(param_vals)):
        new_param_vals[II] = (
            param_vals[II]
            + step_weighting * np.random.randn() * (param_vals_maxs[II] - param_vals_mins[II])
        )

        # if param val is outside of the range, set it to the limit
        if new_param_vals[II] > param_vals_maxs[II]:
            new_param_vals[II] = param_vals_maxs[II]
        if new_param_vals[II] < param_vals_mins[II]:
            new_param_vals[II] = param_vals_mins[II]

    return new_param_vals

##!!!!! CREATE A NEW FUNCTION FOR STEPPING THROUGH PARAMETER VALUES HERE



##!!!!!


# intialise cost to something big
cost = 9999
best_cost = cost
iter_idx = 0
while cost > cost_tolerance and iter_idx < max_iter:

    outputs, t = run_and_get_results(param_vals_current)

    # You can make a new cost function and use it here
    cost = squared_cost(outputs, ground_truth)

    if cost < best_cost:
        best_cost = cost
        # save best fit param vals in history list
        param_vals_history.append(param_vals_current)
    else:
        # if the cost was worse (higher) move back to the best param values so far
        param_vals_current = param_vals_history[-1]
        pass

    print(f'Iteration {iter_idx} best cost is :  {best_cost}, with param vals {param_vals_current}')

    # here is where you can implement any scheme for stepping through parameter guesses
    # Currently we implement a very naive random walk that should be improved
    param_vals_current = random_walk_take_step(param_vals_current)

    iter_idx += 1

print(f'Finished after {iter_idx} iterations')
print(f'Best cost is {best_cost}')
print(f'Best parameters are {param_vals_current}')

# Create plots of your best fit compared to the ground truth

outputs, t = run_and_get_results(param_vals_current)
plt.figure(figsize=(8, 4))
plt.plot(t, outputs, label='Model aortic root pressure')
plt.axhline(ground_truth, color='red', linestyle='--', label='Ground truth (mean)')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (Pa)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_file_path, 'calibration_best_fit.png'))
print(f'Saved plot to {output_file_path}/calibration_best_fit.png')
