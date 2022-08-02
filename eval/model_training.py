import math
from pypointmatcher import pointmatcher, pointmatchersupport
import glob
import numpy as np
import copy
import pandas as pd
import wmrde

from models.kinematic.ICR_based import *
from models.kinematic.Perturbed_unicycle import *
from util.util_func import *

from scipy.optimize import minimize

dt = 0.05
r = 0.175
baseline = 1.296
alpha = 0.8
alpha_l = 0.8
alpha_r = 0.8
y_icr = 3.5
y_icr_l = 3.5
y_icr_r = -3.5
x_icr = -2.5
alpha_params = np.full((13), 1.0)

icr_symmetrical = ICR_symmetrical(r, alpha, y_icr, dt)
icr_assymetrical = ICR_assymetrical(r, alpha_l, alpha_r, x_icr, y_icr_l, y_icr_r, dt)
perturbed_unicycle = Perturbed_unicycle(r, baseline, alpha_params, dt)

dynamic = wmrde.Predictor()

dataset_path = '../data/fr2021/states_inputs/teachb_data.csv'

inputs_states = pd.read_csv(dataset_path)

inputs = inputs_states[['cmd_left_vel', 'cmd_right_vel']].to_numpy()
positions = inputs_states[['icp_pos_x', 'icp_pos_y', 'icp_pos_z']].to_numpy()
orientations_quaternion = inputs_states[['icp_quat_x', 'icp_quat_y', 'icp_quat_z', 'icp_quat_w']].to_numpy()
orientations_euler = np.zeros((orientations_quaternion.shape[0], 3))
gravity_vectors = np.zeros((orientations_quaternion.shape[0], 3))
for i in range(0, orientations_quaternion.shape[0]):
    orientations_euler[i, :] = quaternion_to_euler(orientations_quaternion[i, :])
    gravity_vectors[i, :] = euler_to_rotmat(orientations_euler[i, :])[:, 2]
states = np.hstack((positions, orientations_euler))

icp_ids = inputs_states['icp_index'].to_numpy()
times = inputs_states['ros_time'].to_numpy()

dt = 0.05
horizon = 2.0 # seconds
steps_per_prediction = math.floor(horizon / dt)

prediction_weights = np.eye(6)
prediction_weights_2d = np.eye(6)

# TODO: Step to adjust model params here.


def compute_prediction_error(init_params, model):
    print(init_params)
    model.adjust_motion_params(init_params)
    prediction_error = 0
    for i in range(0, inputs.shape[0]):
        predicted_state = states[i, :]
        prediction_end_id = 0
        if i + steps_per_prediction >= inputs.shape[0]:
            break
        for j in range(0, steps_per_prediction):
            predicted_state = model.predict(predicted_state, inputs[i+j, :])
            if icp_ids[j] == icp_ids[i+steps_per_prediction]:
                prediction_end_id = j
                break
            prediction_end_id = j
        prediction_error += disp_err(states[prediction_end_id, :], predicted_state, prediction_weights)
    print(prediction_error)
    return prediction_error
#
init_params = [0.8, 3.5] # for icr
# init_params = np.full((13), 1.0)
args = {'model': icr_symmetrical}
# args = {'model': perturbed_unicycle}
# test = compute_prediction_error(model=perturbed_unicycle, init_params=init_params)
# test = compute_prediction_error(model=icr_symmetrical, init_params=init_params)
minimize(compute_prediction_error, init_params, args=icr_symmetrical)
# minimize(compute_prediction_error, init_params, args=perturbed_unicycle)