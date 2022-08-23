import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from model_training import compute_prediction_error

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
# from torchmin import minimize

from models.kinematic.ICR_based import *
from models.kinematic.Perturbed_unicycle import *
from models.kinematic.enhanced_kinematic import *
from models.kinematic.unicycle import Unicycle
from util.util_func import *

from eval.torch_dataset import TorchWMRDataset

#import models
dt = 0.05
r = 0.33/2
# r = 0.5/2
baseline = 0.55
body_inertia = 0.8336
body_mass = 70

eval_dataset_path = '/home/dominic/repos/norlab_WMRD/data/husky/parsed_data/grass_2_torch_wheel_encoder.csv'
training_horizon = 2 # seconds
timestep = 0.05 # seconds
timesteps_per_horizon = int(training_horizon / timestep)

body_cmd_dataset_path = '/home/dominic/repos/norlab_WMRD/data/husky/parsed_data/grass_2_torch_body_cmd.csv'
body_encoder_dataset_path = '/home/dominic/repos/norlab_WMRD/data/husky/parsed_data/grass_2_torch_body_encoder.csv'

wmr_eval_dataset = TorchWMRDataset(eval_dataset_path, body_or_wheel_vel='wheel', training_horizon=training_horizon)
wmr_eval_dl = DataLoader(wmr_eval_dataset)

wmr_body_cmd_dataset = TorchWMRDataset(body_cmd_dataset_path, body_or_wheel_vel='body', training_horizon=training_horizon)
wmr_body_encoder_dataset = TorchWMRDataset(body_encoder_dataset_path, body_or_wheel_vel='body', training_horizon=training_horizon)

icr_symmetrical_trained_params = np.load('training_results/husky/icr_symmetrical/doughnut_grass_1.npy')
icr_asymmetrical_trained_params = np.load('training_results/husky/icr_asymmetrical/doughnut_grass_1.npy')
ehnanced_kinematic_trained_params = np.load('training_results/husky/enhanced_kinematic/doughnut_grass_1.npy')

icr_symmetrical = ICR_symmetrical(r, icr_symmetrical_trained_params[0], icr_symmetrical_trained_params[1], dt)
icr_symmetrical.adjust_motion_params(icr_symmetrical_trained_params)
unicycle = Unicycle(dt)
icr_asymmetrical = ICR_asymmetrical(r, icr_asymmetrical_trained_params[0], icr_asymmetrical_trained_params[1],
                                    icr_asymmetrical_trained_params[2], icr_asymmetrical_trained_params[3],
                                    icr_asymmetrical_trained_params[4], dt)
enhanced_kinematic = Enhanced_kinematic(r, baseline, body_inertia, body_mass, ehnanced_kinematic_trained_params, dt)

prediction_weights = np.eye(6)
prediction_weights_2d = np.zeros((6,6))
prediction_weights_2d[0,0] = prediction_weights_2d[1,1] = prediction_weights_2d[5,5] = 1

calib_step_array = np.zeros((wmr_eval_dataset.__len__(), 1))
prediction_errors_array = np.zeros((wmr_eval_dataset.__len__(), 1))

def project_3d_to_2d(full_state):
    return np.array([full_state[0], full_state[1], full_state[5]])

def compute_evaluation(model, unicycle_model, eval_dataset, body_cmd_dataset, body_encoder_dataset, timesteps_per_horizon, prediction_weights):
    n_points = eval_dataset.__len__()
    prediction_errors_array = np.zeros((n_points, 6))
    model_disp_array_wf = np.zeros((n_points, 6))
    body_cmd_disp_array_wf = np.zeros((n_points, 6))
    body_encoder_disp_array_wf = np.zeros((n_points, 6))
    icp_disp_array_wf = np.zeros((n_points, 6))

    model_vel_array_bf = np.zeros((n_points, 3))
    body_cmd_vel_array_bf = np.zeros((n_points, 3))
    body_encoder_vel_array_bf = np.zeros((n_points, 3))
    icp_vel_array_bf = np.zeros((n_points, 3))

    prediction_transl_error = np.zeros(n_points)
    model_transl_disp = np.zeros(n_points)
    body_cmd_transl_disp = np.zeros(n_points)
    body_encoder_transl_disp = np.zeros(n_points)
    icp_transl_disp = np.zeros(n_points)

    for i in range(0, n_points):
        input, target, step, mask, cmd_x, cmd_omega = eval_dataset.__getitem__(i)
        body_cmd_input, body_cmd_target, body_cmd_stp = body_cmd_dataset.__getitem__(i)
        body_encoder_input, body_encoder_target, body_encoder_stp = body_encoder_dataset.__getitem__(i)
        input = input.numpy()
        target = target.numpy()
        step = step.numpy()
        body_cmd_input = body_cmd_input.numpy()
        body_encoder_input = body_encoder_input.numpy()
        predicted_state_model = input[:6]
        predicted_state_body_cmd = input[:6]
        predicted_state_body_encoder = input[:6]
        init_state = input[:6]

        for j in range(0, timesteps_per_horizon):
            input_id = 6 + j * 2
            predicted_state_model = model.predict(predicted_state_model, input[input_id:input_id + 2])
            predicted_state_body_cmd = unicycle_model.predict(predicted_state_body_cmd, body_cmd_input[input_id:input_id + 2])
            predicted_state_body_encoder = unicycle_model.predict(predicted_state_body_encoder, body_encoder_input[input_id:input_id + 2])

        prediction_errors_array[i, :] = target - predicted_state_model
        prediction_errors_array[i, 5] = wrap2pi(prediction_errors_array[i, 5])
        body_cmd_disp_array_wf[i, :] = predicted_state_body_cmd - init_state
        body_cmd_disp_array_wf[i, 5] = wrap2pi(body_cmd_disp_array_wf[i, 5])
        body_encoder_disp_array_wf[i, :] = predicted_state_body_encoder - init_state
        body_encoder_disp_array_wf[i, 5] = wrap2pi(body_encoder_disp_array_wf[i, 5])
        icp_disp_array_wf[i, :] = target - init_state
        icp_disp_array_wf[i, 5] = wrap2pi(icp_disp_array_wf[i, 5])
        model_disp_array_wf[i, :] = predicted_state_model - init_state
        model_disp_array_wf[i, 5] = wrap2pi(model_disp_array_wf[i, 5])

        prediction_transl_error[i] = trans_disp(prediction_errors_array[i, :])
        model_transl_disp[i] = trans_disp(model_disp_array_wf[i, :])
        body_cmd_transl_disp[i] = trans_disp(body_cmd_disp_array_wf[i, :])
        body_encoder_transl_disp[i] = trans_disp(body_encoder_disp_array_wf[i, :])
        icp_transl_disp[i] = trans_disp(icp_disp_array_wf[i, :])

        body_cmd_vel_array_bf[i, 0] = body_cmd_input[6]
        body_cmd_vel_array_bf[i, 2] = body_cmd_input[7]
        body_encoder_vel_array_bf[i, 0] = body_encoder_input[6]
        body_encoder_vel_array_bf[i, 2] = body_encoder_input[7]

        # TODO: Figure out wrap2pi for angular displacement (need to iterate on it) / transform all displacements in body frame...

    return prediction_errors_array, model_disp_array_wf, body_cmd_disp_array_wf, body_encoder_disp_array_wf, icp_disp_array_wf, \
           model_transl_disp, body_cmd_transl_disp, body_encoder_transl_disp, icp_transl_disp, prediction_transl_error, \
           body_cmd_vel_array_bf, body_encoder_vel_array_bf


prediction_errors_array_icr_s, model_disp_array_wf_icr_s, body_cmd_disp_array_wf_icr_s, body_encoder_disp_array_wf_icr_s, icp_disp_array_wf_icr_s, \
model_transl_disp_icr_s, body_cmd_transl_disp_icr_s, body_encoder_transl_disp_icr_s, icp_transl_disp_icr_s, prediction_transl_error_icr_s, \
body_cmd_vel_array_bf, body_encoder_vel_array_bf \
    = compute_evaluation(icr_symmetrical, unicycle, wmr_eval_dataset, wmr_body_cmd_dataset, wmr_body_encoder_dataset,
                         timesteps_per_horizon, prediction_weights)

prediction_errors_array_icr_as, model_disp_array_wf_icr_as, body_cmd_disp_array_wf_icr_as, body_encoder_disp_array_wf_icr_as, icp_disp_array_wf_icr_as, \
model_transl_disp_icr_as, body_cmd_transl_disp_icr_as, body_encoder_transl_disp_icr_as, icp_transl_disp_icr_as, prediction_transl_error_icr_as, \
body_cmd_vel_array_bf, body_encoder_vel_array_bf \
    = compute_evaluation(icr_asymmetrical, unicycle, wmr_eval_dataset, wmr_body_cmd_dataset, wmr_body_encoder_dataset,
                         timesteps_per_horizon, prediction_weights)

prediction_errors_array_enh_kin, model_disp_array_wf_enh_kin, body_cmd_disp_array_wf_enh_kin, body_encoder_disp_array_wf_enh_kin, icp_disp_array_wf_enh_kin, \
model_transl_disp_enh_kin, body_cmd_transl_disp_enh_kin, body_encoder_transl_disp_enh_kin, icp_transl_disp_enh_kin, prediction_transl_error_enh_kin, \
body_cmd_vel_array_bf, body_encoder_vel_array_bf \
    = compute_evaluation(enhanced_kinematic, unicycle, wmr_eval_dataset, wmr_body_cmd_dataset, wmr_body_encoder_dataset,
                         timesteps_per_horizon, prediction_weights)

# plt.plot(body_cmd_disp_array_bf[:, 0], label='unicycle')
# plt.plot(body_encoder_disp_array_bf[:, 0], label='encoder_DD')
# plt.plot(model_disp_array_bf[:, 0], label='model')
# plt.plot(icp_disp_array_bf[:, 0], label = 'icp')
# # plt.plot(prediction_errors_array[:, 5])
# plt.legend()

print('icr symmetrical transl err : ', np.sum(np.abs(prediction_transl_error_icr_s)))
print('icr symmetrical ang err : ', np.sum(np.abs(prediction_errors_array_icr_s[:, 5])))
print('icr asymmetrical transl err : ', np.sum(np.abs(prediction_transl_error_icr_as)))
print('icr asymmetrical ang err : ', np.sum(np.abs(prediction_errors_array_icr_as[:, 5])))
print('enhanced kinematic transl err : ', np.sum(np.abs(prediction_transl_error_enh_kin)))
print('enhanced kinematic ang err : ', np.sum(np.abs(prediction_errors_array_enh_kin[:, 5])))

plt.figure(figsize=(10,10))
ax = plt.gca()
fig = plt.gcf()

alpha_plot = 0.1

cmd = ax.scatter(body_cmd_vel_array_bf[:, 2], body_cmd_vel_array_bf[:, 0],
                  c = 'tab:blue',
                  cmap = 'hot',
                  alpha = alpha_plot,
                  lw=0,
                  s=50,
                  label='Body vel commands',
                 rasterized=True)

# encoder = ax.scatter(body_encoder_disp_array_wf[:, 5], body_encoder_transl_disp,
#                   c = 'tab:orange',
#                   cmap = 'hot',
#                   alpha = alpha_plot,
#                   lw=0,
#                   s=50,
#                   label='Body vel encoders',
#                  rasterized=True)
#
# icp = ax.scatter(icp_disp_array_wf[:, 5], icp_transl_disp,
#                   c = 'tab:green',
#                   cmap = 'hot',
#                   alpha = alpha_plot,
#                   lw=0,
#                   s=50,
#                   label='ICP',
#                  rasterized=True)
#
# model = ax.scatter(model_disp_array_wf[:, 5], model_transl_disp,
#                   c = 'tab:red',
#                   cmap = 'hot',
#                   alpha = alpha_plot,
#                   lw=0,
#                   s=50,
#                   label='Model',
#                  rasterized=True)

ax.set_xlabel('Angular displacement [rad]')
ax.set_ylabel('Translational displacement [m]')
ax.legend()

plt.show()