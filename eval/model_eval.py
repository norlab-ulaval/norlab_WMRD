import numpy as np
import pandas as pd

# from model_training import compute_prediction_error

import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
# from torchmin import minimize

from models.kinematic.ICR_based import *
from models.kinematic.Perturbed_unicycle import *
from util.util_func import *

from eval.torch_dataset import TorchWMRDataset

print('test')

#import models
dt = 0.05
r = 0.33/2
# r = 0.5/2
baseline = 0.55

eval_dataset_path = '/home/dominic/repos/norlab_WMRD/data/husky/parsed_data/grass_2_torch_wheel_encoder.csv'
training_horizon = 2 # seconds
timestep = 0.05 # seconds
timesteps_per_horizon = int(training_horizon / timestep)

wmr_eval_dataset = TorchWMRDataset(eval_dataset_path, body_or_wheel_vel='wheel', training_horizon=training_horizon)
wmr_eval_dl = DataLoader(wmr_eval_dataset)

trained_params = np.load('training_results/husky/doughnut_grass_1.npy')

icr_symmetrical = ICR_symmetrical(r, trained_params[0], trained_params[1], dt)

prediction_weights = np.eye(6)
prediction_weights_2d = np.zeros((6,6))
prediction_weights_2d[0,0] = prediction_weights_2d[1,1] = prediction_weights_2d[5,5] = 1

def compute_prediction_error(init_params, model, dataloader, timesteps_per_horizon, prediction_weights):
    model.adjust_motion_params(init_params)
    print('alpha : ', init_params[0],  ' y_icr : ', init_params[1])
    prediction_error = 0
    for i, (inputs, targets) in enumerate(dataloader):
        # print(inputs)
        # print(targets)
        predicted_state = inputs[0, :6].numpy()
        for j in range(0, timesteps_per_horizon):
            input_id = 6+j*2
            predicted_state = model.predict(predicted_state, inputs[0, input_id:input_id+2].numpy())
        horizon_error = disp_err(predicted_state.reshape((6,1)), targets.numpy().reshape((6,1)), prediction_weights)
        # print(horizon_error)
        prediction_error += horizon_error
    print('total error : ', prediction_error , '[m]')
    return prediction_error

test = compute_prediction_error(model=icr_symmetrical, init_params=trained_params, dataloader=wmr_eval_dl,
                                timesteps_per_horizon=timesteps_per_horizon, prediction_weights=prediction_weights_2d)