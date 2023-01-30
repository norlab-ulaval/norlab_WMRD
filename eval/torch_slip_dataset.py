import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models.kinematic.ideal_diff_drive import Ideal_diff_drive

class TorchWMRSlipDataset(Dataset):
    def __init__(self, pkl_file, gp_state):
        self.data = pd.read_pickle(pkl_file)
        self.rate = 20
        self.timestep = 1/self.rate
        self.gp_state = gp_state

        str_body_vel_disturption_x_list = []
        str_body_vel_disturption_y_list = []
        str_body_vel_disturption_yaw_list = []
        idd_body_vel_x_str_list = []
        idd_body_vel_yaw_str_list = []
        for i in range(0, 40):
            str_body_vel_disturption_x_i = 'body_vel_disturption_x_' + str(i)
            str_body_vel_disturption_x_list.append(str_body_vel_disturption_x_i)
            str_body_vel_disturption_y_i = 'body_vel_disturption_y_' + str(i)
            str_body_vel_disturption_y_list.append(str_body_vel_disturption_y_i)
            str_body_vel_disturption_yaw_i = 'body_vel_disturption_yaw_' + str(i)
            str_body_vel_disturption_yaw_list.append(str_body_vel_disturption_yaw_i)
            str_idd_vel_x_i = 'idd_vel_x_' + str(i)
            idd_body_vel_x_str_list.append(str_idd_vel_x_i)
            str_idd_vel_yaw_i = 'idd_vel_yaw_' + str(i)
            idd_body_vel_yaw_str_list.append(str_idd_vel_yaw_i)
        idd_body_vels_x = self.data[idd_body_vel_x_str_list].to_numpy()
        idd_body_vels_yaw = self.data[idd_body_vel_yaw_str_list].to_numpy()
        idd_body_vels_disturption_x = self.data[str_body_vel_disturption_x_list].to_numpy()
        idd_body_vels_disturption_y = self.data[str_body_vel_disturption_y_list].to_numpy()
        idd_body_vels_disturption_yaw = self.data[str_body_vel_disturption_yaw_list].to_numpy()

        steady_state_mask = self.data['steady_state_mask'] == 1
        transitory_state_mask = self.data['transitory_state_mask'] == 1

        n_windows = idd_body_vels_x.shape[0]
        n_timesteps_per_window = idd_body_vels_disturption_x.shape[1]
        idd_body_accelerations_x = np.zeros((n_windows, n_timesteps_per_window))
        idd_body_accelerations_yaw = np.zeros((n_windows, n_timesteps_per_window))
        for i in range(0, n_windows):
            if transitory_state_mask[i]:
                for j in range(1, n_timesteps_per_window):
                    idd_body_accelerations_x[i, j] = (idd_body_vels_x[i, j] -
                                                    idd_body_vels_x[i, j - 1]) / self.timestep
                    idd_body_accelerations_yaw[i, j] = (idd_body_vels_yaw[i, j] -
                                                      idd_body_vels_yaw[i, j - 1]) / self.timestep

        input = np.column_stack((idd_body_vels_x.flatten(),
                                idd_body_vels_yaw.flatten(),
                                idd_body_accelerations_x.flatten(),
                                idd_body_accelerations_yaw.flatten()))


        if self.gp_state == 'x':
            output = idd_body_vels_disturption_x.flatten()
        if self.gp_state == 'y':
            output = idd_body_vels_disturption_y.flatten()
        if self.gp_state == 'yaw':
            output = idd_body_vels_disturption_yaw.flatten()

        calib_step = self.data['calib_step']
        # calib_mask = self.data['calib_mask'] == 1

        self.x_train = torch.tensor(input)
        self.y_train = torch.tensor(output)
        self.steady_state_mask = torch.tensor(steady_state_mask)
        self.transitory_state_mask = torch.tensor(transitory_state_mask)
        # self.calib_mask = torch.tensor(calib_mask)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.steady_state_mask[idx]