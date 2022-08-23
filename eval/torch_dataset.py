import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TorchWMRDataset(Dataset):
    def __init__(self, csv_file, body_or_wheel_vel, training_horizon=2, steady_state_window = 160, rate=20):
        self.data = pd.read_pickle(csv_file)
        self.training_horizon = training_horizon
        self.steady_state_window = steady_state_window
        self.rate = 20
        self.timestep = 1/self.rate
        self.steady_state_length = self.steady_state_window / self.timestep
        self.horizons_per_step = np.floor(self.steady_state_length / self.training_horizon)

        input = self.data.drop(['gt_icp_x', 'gt_icp_y', 'gt_icp_z', 'gt_icp_roll', 'gt_icp_pitch', 'gt_icp_yaw',
                                'calib_step', 'cmd_vx', 'cmd_omega', 'mask'], axis=1).values
        output = self.data[['gt_icp_x', 'gt_icp_y', 'gt_icp_z', 'gt_icp_roll', 'gt_icp_pitch', 'gt_icp_yaw']].values
        calib_step = self.data['calib_step']
        mask = self.data['mask'] == 1
        cmd_vx = self.data['cmd_vx']
        cmd_omega = self.data['cmd_omega']

        self.x_train = torch.tensor(input)
        self.y_train = torch.tensor(output)
        self.calib_step = torch.tensor(calib_step)
        self.mask = torch.tensor(mask)
        self.cmd_vx = torch.tensor(cmd_vx)
        self.cmd_omega = torch.tensor(cmd_omega)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.calib_step[idx], self.mask[idx], self.cmd_vx[idx], self.cmd_omega[idx]