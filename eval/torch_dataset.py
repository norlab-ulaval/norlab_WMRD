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

        input = self.data.drop(['gt_icp_x', 'gt_icp_y', 'gt_icp_z', 'gt_icp_roll', 'gt_icp_pitch', 'gt_icp_yaw'], axis=1).values
        output = self.data[['gt_icp_x', 'gt_icp_y', 'gt_icp_z', 'gt_icp_roll', 'gt_icp_pitch', 'gt_icp_yaw']].values

        self.x_train = torch.tensor(input)
        self.y_train = torch.tensor(output)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

