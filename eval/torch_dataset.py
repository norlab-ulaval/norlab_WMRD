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
                                'calib_step', 'cmd_vx', 'cmd_omega', 'encoder_vx', 'encoder_omega',
                                'icp_vx', 'icp_vy', 'icp_omega', 'steady_state_mask', 'calib_mask'], axis=1).values
        output = self.data[['gt_icp_x', 'gt_icp_y', 'gt_icp_z', 'gt_icp_roll', 'gt_icp_pitch', 'gt_icp_yaw']].values
        calib_step = self.data['calib_step']
        cmd_vx = self.data['cmd_vx']
        cmd_omega = self.data['cmd_omega']
        encoder_vx = self.data['encoder_vx']
        encoder_omega = self.data['encoder_omega']
        icp_vx = self.data['icp_vx']
        icp_vy = self.data['icp_vy']
        icp_omega = self.data['icp_omega']
        steady_state_mask = self.data['steady_state_mask'] == 1
        calib_mask = self.data['calib_mask'] == 1


        self.x_train = torch.tensor(input)
        self.y_train = torch.tensor(output)
        self.calib_step = torch.tensor(calib_step)
        self.cmd_vx = torch.tensor(cmd_vx)
        self.cmd_omega = torch.tensor(cmd_omega)
        self.encoder_vx = torch.tensor(encoder_vx)
        self.encoder_omega = torch.tensor(encoder_omega)
        self.icp_vx = torch.tensor(icp_vx)
        self.icp_vy = torch.tensor(icp_vy)
        self.icp_omega = torch.tensor(icp_omega)
        self.steady_state_mask = torch.tensor(steady_state_mask)
        self.calib_mask = torch.tensor(calib_mask)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.calib_step[idx], self.cmd_vx[idx], self.cmd_omega[idx], \
               self.encoder_vx[idx], self.encoder_omega[idx], self.icp_vx[idx], self.icp_vy[idx], self.icp_omega[idx], \
               self.steady_state_mask[idx], self.calib_mask[idx]

    def set_quadran_mask(self, quadran):
        new_calib_mask = np.full(self.__len__(), False)
        if quadran == 1:
            for i in range(0, self.__len__()):
                if self.cmd_vx[i] >= 0 and self.cmd_omega[i] <= 0:
                    new_calib_mask[i] = True
        if quadran == 2:
            for i in range(0, self.__len__()):
                if self.cmd_vx[i] >= 0 and self.cmd_omega[i] >= 0:
                    new_calib_mask[i] = True
        if quadran == 3:
            for i in range(0, self.__len__()):
                if self.cmd_vx[i] <= 0 and self.cmd_omega[i] >= 0:
                    new_calib_mask[i] = True
        if quadran == 4:
            for i in range(0, self.__len__()):
                if self.cmd_vx[i] <= 0 and self.cmd_omega[i] <= 0:
                    new_calib_mask[i] = True
        self.calib_mask = torch.tensor(new_calib_mask)

    def set_area_mask(self, vx_center, omega_center, vx_interval, omega_interval):
        new_calib_mask = np.full(self.__len__(), False)
        for i in range(0, self.__len__()):
            if np.abs(self.cmd_vx[i] - vx_center) <= vx_interval / 2 and np.abs(self.cmd_omega[i] - omega_center) <= omega_interval / 2 :
                new_calib_mask[i] = True
        self.calib_mask = torch.tensor(new_calib_mask)