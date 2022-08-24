import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util.util_func import *
from util.model_func import diff_drive

class DatasetParser:
    def __init__(self, raw_dataset_path, export_dataset_path, steady_state_step_len, wheel_radius, baseline, rate):
        self.dataframe = pd.read_pickle(raw_dataset_path)
        self.steady_state_step_len = steady_state_step_len
        self.wheel_radius = wheel_radius
        self.baseline = baseline
        self.k = np.array([self.wheel_radius, self.baseline])
        self.rate = rate

    def extract_values_from_dataset(self):
        run = self.dataframe

        self.timestamp = run['ros_time'].to_numpy()
        self.timestamp = (self.timestamp - self.timestamp[0]) * 10 ** (-9)  # time (s)

        self.icp_id = run['icp_index'].to_numpy()
        self.joy = run['joy_switch'].to_numpy()
        self.joy = self.joy == 'True'
        self.good_calib_step = run['good_calib_step'].to_numpy()
        self.good_calib_step = self.good_calib_step == 'True'

        self.icp_x = run['icp_pos_x'].to_numpy()  # icp x position (m)
        self.icp_y = run['icp_pos_y'].to_numpy()  # icp y position (m)
        self.icp_z = run['icp_pos_z'].to_numpy()  # icp y position (m)

        self.icp_quat_x = run['icp_quat_x'].to_numpy()
        self.icp_quat_y = run['icp_quat_y'].to_numpy()
        self.icp_quat_z = run['icp_quat_z'].to_numpy()
        self.icp_quat_w = run['icp_quat_w'].to_numpy()

        self.wheel_pos_left = run['wheel_pos_left'].to_numpy()
        self.wheel_pos_right = run['wheel_pos_right'].to_numpy()

        self.cmd_vx = run['cmd_vel_x'].to_numpy()
        self.cmd_omega = run['cmd_vel_omega'].to_numpy()

        self.n_points = self.timestamp.shape[0]

        self.icp_roll = np.zeros(self.n_points)
        self.icp_pitch = np.zeros(self.n_points)
        self.icp_yaw = np.zeros(self.n_points)

        for i in range(0, self.n_points):
            self.icp_roll[i], self.icp_pitch[i], self.icp_yaw[i] = quaternion_to_euler(self.icp_quat_w[i],
                                                                                       self.icp_quat_x[i],
                                                                                       self.icp_quat_y[i],
                                                                                       self.icp_quat_z[i])  # icp orientation (rad)

        self.icp_id_arr = run['icp_index'].to_numpy()

        self.imu_pitch = run['imu_y'].to_numpy()
        self.imu_roll = run['imu_x'].to_numpy()
        self.imu_yaw = run['imu_z'].to_numpy()
        self.imu_euler = np.column_stack((self.imu_roll, self.imu_pitch, self.imu_yaw))

        self.icp_quat = np.column_stack((self.icp_quat_x, self.icp_quat_y,
                                    self.icp_quat_z, self.icp_quat_w))

        self.icp_euler = np.zeros((self.icp_quat.shape[0], 3))

        for i in range(self.icp_quat.shape[0]):
            self.icp_euler[i, 0], self.icp_euler[i, 1], self.icp_euler[i, 2] = quaternion_to_euler(self.icp_quat[i, 3],
                                                                                                   self.icp_quat[i, 0],
                                                                                                   self.icp_quat[i, 1],
                                                                                                   self.icp_quat[i, 2])

        self.icp_states = np.column_stack((self.timestamp, self.icp_id, self.icp_x, self.icp_y, self.icp_z, self.icp_euler))
        self.icp_states_2d = np.column_stack((self.timestamp, self.icp_id, self.icp_x, self.icp_y, self.icp_euler[:, 2]))

    def create_calibration_step_array(self):
        self.calib_step = np.zeros(self.n_points)
        new_command_step = np.zeros(self.n_points)
        prev_cmd_omega = 0
        prev_cmd_vx = 0
        cmd_step_id = 0
        step_time = 0

        for i in range(1, self.n_points):
            if self.cmd_omega[i] != prev_cmd_omega and self.cmd_omega[i] != 0:  # catches all steps except first angular of each linear step
                cmd_step_id += 1
                prev_cmd_omega = self.cmd_omega[i]
                new_command_step[i] = 1
            if self.cmd_omega[i] == 0 and (
                    self.cmd_vx[i] - self.cmd_vx[i - 1]) == 0.5:  # catches the first angular step of each linear step
                cmd_step_id += 1
                prev_cmd_omega = self.cmd_omega[i]
                new_command_step[i] = 1

            self.calib_step[i] = cmd_step_id

    def compute_wheel_vels(self):
        self.wheel_left_vel = np.zeros(self.n_points)
        self.wheel_right_vel = np.zeros(self.n_points)

        for i in range(20, self.n_points):
            dt = self.timestamp[i] - self.timestamp[i - 1]
            if dt < 0.01:
                self.wheel_left_vel[i] = self.wheel_left_vel[i - 1]
                self.wheel_right_vel[i] = self.wheel_right_vel[i - 1]
            else:
                self.wheel_left_vel[i] = (self.wheel_pos_left[i] - self.wheel_pos_left[i - 1]) / dt
                self.wheel_right_vel[i] = (self.wheel_pos_right[i] - self.wheel_pos_right[i - 1]) / dt

        n_points_convolution = 20
        self.wheel_left_vel = np.convolve(self.wheel_left_vel, np.ones((n_points_convolution,)) / n_points_convolution,
                                     mode='same')
        self.wheel_right_vel = np.convolve(self.wheel_right_vel, np.ones((n_points_convolution,)) / n_points_convolution,
                                      mode='same')
        self.wheel_vels = np.vstack((self.wheel_left_vel, self.wheel_right_vel)).T

    def compute_diff_drive_body_vels(self):
        self.diff_drive_vels = np.zeros((self.n_points, 3))

        for i in range(0, self.n_points):
            self.diff_drive_vels[i, :] = diff_drive(self.wheel_vels[i, :], self.k)

    def compute_icp_based_velocity(self):
        self.icp_vx = np.zeros(self.n_points)
        self.imu_omega = np.zeros(self.n_points)

        propa_cos = np.cos(self.icp_states[0, 4])
        propa_sin = np.sin(self.icp_states[0, 4])
        propa_mat = np.array([[propa_cos, -propa_sin, 0.0],
                              [propa_sin, propa_cos, 0.0], [0.0, 0.0, 1.0]])

        self.icp_vels = np.zeros((self.n_points, 3))
        icp_disp = np.zeros((1, 3))

        dt = 0

        for i in range(1, self.n_points - 1):
            dt += self.timestamp[i + 1] - self.timestamp[i]
            if self.icp_id[i + 1] != self.icp_id[i]:
                icp_disp = self.icp_states_2d[i + 1, 2:] - self.icp_states_2d[i, 2:]
                icp_disp[2] = wrap2pi(icp_disp[2])

                #         print(icp_states[i,4])
                propa_cos = np.cos(self.icp_states[i, 4])
                propa_sin = np.sin(self.icp_states[i, 4])
                propa_mat[0, 0] = propa_cos
                propa_mat[0, 1] = -propa_sin
                propa_mat[1, 0] = propa_sin
                propa_mat[1, 1] = propa_cos
                #         print(i)
                #         print(icp_disp)
                icp_disp = propa_mat.T @ icp_disp
                #         print(icp_disp)

                self.icp_vels[i, :] = icp_disp / dt

                dt = 0

            else:
                self.icp_vels[i, :] = self.icp_vels[i - 1, :]

        n_points_convolution = 10
        self.icp_vels[:, 0] = np.convolve(self.icp_vels[:, 0], np.ones((n_points_convolution,)) / n_points_convolution,
                                     mode='same')

    def create_steady_state_mask(self):
        steady_state_mask = np.full(self.n_points, False)

        for i in range(0, self.n_points - 1):
            if self.calib_step[i + 1] != self.calib_step[i]:
                self.steady_state_mask[i - self.steady_state_step_len:i] = True

    def concatenate_into_full_dataframe(self):
        self.parsed_dataset = np.concatenate((self.timestamp.reshape(self.n_points, 1), self.imu_euler,
                                              self.cmd_vx.reshape(self.n_points, 1), self.cmd_omega.reshape(self.n_points, 1),
                                              self.icp_states[:, 2:], self.icp_vels,
                                              self.wheel_left_vel.reshape(self.n_points, 1), self.wheel_right_vel.reshape(self.n_points, 1),
                                              self.diff_drive_vels, self.calib_step.reshape(self.n_points, 1),
                                              self.steady_state_mask.reshape(self.n_points, 1)), axis=1)
        cols = ['timestamp', 'imu_roll_vel', 'imu_pitch_vel', 'imu_yaw_vel', 'cmd_vx', 'cmd_omega',
                'icp_x', 'icp_y', 'icp_z', 'icp_roll', 'icp_pitch', 'icp_yaw', 'icp_vx', 'icp_vy', 'icp_omega',
                'wheel_left_vel', 'wheel_right_vel', 'diff_drive_vels_x', 'diff_drive_vels_y', 'diff_drive_vels_omega',
                'calib_step', 'steady_state_mask']

        self.parsed_dataset_df = pd.DataFrame(self.parsed_dataset, columns=cols)

    def find_training_horizons(self):
        self.parsed_dataset_steady_state = self.parsed_dataset[steady_state_mask]
        n_points_steady_state = self.parsed_dataset_steady_state.shape[0]

        training_horizon = 2

        self.horizon_starts = []
        self.horizon_ends = []

        for i in range(1, n_points_steady_state):
            if self.parsed_dataset_steady_state[i - 1, 20] != self.parsed_dataset_steady_state[i, 20]:
                self.horizon_starts.append(i)
                horizon_elapsed = 0
                j = i
                if self.parsed_dataset_steady_state[j, 20] == self.parsed_dataset_steady_state[-1, 20]:
                    self.horizon_starts.pop()
                    break
                while self.parsed_dataset_steady_state[j + 1, 20] == self.parsed_dataset_steady_state[j, 20]:
                    horizon_elapsed += (self.parsed_dataset[j + 1, 0] - self.parsed_dataset[j, 0])
                    if horizon_elapsed >= 2.0:
                        self.horizon_ends.append(j)
                        self.horizon_starts.append(j + 1)
                        self.horizon_elapsed = 0
                    j += 1
                self.horizon_starts.pop()

    def build_torch_ready_dataset(self):
        self.rate = 0.05
        timesteps_per_horizon = int(self.training_horizon / self.rate)

        torch_input_array = np.zeros((len(self.horizon_starts),
                                      15 + timesteps_per_horizon * 2))  # [icp_x, icp_y, icp_yaw, vx0, vomega0, vx1, vomega1, vx2, vomega2, vx3, vomega3]
        torch_output_array = np.zeros((len(self.horizon_starts), 6))  # [icp_x, icp_y, icp_yaw]

        for i in range(0, len(self.horizon_starts)):
            horizon_start = self.horizon_starts[i]
            horizon_end = self.horizon_ends[i]
            torch_input_array[i, :6] = self.parsed_dataset_steady_state[horizon_start, 6:12]  # init_state
            torch_input_array[i, 6] = self.parsed_dataset_steady_state[horizon_start, 20]  # calib_step
            torch_input_array[i, 7] = self.parsed_dataset_steady_state[horizon_start, 4]  # cmd_vx
            torch_input_array[i, 8] = self.parsed_dataset_steady_state[horizon_start, 5]  # cmd_omega
            if torch_input_array[i, 8] <= 0:  # and torch_input_array[i, 8] <= 0:
                torch_input_array[i, 9] = 0
            else:
                torch_input_array[i, 9] = 1

            torch_input_array[i, 10] = np.mean(self.parsed_dataset_steady_state[horizon_start:horizon_end, 17])  # encoder_vx
            torch_input_array[i, 11] = np.mean(
                self.parsed_dataset_steady_state[horizon_start:horizon_end, 19])  # encoder_omega
            torch_input_array[i, 12] = np.mean(self.parsed_dataset_steady_state[horizon_start:horizon_end, 12])  # icp_vx
            torch_input_array[i, 13] = np.mean(self.parsed_dataset_steady_state[horizon_start:horizon_end, 13])  # icp_vy
            torch_input_array[i, 14] = np.mean(self.parsed_dataset_steady_state[horizon_start:horizon_end, 14])  # icp_omega

            for j in range(0, timesteps_per_horizon):
                torch_input_array[i, 15 + j * 2] = self.parsed_dataset_steady_state[horizon_start + j, 15]
                torch_input_array[i, 15 + j * 2 + 1] = self.parsed_dataset_steady_state[horizon_start + j, 16]
            torch_output_array[i, :] = self.parsed_dataset_steady_state[horizon_end, 6:12]

        torch_array = np.concatenate((torch_input_array, torch_output_array), axis=1)

        cols = ['init_icp_x', 'init_icp_y', 'init_icp_z', 'init_icp_roll', 'init_icp_pitch', 'init_icp_yaw']
        cols.append('calib_step')
        cols.append('cmd_vx')
        cols.append('cmd_omega')
        cols.append('mask')
        cols.append('encoder_vx')
        cols.append('encoder_omega')
        cols.append('icp_vx')
        cols.append('icp_vy')
        cols.append('icp_omega')
        for i in range(0, timesteps_per_horizon):
            str_cmd_vx_i = 'cmd_left_wheel_' + str(i)
            str_cmd_omega_i = 'cmd_right_wheel_' + str(i)
            cols.append(str_cmd_vx_i)
            cols.append(str_cmd_omega_i)
        cols.append('gt_icp_x')
        cols.append('gt_icp_y')
        cols.append('gt_icp_z')
        cols.append('gt_icp_roll')
        cols.append('gt_icp_pitch')
        cols.append('gt_icp_yaw')

        self.torch_dataset_df = pd.DataFrame(torch_array, columns=cols)

    def process_data(self):
        self.extract_values_from_dataset()
        self.create_calibration_step_array()
        self.compute_wheel_vels()
        self.compute_diff_drive_body_vels()
        self.compute_icp_based_velocity()
        self.create_steady_state_mask()
        self.concatenate_into_full_dataframe()
        self.find_training_horizons()
        self.build_torch_ready_dataset()

        self.torch_dataset_df.to_pickle(self.export_dataset_path)