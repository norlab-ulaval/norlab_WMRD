import numpy as np
from util.transform_algebra import *

class Enhanced_kinematic:
    def __init__(self, radius, baseline, body_inertia, body_mass, params, stoch_params, dt):
        self.dt = dt
        self.radius = radius
        self.baesline = baseline
        self.body_mass = body_mass
        self.body_intertia = body_inertia
        self.params = params
        # self.params = np.reshape(params, (13,1))
        # self.block_diag_coeff_matrix = np.zeros((3 ,13))
        # self.block_diag_coeff_matrix[0, :4] = coeff_x
        # self.block_diag_coeff_matrix[1, 4:8] = coeff_y
        # self.block_diag_coeff_matrix[2, 8:12] = coeff_z

        self.diff_drive_jacobian = np.array([[self.radius/2, self.radius/2],
                                             [0, 0],
                                             [-self.radius/self.baesline, self.radius/self.baesline]])

        self.body_force = np.zeros((3,1))
        self.diff_drive_vel = np.zeros((3, 1))
        self.slip_vel = np.zeros((3, 1))
        self.prev_diff_drive_vel = np.zeros((3, 1))
        self.input_acceleration = np.zeros((3, 1))

        self.rotation_body_to_world = np.eye(2)
        self.body_vel_world_3d = np.zeros(6)
        self.body_vel_world_2d = np.zeros(3)
        self.state_2d = np.zeros(3)

        self.state_transition_matrix = np.eye(3)
        # self.prediction_covariance = np.eye(3)
        self.noise_spectral_density = np.array([[stoch_params[0], stoch_params[1], stoch_params[2]],
                                                [stoch_params[1], stoch_params[3], stoch_params[4]],
                                                [stoch_params[2], stoch_params[4], stoch_params[5]]])


    def compute_slip_velocity(self, gravity_vector):
        self.body_force[0] = self.body_mass * gravity_vector[0]
        self.body_force[1] = self.diff_drive_vel[0] * self.diff_drive_vel[2] * (self.body_mass - self.body_intertia)
        self.body_force[2] = self.body_mass * gravity_vector[2]

        self.diff_drive_acceleration = self.diff_drive_vel - self.prev_diff_drive_vel

        # self.slip_vel[0] = self.params[0] * (self.body_force[0] / self.body_force[2]) * self.diff_drive_vel[0] \
        #                    + self.params[1] * self.diff_drive_vel[0]
        self.slip_vel[0] = self.params[0] * self.diff_drive_vel[0] + self.params[1] * self.diff_drive_acceleration[0]
        self.slip_vel[1] = self.params[2] * (self.body_force[1] / self.body_force[2]) * self.diff_drive_vel[0]
        # self.slip_vel[2] = self.params[3] * (self.body_force[1] / self.body_force[2]) * self.diff_drive_vel[0] \
        #                    + self.params[4] * self.diff_drive_vel[0] + self.params[5] * self.diff_drive_vel[2]
        self.slip_vel[2] = self.params[3] * self.diff_drive_vel[0] + self.params[4] * self.diff_drive_vel[2] + \
                           self.params[5] * self.diff_drive_acceleration[2]

    def predict(self, init_state, input):
        """
        :param init_state: initial state array [x, y, z, roll, pitch, yaw]
        :param input: input array [omega_l, omega_r]
        :return: next_state
        """
        self.state_2d[:2] = init_state[:2]
        self.state_2d[2] = init_state[-1]
        yaw_to_rotmat2d(self.rotation_body_to_world, init_state[-1])
        gravity_vector = 9.8 * euler_to_rotmat(init_state[2:])[:, 2:]

        self.diff_drive_vel = np.reshape(self.diff_drive_jacobian @ input, (3,1))

        self.compute_slip_velocity(gravity_vector)

        body_vel = self.diff_drive_vel - self.slip_vel
        self.body_vel_world_2d[:2] = np.reshape(self.rotation_body_to_world @ body_vel[:2], (2))
        self.body_vel_world_2d[2] = body_vel[2]
        self.body_vel_world_3d[:2] = self.body_vel_world_2d[:2]
        self.body_vel_world_3d[-1] = self.body_vel_world_2d[-1]
        self.prev_diff_drive_vel = self.diff_drive_vel
        return init_state + self.body_vel_world_3d * self.dt

    def propagate_uncertainty(self, init_state, input, prediction_covariance):
        self.state_2d[:2] = init_state[:2]
        self.state_2d[2] = init_state[-1]
        yaw_to_rotmat2d(self.rotation_body_to_world, init_state[-1])
        gravity_vector = 9.8 * euler_to_rotmat(init_state[2:])[:, 2:]

        self.diff_drive_vel = np.reshape(self.diff_drive_jacobian @ input, (3, 1))

        self.compute_slip_velocity(gravity_vector)
        body_vel = self.diff_drive_vel - self.slip_vel
        self.body_vel_world_2d[:2] = np.reshape(self.rotation_body_to_world @ body_vel[:2], (2))
        self.body_vel_world_2d[2] = body_vel[2]
        self.body_vel_world_3d[:2] = self.body_vel_world_2d[:2]
        self.body_vel_world_3d[-1] = self.body_vel_world_2d[-1]

        self.state_transition_matrix[0,2] = -self.body_vel_world_2d[1]
        self.state_transition_matrix[1,2] = self.body_vel_world_2d[0]

        self.input_covariance = self.rotation_body_to_world @ self.noise_spectral_density @ self.rotation_body_to_world.T * self.dt

        self.prediction_covariance = self.state_transition_matrix @ prediction_covariance @ self.state_transition_matrix.T + self.input_covariance

    def adjust_motion_params(self, params):
        """

        :param params: alpha parameter array (13X1)
        :return:
        """
        self.params = params
        return None