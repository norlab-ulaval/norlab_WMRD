import numpy as np
from util.transform_algebra import *

class Perturbed_unicycle:
    def __init__(self, radius, baseline, alpha_params, dt):
        self.dt = dt
        self.radius = radius
        self.baesline = baseline
        self.alpha_params = np.reshape(alpha_params, (13,1))
        # self.block_diag_coeff_matrix = np.zeros((3 ,13))
        # self.block_diag_coeff_matrix[0, :4] = coeff_x
        # self.block_diag_coeff_matrix[1, 4:8] = coeff_y
        # self.block_diag_coeff_matrix[2, 8:12] = coeff_z

        self.diff_drive_jacobian = np.array([[self.radius/2, self.radius/2],
                                             [0, 0],
                                             [-self.radius/self.baesline, self.radius/self.baesline]])

        self.diff_drive_vel = np.zeros((3, 1))

        self.coeff_x = np.zeros(4)
        self.coeff_y = np.zeros(4)
        self.coeff_z = np.zeros(5)
        self.coeff_matrix = np.zeros((3, 13))
        self.slip_vel = np.zeros((3, 1))

        self.rotation_body_to_world = np.eye(2)
        self.body_vel_world_3d = np.zeros(6)
        self.body_vel_world_2d = np.zeros(3)
        self.state_2d = np.zeros(3)

    def update_coeff_matrix(self, gravity_vector):
        self.coeff_x[0] = self.coeff_y[0] = self.coeff_z[0] = self.diff_drive_vel[0]
        self.coeff_x[1] = np.abs(self.diff_drive_vel[2])
        self.coeff_x[2] = self.coeff_x[0] * self.coeff_x[1]
        self.coeff_x[3] = self.coeff_z[3] = gravity_vector[0]

        self.coeff_y[1] = self.coeff_z[1] = self.diff_drive_vel[2]
        self.coeff_y[2] = self.coeff_z[2] = self.coeff_y[0] * self.coeff_y[1]
        self.coeff_y[3] = gravity_vector[1]

        self.coeff_z[4] = gravity_vector[1]

        self.coeff_matrix[0, :4] = self.coeff_x
        self.coeff_matrix[1, 4:8] = self.coeff_y
        self.coeff_matrix[2, 8:14] = self.coeff_z

    def predict(self, init_state, input):
        """
        :param init_state: initial state array [x, y, z, roll, pitch, yaw]
        :param input: input array [omega_l, omega_r]
        :return: next_state
        """
        self.state_2d[:2] = init_state[:2]
        self.state_2d[2] = init_state[-1]
        yaw_to_rotmat2d(self.rotation_body_to_world, init_state[-1])
        gravity_vector = euler_to_rotmat(init_state[2:])[:, 2:]

        self.diff_drive_vel = np.reshape(self.diff_drive_jacobian @ input, (3,1))

        self.update_coeff_matrix(gravity_vector)
        self.slip_vel = self.coeff_matrix @ self.alpha_params

        body_vel = self.diff_drive_vel + self.slip_vel
        self.body_vel_world_2d[:2] = np.reshape(self.rotation_body_to_world @ body_vel[:2], (2))
        self.body_vel_world_2d[2] = body_vel[2]
        self.body_vel_world_3d[:2] = self.body_vel_world_2d[:2]
        self.body_vel_world_3d[-1] = self.body_vel_world_2d[-1]
        return init_state + self.body_vel_world_3d * self.dt

    def adjust_motion_params(self, alpha_params):
        """

        :param params: alpha parameter array (13X1)
        :return:
        """
        self.alpha_params = np.reshape(alpha_params, (13,1))
        return None