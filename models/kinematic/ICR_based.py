import numpy as np
from util.transform_algebra import *

class ICR_assymetrical:
    def __init__(self, r, alpha_l, alpha_r, x_icr, y_icr_l, y_icr_r, dt):
        self.dt = dt
        self.r = r
        self.icr_factor = r / (y_icr_l - y_icr_r)
        self.icr_jacobian = np.array([[-y_icr_r, y_icr_l],
                                      [x_icr, -x_icr],
                                      [-1, 1]])
        self.alpha_matrix = np.array([[alpha_l, 0],
                                      [0, alpha_r]])

        self.jacobian = self.icr_factor * self.icr_jacobian @ self.alpha_matrix

        self.rotation_body_to_world = np.eye(2)
        self.body_vel_world_3d = np.zeros(6)
        self.body_vel_world_2d = np.zeros(3)
        self.state_2d = np.zeros(3)

    def predict(self, init_state, input):
        """
        :param init_state: initial state array [x, y, z, roll, pitch, yaw]
        :param input: input array [omega_l, omega_r]
        :return: next_state
        """
        self.state_2d[:2] = init_state[:2]
        self.state_2d[2] = init_state[-1]
        yaw_to_rotmat2d(self.rotation_body_to_world, init_state[-1])
        body_vel = self.jacobian @ input
        self.body_vel_world_2d[:2] = self.rotation_body_to_world @ body_vel[:2]
        self.body_vel_world_2d[2] = body_vel[2]
        self.body_vel_world_3d[:2] = self.body_vel_world_2d[:2]
        self.body_vel_world_3d[-1] = self.body_vel_world_2d[-1]
        return init_state + self.body_vel_world_3d * self.dt

    def adjust_motion_params(self, params):
        """

        :param params: motion params array [alpha_l, alpha_r, x_icr, y_icr_l, y_icr_r]
        :return:
        """
        self.icr_factor = self.r / (params[3] - params[4])
        self.icr_jacobian = np.array([[-params[4], params[3]],
                                      [params[2], -params[2]],
                                      [-1, 1]])
        self.alpha_matrix = np.array([[params[0], 0],
                                      [0, params[1]]])

        self.jacobian = self.icr_factor * self.icr_jacobian @ self.alpha_matrix
        return None

class ICR_symmetrical:
    def __init__(self, r, alpha, y_icr, dt):
        self.dt = dt
        self.r = r
        self.jacobian = r * alpha * np.array([[0.5, 0.5],
                                              [0, 0],
                                              [-1/(2*y_icr), 1/(2*y_icr)]])

        self.rotation_body_to_world = np.eye(2)
        self.body_vel_world_3d = np.zeros(6)
        self.body_vel_world_2d = np.zeros(3)
        self.state_2d = np.zeros(3)

    def predict(self, init_state, input):
        """
        :param init_state: initial state array [x, y, z, roll, pitch, yaw]
        :param input: input array [omega_l, omega_r]
        :return: next_state
        """
        self.state_2d[:2] = init_state[:2]
        self.state_2d[2] = init_state[-1]
        yaw_to_rotmat2d(self.rotation_body_to_world, init_state[-1])
        body_vel = self.jacobian @ input
        self.body_vel_world_2d[:2] = self.rotation_body_to_world @ body_vel[:2]
        self.body_vel_world_2d[2] = body_vel[2]
        self.body_vel_world_3d[:2] = self.body_vel_world_2d[:2]
        self.body_vel_world_3d[-1] = self.body_vel_world_2d[-1]

        return init_state + self.body_vel_world_3d * self.dt

    def adjust_motion_params(self, params):
        """

        :param params: array of motion params [alpha, y_icr]
        :return:
        """
        self.jacobian = self.r * params[0] * np.array([[0.5, 0.5],
                                              [0, 0],
                                              [-1 / (2 * params[1]), 1 / (2 * params[1])]])
        return None