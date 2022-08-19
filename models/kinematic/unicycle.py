import numpy as np
from util.transform_algebra import *

class Unicycle:
    def __init__(self, dt):
        self.dt = dt

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
        body_vel = np.zeros(3)
        body_vel[0] = input[0]
        body_vel[2] = input[1]
        self.body_vel_world_2d[:2] = self.rotation_body_to_world @ body_vel[:2]
        self.body_vel_world_2d[2] = body_vel[2]
        self.body_vel_world_3d[:2] = self.body_vel_world_2d[:2]
        self.body_vel_world_3d[-1] = self.body_vel_world_2d[-1]
        return init_state + self.body_vel_world_3d * self.dt
