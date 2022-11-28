import numpy as np
from util.transform_algebra import *

class Bounded_powertrain:
    def __init__(self, max_vel_left, max_vel_right, min_vel_left, min_vel_right):
        self.max_vel_left = max_vel_left
        self.min_vel_left = min_vel_left
        self.max_vel_right = max_vel_right
        self.min_vel_right = min_vel_right

    def compute_bounded_wheel_vels(self, wheel_vel_left, wheel_vel_right):
        bounded_wheel_vel_left = np.clip(wheel_vel_left, self.min_vel_left, self.max_vel_left)
        bounded_wheel_vel_right = np.clip(wheel_vel_right, self.min_vel_right, self.max_vel_right)
        return bounded_wheel_vel_left, bounded_wheel_vel_right