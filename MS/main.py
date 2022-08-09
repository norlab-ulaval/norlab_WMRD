import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt


# Vehicles measurements

b_body_width_in_meters = 1
r_wheels_radius_in_meters = 0.10
frequency = 10

# Some maths and definitions

J = r_wheels_radius_in_meters * np.array([[(1/2),(1/2)],
             [0,0],
             [(-1/b_body_width_in_meters), (1/b_body_width_in_meters)]])

def get_twist_speeds_from_wheels_speed(W_wheels_speed_in_rads_per_sec):
    twist_speeds = J @ W_wheels_speed_in_rads_per_sec
    return twist_speeds

def get_wheels_speed_from_twist_speeds(twist_speeds):
    wheels_speed = np.linalg.pinv(J) @ twist_speeds
    return wheels_speed

# Testing
#W_wheels_speed_in_rads_per_sec = np.array([1, 1])
#get_twist_speeds_from_wheels_speed(W_wheels_speed_in_rads_per_sec)

#test = np.array([0,0,0.5])
#get_wheels_speed_from_twist_speeds(test)


def import_joint_states(path):
    df = pd.read_csv(path, dtype="string", delimiter=",")
    return df

def convert_list_of_strings_to_array_of_floats(list_of_strings):
    list_of_floats = []
    for string in list_of_strings:
        list_of_floats.append(literal_eval(string))
    return np.array(list_of_floats)

def divide_by_wheel(positions):
    front_left = positions[:,0]
    front_right = positions[:,1]
    rear_left = positions[:,2]
    rear_right = positions[:,3]
    return front_left, front_right, rear_left, rear_right

def compute_speed(wheel_ticks, dt):
    wheel_speed = np.diff(wheel_ticks) / dt
    return wheel_speed

def convoluate_speed_npoints(speed, npoints):
    speed_conv = np.convolve(speed, np.ones((npoints,))/npoints, mode='same')
    return speed_conv

def rad_per_sec_to_m_per_sec(speed, radius):
    return speed * radius

list_of_strings = import_joint_states("_slash_joint_states.csv").values[:,8]
positions = convert_list_of_strings_to_array_of_floats(list_of_strings)
front_left, front_right, rear_left, rear_right = divide_by_wheel(positions)

speed = compute_speed(front_left, 1/frequency)
speed = convoluate_speed_npoints(speed, 10)
speed_in_m_per_sec = rad_per_sec_to_m_per_sec(speed, r_wheels_radius_in_meters)

plt.figure(1)
plt.plot(speed_in_m_per_sec)
plt.title("Speed read from Joint States")
plt.xlabel("Timestamp")
plt.ylabel("Speed in m/s")

plt.figure(2)
plt.plot(speed)
plt.title("Speed read from Joint States")
plt.xlabel("Timestamp")
plt.show()