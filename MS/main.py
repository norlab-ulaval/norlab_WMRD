import numpy as np
import pandas as pd

# Vehicles measurements

b_body_width_in_meters = 1
r_wheels_radius_in_meters = 1

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
    odom = pd.read_csv(path, skiprows=[1], dtype="string")
    return odom


def remove_brackets_from_string_array(string_array):
    new_string_array = []
    for string in string_array:
        allo = string.replace("[", "").replace("]", "").replace("'", "")
        new_string_array.append(allo.split(","))
    return new_string_array

odom = import_joint_states("_slash_joint_states.csv")
string_array_without_brackets = remove_brackets_from_string_array(odom.values[:,8])
print(string_array_without_brackets[0])






