#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def powerTrain_vel(tau_c, tau_d, cmd_vel, model_vel, index, dt):

    if cmd_vel[index] > -13 and cmd_vel[index] < 13:
        cmd_delay_index = int(index - tau_d / dt)
        new_vel = model_vel[index] + (1 / tau_c) * (cmd_vel[cmd_delay_index] - model_vel[index]) * dt
    
    elif cmd_vel[index] <= -13:
        new_vel = -13
        
    else:
        new_vel = 13

    return new_vel

def central_diff(f_over, f_under, d):
    
    return (f_over - f_under) / (2 * d)

def forward_diff(f_over, f, d):
    
    return np.divide(f_over - f, d)
    
    
def quaternion_to_euler(w, x, y, z):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1,
                     np.sign(sinp) * np.pi / 2,
                     np.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def wrap2pi(angle):
    if angle <= np.pi and angle >= -np.pi:
        return(angle)
    elif angle < -np.pi:
        return(wrap2pi(angle + 2*np.pi))
    else:
        return(wrap2pi(angle - 2*np.pi))
    
def comp_disp(x_fut, x_pres):
    x_disp = x_fut - x_pres
    return np.linalg.norm(x_disp)

def disp_err(x_pred, x):
    x_err = x_pred - x
    x[2] = wrap2pi(x[2])
    return x_err @ x_err
    
def up_propa_mat(mat, ang):
    propa_cos = np.cos(ang)
    propa_sin = np.sin(ang)
    mat[0,0] = propa_cos
    mat[0,1] = -propa_sin
    mat[1,0] = propa_sin
    mat[1,1] = propa_cos
    
    return mat

def euler_to_rotmat(euler):
    R = np.empty((3,3))
    s_roll = np.sin(euler[0])
    c_roll = np.cos(euler[0])
    s_pitch = np.sin(euler[1])
    c_pitch = np.cos(euler[1])
    s_yaw = np.sin(euler[2])
    c_yaw = np.cos(euler[2])

    R[0,0] = c_pitch * c_yaw
    R[0, 1] = c_yaw * s_pitch * s_roll - c_roll * s_yaw
    R[0, 2] = s_roll * s_yaw + c_roll * c_yaw * s_pitch
    R[1, 0] = c_pitch * s_yaw
    R[1, 1] = c_roll * c_yaw + s_pitch * s_roll * s_yaw
    R[1, 2] = c_roll * s_pitch * s_yaw - c_yaw * s_roll
    R[2, 0] = -s_pitch
    R[2, 1] = c_pitch * s_roll
    R[2, 2] = c_pitch * c_roll
    return R

def pose_to_transform(euler, p):
    T = np.empty((4,4))
    T[0:3, 0:3] = euler_to_rotmat(euler)
    T[0:3, 3] = p
    return T

def rigid_tranformation_3d(params):
    """Returns a rigid transformation matrix

    :params: numpy array, params[0]=tx, params[1]=ty, params[2]=theta
    :returns: LaTeX bmatrix as a string
    """
    return np.array([[np.cos(params[2]), -np.sin(params[2]), params[0]],
                     [np.sin(params[2]), np.cos(params[2]), params[1]],
                     [0, 0, 1]])

