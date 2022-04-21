import math
from pypointmatcher import pointmatcher, pointmatchersupport
import glob
import numpy as np
import copy
import pandas as pd

def rigid_transform_from_quat(t, Q):
    '''From https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/'''
    T = np.eye(4)
    T[0, 3] = t[0]
    T[1, 3] = t[1]
    T[2, 3] = t[2]
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
    # First row of the rotation matrix
    T[0,0] = 2 * (q0 * q0 + q1 * q1) - 1
    T[0,1] = 2 * (q1 * q2 - q0 * q3)
    T[0,2] = 2 * (q1 * q3 + q0 * q2)
    # Second row of the rotation matrix
    T[1,0] = 2 * (q1 * q2 + q0 * q3)
    T[1,1] = 2 * (q0 * q0 + q2 * q2) - 1
    T[1,2] = 2 * (q2 * q3 - q0 * q1)
    # Third row of the rotation matrix
    T[2,0] = 2 * (q1 * q3 - q0 * q2)
    T[2,1] = 2 * (q2 * q3 + q0 * q1)
    T[2,2] = 2 * (q0 * q0 + q3 * q3) - 1
    return(T)

PM = pointmatcher.PointMatcher
icp = PM.ICP()
icp.loadFromYaml("../icp_config/icp.yaml")
matcher_params = pointmatchersupport.Parametrizable.Parameters()
matcher_params["knn"] = "3"
matcher = PM.get().MatcherRegistrar.create("KDTreeMatcher", matcher_params)

selected_pcls_T = np.load('../../fr2021_data/icp/run5_500_pcls.npy')
if path == 'b':
    map = PM.DataPoints.load('../../fr2021_data/maps_icp/laverdiereb.vtk')
    pcl_list = sorted(glob.glob('../../fr2021_data/pcl_runs/results_fr2021_run6/*'))
    selected_pcls_T = np.load('../../fr2021_data/icp/run6_500_pcls.npy')
if path == 'c' :
    map = PM.DataPoints.load('../../fr2021_data/maps_icp/gazebo.vtk')
    pcl_list = sorted(glob.glob('../../fr2021_data/pcl_runs/results_fr2021_run2/*'))
    selected_pcls_T = np.load('../../fr2021_data/icp/run2_500_pcls.npy')

for i in range(0, selected_pcls_T.shape[0]):
    print(f'%s / %s' % (i+1, selected_pcls_T.shape[0]))
    map_tmp = PM.DataPoints(map)
    input_pcl = PM.DataPoints.load(pcl_list[int(selected_pcls_T[i, 0])])
    T = rigid_transform_from_quat(selected_pcls_T[i, 2:5], selected_pcls_T[i, 5:])
    T_inv = np.linalg.inv(T)
    icp.transformations.apply(input_pcl, T_inv)
    icp.transformations.apply(map_tmp, T_inv)
    matcher.init(map_tmp)
    for j in range(0, pts.shape[0]):
        dx = pts[j]
        dy = 0
        yaw_angle = 0
        err_arr[i, j] = np.mean(perturbate_pcl(icp, matcher, map_tmp, input_pcl, dx, dy, yaw_angle))