import numpy as np
from util.util_func import *
from util.transform_algebra import *

from models.general.general_model import Gen_Model
from models.kinematic.kinematic_model import Kin_Model

## Warthog dimensions
k1 = 0
k2 = 0.5826
k3 = 0.24979 + 0.27218
k4 = 0.457367
k5 = 0
k6 = 0.012977

wheel_radius = 0.3

map_path = 'data/maps/low_speed_CCW.vtk'

kin_model = Kin_Model(map_path)

kin_model.add_body_frame(name="Body")


transform = np.eye(4)
euler = np.array([0, 0, 0])
p = np.array([k1, k2, -k3])
euler_pose_to_transform(euler, p, transform)
kin_model.add_joint_frame(name="DL", parent_name="Body", dof_string="Ry", is_actuated=False,
                          rigid_transform_parent_no_disp=transform)
euler = np.array([0, 0, 0])
p = np.array([k4, k5, -k6])
euler_pose_to_transform(euler, p, transform)
kin_model.add_wheel_frame(name="FL", parent_name="DL", dof_string="Ry", is_actuated=True,
                          rigid_transform_parent_no_disp=transform)
euler = np.array([0, 0, 0])
p = np.array([-k4, k5, -k6])
euler_pose_to_transform(euler, p, transform)
kin_model.add_wheel_frame(name="RL", parent_name="DL", dof_string="Ry", is_actuated=True,
                          rigid_transform_parent_no_disp=transform)

euler = np.array([0, 0, 0])
p = np.array([k1, -k2, -k3])
euler_pose_to_transform(euler, p, transform)
kin_model.add_joint_frame(name="DR", parent_name="Body", dof_string="Ry", is_actuated=False,
                          rigid_transform_parent_no_disp=transform)
euler = np.array([0, 0, 0])
p = np.array([k4, -k5, -k6])
euler_pose_to_transform(euler, p, transform)
kin_model.add_wheel_frame(name="FR", parent_name="DR", dof_string="Ry", is_actuated=True,
                          rigid_transform_parent_no_disp=transform)
euler = np.array([0, 0, 0])
p = np.array([-k4, -k5, -k6])
euler_pose_to_transform(euler, p, transform)
kin_model.add_wheel_frame(name="RR", parent_name="DR", dof_string="Ry", is_actuated=True,
                          rigid_transform_parent_no_disp=transform)

kin_model.update_arrays()

kin_model.define_kinematic_chains()

init_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0])

kin_model.forward_position_kinematics(init_state)

kin_model.compute_contact_frames()

# for i in range(0, kin_model.number_frames):
#     print(kin_model.frames[i].kinematic_chain_to_body)
#
# kin_model.wheel_radius = wheel_radius
#
# # tests general model
# print(kin_model.frames[2].name)
# print(kin_model.frames[2].rigid_transform_parent_joint_nodisp)
#
# # tests map
# # print(kin_model.map.features)
# test_point = np.array([[0.0, 1.0, 2.0, 3.0],
#                         [0.0, 1.0, 2.0, 3.0],
#                         [0.0, 0.0, 0.0, 0.0],
#                         [1.0, 1.0, 1.0, 1.0]])
#
#
# # init_pose = np.array([0, 1.0, 0, 1.0, 0, 0, 0])
# # test_transform = np.eye(4)
# # pose_to_transform(init_pose, test_transform)
# # print(test_transform)
#
#
# kin_model.compute_contact_height()
# print(kin_model.contact_height_errors)