import numpy as np
from util.util_func import *

from models.kinematic.kinematic import Kin_Model

## Warthog dimensions
k1 = 0
k2 = 0.5826
k3 = 0.24979
k4 = 0.457367
k5 = 0
k6 = 0.012977

kin_model = Kin_Model()

kin_model.add_body_frame(name="Body")

euler = np.array([0, 0, 0])
p = np.array([k1, k2, -k3])
kin_model.add_joint_frame(name="DL", parent_name="Body", dof_string="RY", is_actuated=False,
                          rigid_transform_parent_no_disp=pose_to_transform(euler, p))
euler = np.array([0, 0, 0])
p = np.array([k4, k5, -k6])
kin_model.add_wheel_frame(name="FL", parent_name="DL", dof_string="RY", is_actuated=True,
                          rigid_transform_parent_no_disp=pose_to_transform(euler, p))
euler = np.array([0, 0, 0])
p = np.array([-k4, k5, -k6])
kin_model.add_wheel_frame(name="RL", parent_name="DL", dof_string="RY", is_actuated=True,
                          rigid_transform_parent_no_disp=pose_to_transform(euler, p))


print(kin_model.frames[2].name)
print(kin_model.frames[2].rigid_transform_parent_joint_nodisp)