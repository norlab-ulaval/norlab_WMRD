import numpy as np
from util.util_func import *
from util.transform_algebra import *

from models.general.general_model import Gen_Model
from models.kinematic.kinematic_model import Kin_Model
import wmrde

## Warthog dimensions
k1 = 0
k2 = 0.5826
k3 = 0.24979 + 0.27218
k4 = 0.457367
k5 = 0
k6 = 0.012977

wheel_radius = 0.3

map_path = 'data/maps/flat_surface_100000.vtk'

kin_model = Kin_Model(map_path)

kin_model.add_body_frame(name="Body")


transform_dl = np.eye(4)
euler_dl = np.array([0, 0, 0])
p_dl = np.array([k1, k2, -k3])
euler_pose_to_transform(euler_dl, p_dl, transform_dl)
kin_model.add_joint_frame(name="DL", parent_name="Body", dof_string="Ry", is_actuated=False,
                          rigid_transform_parent_no_disp=transform_dl)

transform_fl = np.eye(4)
euler_fl = np.array([0, 0, 0])
p_fl = np.array([k4, k5, -k6])
euler_pose_to_transform(euler_fl, p_fl, transform_fl)
kin_model.add_wheel_frame(name="FL", parent_name="DL", dof_string="Ry", is_actuated=True,
                          rigid_transform_parent_no_disp=transform_fl)

transform_rl = np.eye(4)
euler_rl = np.array([0, 0, 0])
p_rl = np.array([-k4, k5, -k6])
euler_pose_to_transform(euler_rl, p_rl, transform_rl)
kin_model.add_wheel_frame(name="RL", parent_name="DL", dof_string="Ry", is_actuated=True,
                          rigid_transform_parent_no_disp=transform_rl)

transform_dr = np.eye(4)
euler_dr = np.array([0, 0, 0])
p_dr = np.array([k1, -k2, -k3])
euler_pose_to_transform(euler_dr, p_dr, transform_dr)
kin_model.add_joint_frame(name="DR", parent_name="Body", dof_string="Ry", is_actuated=False,
                          rigid_transform_parent_no_disp=transform_dr)

transform_fr = np.eye(4)
euler_fr = np.array([0, 0, 0])
p_fr = np.array([k4, -k5, -k6])
euler_pose_to_transform(euler_fr, p_fr, transform_fr)
kin_model.add_wheel_frame(name="FR", parent_name="DR", dof_string="Ry", is_actuated=True,
                          rigid_transform_parent_no_disp=transform_fr)

transform_rr = np.eye(4)
euler_rr = np.array([0, 0, 0])
p_rr = np.array([-k4, -k5, -k6])
euler_pose_to_transform(euler_rr, p_rr, transform_rr)
kin_model.add_wheel_frame(name="RR", parent_name="DR", dof_string="Ry", is_actuated=True,
                          rigid_transform_parent_no_disp=transform_rr)

kin_model.holonomic_joint_constraints = np.array([1, 0, 0, 1, 0, 0])
kin_model.free_states = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0])

kin_model.wheel_radius = 0.3

kin_model.update_arrays()

# kin_model.init_terrain_contact()

kin_model.define_kinematic_chains()

init_state = np.array([0.0, 0.0, wheel_radius+k6+k3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# kin_model.forward_position_kinematics(init_state)
# kin_model.compute_wheel_jacobians()

kin_model.init_terrain_contact(init_state)

y = np.zeros(69)
y[0:12] = np.array([5.29667e-18, -0.0221193, 0, 0, 0, 1.03635, 0.000373859, 0, 0, 0.000373859, 0, 0])
ydot = np.zeros(69)
u = np.full(4, 10)

## Predictor interface class test
# predictor = wmrde.Predictor()
# print(predictor)
# print(predictor.predict(y, u, ydot, 0.05))