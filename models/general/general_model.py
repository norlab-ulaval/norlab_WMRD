import math
import numpy as np

class Gen_Model():
    def __init__(self):
        # TODO: Use a URDF file to initialize all frames
        self.number_frames = 0
        self.number_wheels = 0
        self.number_actuated = 0
        self.wheel_radius = 0

        self.frames = []
        self.holonomic_joint_constraints = None

    def name_to_id(self, name):
        for i in range(0, len(self.frames)):
            if self.frames[i].name == name:
                return i

    def add_body_frame(self, name):
        body_frame = self.Frame()
        body_frame.name = name

        self.frames.append(body_frame)
        self.number_frames += 1

    def add_joint_frame(self, name, parent_name, dof_string, is_actuated, rigid_transform_parent_no_disp):
        joint_frame = self.Frame()
        joint_frame.name = name
        joint_frame.parent_id = self.name_to_id(parent_name)
        joint_frame.rigid_transform_parent_joint_nodisp = rigid_transform_parent_no_disp

        self.frames.append(joint_frame)
        self.number_frames += 1
        if joint_frame.is_actuated:
            self.number_actuated += 1

    def add_wheel_frame(self, name, parent_name, dof_string, is_actuated, rigid_transform_parent_no_disp):
        wheel_frame = self.Frame()
        wheel_frame.name = name
        wheel_frame.parent_id = self.name_to_id(parent_name)

        wheel_frame.is_wheel = True
        wheel_frame.is_actuated = is_actuated
        wheel_frame.rigid_transform_parent_joint_nodisp = rigid_transform_parent_no_disp

        wheel_frame.rigid_transform_contact_to_world = np.eye(4)
        wheel_frame.rigid_transform_contact_to_wheel = np.eye(4)
        wheel_frame.contact_angle = 0

        self.frames.append(wheel_frame)
        self.number_frames += 1
        self.number_wheels+= 1
        if wheel_frame.is_actuated:
            self.number_actuated += 1

    def define_kinematic_chains(self):
        for i in range(0, self.number_frames):
            self.frames[i].kinematic_chain_to_body.append(i)
            next_parent_id = self.frames[i].parent_id
            while next_parent_id != -1:
                self.frames[i].kinematic_chain_to_body.append(next_parent_id)
                next_parent_id = self.frames[next_parent_id].parent_id

    # inner frame class
    class Frame():
        def __init__(self):
            self.name = ""
            self.dof_string = "Ry"
            self.parent_id = -1

            self.is_wheel = False
            self.is_sprocket = False
            self.is_actuated = False
            self.is_fixed = False

            self.euler_rotation = np.zeros(3)

            self.rigid_transform_parent_joint_nodisp = np.eye(4)
            self.rigid_transform_parent_joint = np.eye(4)
            self.rigid_transform_joint_state = np.eye(4)
            self.rigid_transform_to_world = np.eye(4)

            self.scalar_mass = 0
            self.center_of_mass = np.array([0, 0, 0])
            self.moment_of_intertia = np.zeros((3,3))

            self.kinematic_chain_to_body = []






