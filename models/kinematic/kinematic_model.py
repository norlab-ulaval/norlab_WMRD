from pypointmatcher import pointmatcher, pointmatchersupport
import numpy as np
from models.general.general_model import Gen_Model
from util.transform_algebra import *
from util.util_func import *

class Kin_Model(Gen_Model):
    def __init__(self, map_path):
        super().__init__()
        self.PM = pointmatcher.PointMatcher
        matcher_params = pointmatchersupport.Parametrizable.Parameters()
        matcher_params["maxDist"] = "1.0"
        matcher_params["knn"] = "10"
        self.matcher = self.PM.get().MatcherRegistrar.create("KDTreeMatcher", matcher_params)
        self.map = self.PM.DataPoints.load(map_path)
        self.matcher.init(self.map)

        self.optimization_step = 0.1

    def update_arrays(self):
        self.wheel_poses = np.zeros((4, self.number_wheels))
        self.contact_height_errors = np.zeros(self.number_wheels)
        self.contact_cost = 0
        self.contact_velocity_derivative = np.zeros((1+self.number_wheels, 5+self.number_frames))
        self.contact_velocity_derivative[self.number_wheels, 6:] = self.holonomic_joint_constraints
        self.contact_state_derivative = np.zeros((1+self.number_wheels, 5+self.number_frames))
        self.contact_free_state_derivative = np.zeros((1+self.number_wheels, 1+self.number_wheels))

        init_query = np.zeros((4, self.number_wheels))
        init_query[3, :] = 1.0
        self.query_array = self.PM.DataPoints(features=init_query, featureLabels=self.map.featureLabels)

        for i in range(0, self.number_frames):
            if self.frames[i].is_wheel:
                self.frames[i].wheel_jacobian = np.zeros((3, 5 + self.number_frames))
        self.full_wheel_jacobians = np.zeros((3*self.number_wheels, 5+self.number_frames))

        self.omega_matrix = np.eye(3)
        self.block_diagonal_transform = np.eye(6 + self.number_frames - 1)

        self.hessian_matrix = np.zeros((5+self.number_frames, 5+self.number_frames))
        self.hessian_mask = np.zeros((5+self.number_frames, 5+self.number_frames))
        self.hessian_mask[2::3] = 1
        self.gradient_matrix = np.zeros((5+self.number_frames, 5+self.number_frames))

    def compute_spatial_velocity_conversion(self, state):
        euler_pose_to_omega_submatrix(quaternion_to_euler(state[3:]), self.omega_matrix)
        self.block_diagonal_transform[:3, :3] = self.omega_matrix
        self.block_diagonal_transform[3:6, 3:6] = self.frames[0].rigid_transform_to_world[:3, :3]

    def forward_position_kinematics(self, state):
        """ Algorithm 1 in Seegmiller thesis
        :param state: state array [pose, config]:
        :return:
        """
        pose = state[:7]
        config = state[7:]
        quaternion_pose_to_transform(pose, self.frames[0].rigid_transform_to_world)
        for i in range(1, self.number_frames):
                # TODO: Code all possibilities for all joint types
            if self.frames[i].dof_string == "Ry":
                parent_id = self.frames[i].parent_id
                self.frames[i].euler_rotation[1] = config[i-1]
                euler_to_transform(self.frames[i].euler_rotation, self.frames[i].rigid_transform_joint_state)
                self.frames[i].rigid_transform_parent_joint = self.frames[i].rigid_transform_joint_state @ \
                                                              self.frames[i].rigid_transform_parent_joint_nodisp
                self.frames[i].rigid_transform_to_world = self.frames[i].rigid_transform_parent_joint @ \
                                                          self.frames[parent_id].rigid_transform_to_world
        self.compute_spatial_velocity_conversion(state)
        return 1

    def find_wheel_poses(self):
        wheel_count = 0
        for i in range(1, self.number_frames):
            if self.frames[i].is_wheel == True:
                self.wheel_poses[:, wheel_count] = self.frames[i].rigid_transform_to_world[:, 3]
                wheel_count += 1

    def find_closest_map_points_from_wheels(self):
        self.query_array.features = self.wheel_poses
        return self.matcher.findClosests(self.query_array)

    def compute_contact_frames(self):
        self.find_wheel_poses()
        closest_map_points = self.find_closest_map_points_from_wheels()

        wheel_count = 0
        for i in range(0, self.number_frames):
            if self.frames[i].is_wheel:
                self.frames[i].rigid_transform_contact_to_world[:3, 3] = self.map.features[:3, closest_map_points.ids[0, wheel_count]]
                self.frames[i].rigid_transform_contact_to_wheel[:3, 3] = self.frames[i].rigid_transform_contact_to_world[:3, 3] - \
                                                                         self.frames[i].rigid_transform_to_world[:3, 3]
                self.frames[i].rigid_transform_contact_to_wheel[1, 3] = 0
                contact_point_normal_world = self.map.getDescriptorViewByName("normals")[:, closest_map_points.ids[0, wheel_count]]
                contact_point_normal_world_xz = contact_point_normal_world
                contact_point_normal_world_xz[1] = 0
                contact_point_normal_world_xz = contact_point_normal_world_xz / np.linalg.norm(contact_point_normal_world_xz)

                self.frames[i].rigid_transform_contact_to_world[:3, 2] = contact_point_normal_world_xz
                self.frames[i].rigid_transform_contact_to_world[:3, 1] = self.frames[i].rigid_transform_to_world[:3, 1]
                self.frames[i].rigid_transform_contact_to_world[:3, 0] = np.cross(self.frames[i].rigid_transform_to_world[:3, 1],
                                                                                  contact_point_normal_world_xz)

                self.frames[i].rigid_transform_contact_to_wheel = self.frames[i].rigid_transform_contact_to_world @ \
                                                                  np.linalg.inv(self.frames[i].rigid_transform_to_world)
                self.frames[i].contact_point_angle = np.arctan2(self.frames[i].rigid_transform_contact_to_wheel[0,2],
                                                                self.frames[i].rigid_transform_contact_to_wheel[0,0])

                self.contact_height_errors[wheel_count] = np.linalg.norm(self.frames[i].rigid_transform_contact_to_wheel[:3, 3]) - \
                                                                    self.wheel_radius

                wheel_count += 1

    def compute_wheel_jacobians(self):
        # TODO: Validate computation, re-check result from line 12 of algo 2
        # self.compute_contact_frames()
        wheel_count = 0
        for i in range(1, self.number_frames):
            if self.frames[i].is_wheel == True:
                for next_parent_id in self.frames[i].kinematic_chain_to_body:
                    # TODO: Implement for all DoF types
                    if self.frames[i].dof_string == "Ry":
                        frame_to_world_vector = self.frames[next_parent_id].rigid_transform_to_world[:3, 3]
                        frame_to_world_rotation_vector = self.frames[next_parent_id].rigid_transform_to_world[:3, 1]


                        self.frames[i].wheel_jacobian[:, next_parent_id+5] = np.cross(frame_to_world_rotation_vector,
                                                                       (self.frames[i].rigid_transform_contact_to_world[:3, 3] - frame_to_world_vector))

                contact_to_world_body_to_world_diff_vector = self.frames[i].rigid_transform_contact_to_world[:3, 3] - \
                                                             self.frames[0].rigid_transform_to_world[:3, 3]

                self.frames[i].wheel_jacobian[:, :3] = cross_product_skew_symmetric_from_vector(
                    contact_to_world_body_to_world_diff_vector).T @ \
                                                       self.frames[0].rigid_transform_to_world[:3, :3]
                self.frames[i].wheel_jacobian[:, 3:6] = self.frames[0].rigid_transform_to_world[:3, :3]
                self.frames[i].wheel_jacobian = self.frames[i].rigid_transform_contact_to_world[:3, :3].T @ self.frames[
                    i].wheel_jacobian

                self.full_wheel_jacobians[wheel_count*3:wheel_count*3+3, :] = self.frames[i].wheel_jacobian
                wheel_count += 1

    def compute_terrain_cost(self):
        self.contact_cost = self.contact_height_errors.T @ self.contact_height_errors

    def compute_terrain_gradient(self):
        self.contact_velocity_derivative[:4, :] = self.full_wheel_jacobians[2::3, :]
        self.contact_state_derivative = self.contact_velocity_derivative @ self.block_diagonal_transform.T
        self.contact_free_state_derivative = self.contact_state_derivative[:, np.where(self.free_states)]
        # TODO: finish gradient computation

    def init_terrain_contact(self, state):
        #TODO: Implement Newton minimization (first step is to define gradient and Hessian)
        self.forward_position_kinematics(state)
        self.compute_spatial_velocity_conversion(state)
        self.compute_contact_frames()
        self.compute_wheel_jacobians()
        self.compute_terrain_cost()
        self.compute_terrain_gradient()
        print("A = ")
        print(self.full_wheel_jacobians)
        print("derrdot_dqvel = ")
        print(self.contact_velocity_derivative)
        print("derr_dstate = ")
        print(self.contact_state_derivative)
        print("derr_dx = ")
        print(self.contact_free_state_derivative)