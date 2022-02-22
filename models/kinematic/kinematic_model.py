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

    def update_arrays(self):
        self.wheel_poses = np.zeros((4, self.number_wheels))
        self.contact_height_errors = np.zeros(self.number_wheels)

        init_query = np.zeros((4, self.number_wheels))
        init_query[3, :] = 1.0
        self.query_array = self.PM.DataPoints(features=init_query, featureLabels=self.map.featureLabels)

        self.wheel_jacobians = np.zeros(3* self.number_wheels)

    def forward_position_kinematics(self, state):
        """ Algorithm 1 in Seegmiller thesis
        :param state: state array [pose, config]:
        :return:
        """
        print(state)
        print(self.frames[-1].rigid_transform_to_world)
        pose = state[:7]
        config = state[7:]
        quaternion_pose_to_transform(pose, self.frames[0].rigid_transform_parent_joint)
        for i in range(1, self.number_frames):
            if self.frames[i].is_actuated:
                # TODO: Code all possibilities for all joint types
                if self.frames[i].dof_string == "Ry":
                    self.frames[i].euler_rotation[1] = config[i-1]
                    euler_to_transform(self.frames[i].euler_rotation, self.frames[i].rigid_transform_joint_state)
                    self.frames[i].rigid_transform_parent_joint = self.frames[i].rigid_transform_joint_state @ \
                                                                  self.frames[i].rigid_transform_parent_joint_nodisp
                    self.frames[i].rigid_transform_to_world = self.frames[i].rigid_transform_parent_joint @ \
                                                              self.frames[i-1].rigid_transform_to_world

        print(self.frames[-1].rigid_transform_to_world)
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

    def compute_contact_height(self):
        self.find_wheel_poses()
        closest_map_points = self.find_closest_map_points_from_wheels()

        for i in range(0, self.number_wheels):
            self.contact_height_errors[i] = comp_disp(self.map.features[:3, closest_map_points.ids[0, i]],
                                                      self.wheel_poses[:3, i]) - self.wheel_radius


    def compute_wheel_jacobians(self):
        # TODO: Complete wheel jacobians

