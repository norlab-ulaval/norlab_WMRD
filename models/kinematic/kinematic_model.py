from pypointmatcher import pointmatcher, pointmatchersupport
import numpy as np
from models.general.general_model import Gen_Model
from util.transform_algebra import *

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

        init_query = np.zeros((4, self.number_wheels))
        init_query[3, :] = 1.0
        self.query_array = self.PM.DataPoints(features=init_query, featureLabels=self.map.featureLabels)

    def forward_position_kinematics(self, state):
        """

        :param state: state array [pose, config]:
        :return:
        """
        pose = state[:6]
        config = state[6:]
        pose_to_transform(pose, self.frames[0].rigid_transform_parent_joint)
        for i in range(1, self.number_frames):
            if self.frames[i].is_actuated:
                # Todo: Code all possibilities for all joint types
                if self.frames[i].dof_string == "Ry":
                    print(1)
                    #TODO: Complete

    def find_closest_map_points_from_wheels(self, wheel_poses):
        self.query_array.features = wheel_poses
        return self.matcher.findClosests(self.query_array)

    def compute_contact_height(self, state, wheel_poses):
        # TODO : call find_closest_map_points_from_wheels method and use wheel radius to compute contact height error

        closest_points = self.find_closest_map_points_from_wheels(wheel_poses)
        print(closest_points)


