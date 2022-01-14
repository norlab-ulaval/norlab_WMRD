from pypointmatcher import pointmatcher, pointmatchersupport
import numpy as np
from models.general.general_model import Gen_Model

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

    def find_closest_map_points_from_wheels(self, wheel_poses):
        self.query_array.features = wheel_poses
        return self.matcher.findClosests(self.query_array)

    def compute_contact_height(self, wheel_poses):
        # TODO : call find_closest_map_points_from_wheels method and use wheel radius to compute contact height error



