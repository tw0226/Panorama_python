import numpy as np
import cv2

class Matching:
    def __init__(self, HOG_feature1, HOG_feature2, threshold):
        self.HOG_feature_1 = HOG_feature1
        self.HOG_feature_2 = HOG_feature2
        self.distance_threshold = threshold
        self.matching_points = []
    def matching_point_making(self):
        matching_point_list = []
        for feature_1 in self.HOG_feature_1:
            y1, x1, vector_1 = feature_1
            min = 1e+7
            matching_point = None
            for feature_2 in self.HOG_feature_2:
                y2, x2, vector_2 = feature_2
                distance = np.linalg.norm(vector_1 - vector_2)
                if distance < min:
                    min = distance
                    matching_point = ((y1, x1), (y2, x2), distance)
            if min < self.distance_threshold:
                matching_point_list.append(matching_point)
        matching_point_list = sorted(matching_point_list, key=lambda distance: distance[2])
        self.matching_points = matching_point_list

    def get_matching_points(self):
        return self.matching_points