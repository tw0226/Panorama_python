import numpy as np
import math
import cv2

class MatCollection:
    def __init__(self, img):
        self.img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.threshold = 200
        self.magnitude_Mat = None
        self.direction_Mat = None
        self.gradient_Mat = None

    def set_magnitude_threshold(self, threshold):
        self.threshold = threshold
    def get_gradient_Mat(self):
        return self.gradient_Mat
    def get_magnitude_Mat(self):
        return self.magnitude_Mat
    def get_direction_Mat(self):
        return self.direction_Mat
    def make_gradient_map(self):
        img = self.img
        height, width = img.shape
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        self.gradient_Mat = sobelx, sobely

    def calculate_gradient_magnitude(self):
        gx_Mat,gy_Mat = self.gradient_Mat
        height, width = gx_Mat.shape
        magnitude_Mat = np.zeros(gx_Mat.shape)
        for y in range(height):
            for x in range(width):
                magnitude = math.sqrt(pow(gx_Mat[y, x], 2) + pow(gy_Mat[y, x], 2))
                #if magnitude > self.threshold:
                magnitude_Mat[y, x] = magnitude
        self.magnitude_Mat = magnitude_Mat
    def calculate_gradient_direction(self):
        gx_Mat, gy_Mat = self.gradient_Mat
        direction_Mat = np.zeros(gx_Mat.shape)
        height, width = gx_Mat.shape
        for y in range(height):
            for x in range(width):
                #direction = math.atan2(gy_Mat[y, x], gx_Mat[y, x]) * 180
                direction = cv2.fastAtan2(gy_Mat[y, x], gx_Mat[y, x])
                direction_Mat[y, x] = direction
        self.direction_Mat = direction_Mat
    def compute_all(self, threshold):
        self.set_magnitude_threshold(threshold)
        self.make_gradient_map()
        self.calculate_gradient_magnitude()
        self.calculate_gradient_direction()

class HOG:
    def __init__(self, FeatureMap, Moravec):
        self.feature_Map = FeatureMap
        self.moravec_Map = Moravec
        self.feature_len = 8
        self.HOG_feature_list = None
        self.cell_size = 15
    def set_feature_vector_len(self, len):
        self.feature_len = len

    def set_cell_size(self, cell):
        self.cell_size = cell

    def check_pixel_no_exception(self, x, y, img, cell_size):
        height, width = img.shape
        if x>round(cell_size//2) and x< round(height-cell_size//2)-1 and y>round(cell_size//2) and y< round(width - cell_size//2)-1:
            return True
        else:
            return False

    def create_HOG_feature_Map(self):
        height, width = self.feature_Map.get_direction_Mat().shape
        self.HOG_feature_list = []

        direction_Mat = self.feature_Map.get_direction_Mat()
        magnitude_Mat = self.feature_Map.get_magnitude_Mat()
        feature_list = self.moravec_Map.get_feature_list()
        for feature in feature_list:
            y, x = feature
            if self.check_pixel_no_exception(y, x, direction_Mat, cell_size=self.cell_size):
                HOG_feature_vector = np.zeros(self.feature_len)
                for i in range(-self.cell_size//2, self.cell_size//2 + 1):
                    for j in range(-self.cell_size // 2, self.cell_size // 2 + 1):
                        HOG_feature_vector[int(direction_Mat[y + i, x + j]//(360/self.feature_len))] += magnitude_Mat[y + i, x + j]
                self.HOG_feature_list.append((y, x, HOG_feature_vector))

    def get_HOG_feature_Map(self):
        return self.HOG_feature_list

    def compute_all(self, vector_len):
        self.set_feature_vector_len(vector_len)
        self.set_cell_size(13)
        self.create_HOG_feature_Map()
        return self.get_HOG_feature_Map()