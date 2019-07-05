import cv2
import numpy as np

class Moravec:
    def __init__(self, img):
        self.img = img
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.threshold = 1000
        self.feature_list = []

    def set_threshold(self, threshold):
        self.threshold = threshold

    def calculate_moravec(self):
        img = self.img_gray
        height, width = img.shape
        shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for y in range(2, height - 2):
            for x in range(2, width - 2):
                edge = list()
                min = 1e+8
                a = img[y - 1:y + 2, x - 1:x + 2]
                a = a.astype(np.float)
                for shift in shifts:
                    b = img[y-1+shift[0]:y+2+shift[0], x-1+shift[1]:x+2+shift[1]]
                    b = b.astype(np.float)
                    sum = np.sum((a-b)**2)
                    if min > sum:
                        min = sum
                if min > self.threshold:
                    self.feature_list.append((y, x))

    def get_feature_list(self):
        return self.feature_list

    def get_featureMap(self):
        height, width, _ = self.img.shape
        visual_img = self.img.copy()
        for feature_point in self.feature_list:
            y, x = feature_point
            visual_img = cv2.circle(visual_img, center=(x, y), radius=3, color=(0, 0, 255), thickness=-1)
        return visual_img

    def compute_all(self, moravec_threshold):
        self.set_threshold(moravec_threshold)
        self.calculate_moravec()
        return self.get_featureMap()