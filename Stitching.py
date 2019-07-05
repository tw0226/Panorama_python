import cv2
import numpy as np
from numpy.linalg import inv

class Stitch:
    def __init__(self, img1, img2, matching_points):
        self.img1 = img1
        self.img2 = img2
        self.matching_points = matching_points
        self.point_pair_map = None
        self.T = None

    def set_best_T(self, T):
        self.T = T

    def calculate_point_pair_match_Map(self):
        stitched_img = cv2.hconcat([self.img1, self.img2])
        for matching_points in self.matching_points:
            y1, x1 = matching_points[0]
            y2, x2 = matching_points[1]
            # print(x1,y1, x2, y2)
            cv2.line(stitched_img, (x1, y1), (self.img2.shape[0]+x2, y2), color=(255, 0, 0), thickness=1)
        self.point_pair_map = stitched_img

    def get_point_pair_Map(self):
        cv2.imshow("point pair map", self.point_pair_map)
        cv2.imwrite("point pair map.jpg", self.point_pair_map)
        cv2.waitKey(0)

    def create_panorama(self):
        empty_img = np.zeros((self.img1.shape), dtype="uint8")
        stitched_img = cv2.hconcat([self.img1, empty_img])

        for y in range(self.img2.shape[0]):
            for x in range(self.img2.shape[1]):
                new_points = np.matmul([x, y, 1], inv(self.T))
                new_points[new_points < 0] = 0
                new_y, new_x = int(new_points[0]), int(new_points[1])
                # print(new_y, new_x, y, x, self.img2[y][x])
                stitched_img[int(new_x), int(new_y), :] = self.img2[y][x][:]
        cv2.imshow("panorama", stitched_img)
        cv2.imwrite("panorama.jpg", stitched_img)
        cv2.waitKey(0)
    def compute_all(self):
        self.calculate_point_pair_match_Map()
        self.get_point_pair_Map()