import numpy as np
from numpy.linalg import inv

class RANSAC:
    def __init__(self, matching_point_list):
        self.T = []
        self.matching_points = matching_point_list
        self.loop = 10
        self.inlier_threshold = 100000
        self.error_threshold = 1000
    def set_loop(self, loop):
        self.loop = loop
    def set_error_threshold(self, threshold):
        self.error_threshold = threshold
    def set_inlier_threshold(self, threshold):
        self.inlier_threshold = threshold


    def compute_RANSAC(self):
        best_T = None
        inlier = []
        for i in range(self.loop):
            random_number = np.random.choice(len(self.matching_points), 3, replace=True)
        a_matrix, b_matrix = [], []

        rest_of_matching_list = self.matching_points.copy()
        for index in sorted(random_number, reverse=True):
            del rest_of_matching_list[index]

        for random in random_number:
            y, x = self.matching_points[random][0]
            a_matrix.append([y, x, 1])
            y, x = self.matching_points[random][1]
            b_matrix.append([y, x, 1])
        T = self.calculate_T([a_matrix, b_matrix])

        for random in random_number:
            x, y = self.matching_points[random][0]
            x2, y2 = self.matching_points[random][1]
            inlier.append([[x, y, 1], [x2, y2, 1]])

        for matching_points in rest_of_matching_list:
            a_y, a_x = matching_points[0]
            b_y, b_x = matching_points[1]
            rest_a = np.array([a_x, a_y, 1])
            rest_b = np.array([b_x, b_y, 1])

            error = np.sum(np.power(np.dot(rest_a.T, T) - rest_b.T, 2))
            if error < self.inlier_threshold:
                inlier.append([rest_a, rest_b])

        if len(inlier) > int(len(matching_points) * 0.7):
            T = self.calculate_T(inlier)

        error = 0
        for matching_point in matching_points:
            a_y, a_x = matching_points[0]
            b_y, b_x = matching_points[1]
            rest_a = np.array([a_x, a_y, 1])
            rest_b = np.array([b_x, b_y, 1])
            error += np.sum(np.power(np.dot(rest_a.T, T) - rest_b.T, 2))
            if error < self.error_threshold:
                best_T = T
        return best_T
    def calculate_T(self, matching_points):
        A = np.zeros((6, 6))
        B = np.zeros((6, 1))
        matching_points = np.array(matching_points)
        key_points_a , key_points_b = matching_points[:, 0], matching_points[:, 1]
        # print(key_points_a, key_points_b)
        for point_a, point_b in zip(key_points_a, key_points_b):
            # print(point_a, point_b)
            y1, x1 = point_a[0], point_a[1]
            y2, x2 = point_b[0], point_b[1]
            A[0][0] += x1 * x1
            A[0][1] += x1 * y1
            A[0][2] += x1

            A[1][0] = A[0][1]
            A[1][1] += y1 * y1
            A[1][2] += y1

            A[2][0] = A[0][2]
            A[2][1] = A[1][2]
            A[2][2] += 1

            B[0] += x1 * x2
            B[1] += y1 * x2
            B[2] += x2
            B[3] += x1 * y2
            B[4] += y1 * y2
            B[5] += y2
        A[3:, 3:] = A[:3, :3]
        M = np.matmul(np.linalg.inv(A), B)
        t_11 = float(M[0])
        t_21 = float(M[1])
        t_31 = float(M[2])
        t_12 = float(M[3])
        t_22 = float(M[4])
        t_32 = float(M[5])

        T = np.array([[t_11, t_12, 0],
                      [t_21, t_22, 0],
                      [t_31, t_32, 1]])
        return T

