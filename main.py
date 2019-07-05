import numpy as np
import cv2
import math
import HOG
import RANSAC
import Moravec
import Stitching
import Matching
from numpy.linalg import inv
img1, img2 = cv2.imread('girl1.jpg', cv2.IMREAD_COLOR), cv2.imread('girl2.jpg', cv2.IMREAD_COLOR)


def set_threshold(moravec_threshold, magnitude_threshold, feature_len, distance_threshold, loop, inlier_threshold, error_threshold):
    ### Moravec
    moravec_img1 = Moravec.Moravec(img1)
    moravec_img1.set_threshold(moravec_threshold)
    moravec_img1.calculate_moravec()
    moravec_feature_map1 = moravec_img1.get_featureMap()

    moravec_img2 = Moravec.Moravec(img2)
    moravec_img2.set_threshold(moravec_threshold)
    moravec_img2.calculate_moravec()
    moravec_feature_map2 = moravec_img2.get_featureMap()

    cv2.imshow("feature_map1", moravec_feature_map1)
    cv2.imshow("feature_map2", moravec_feature_map2)
    cv2.imwrite("feature_map1.jpg", moravec_feature_map1)
    cv2.imwrite("feature_map2.jpg", moravec_feature_map2)
    cv2.waitKey(0)

    ##Feature Map
    MatCollection_1 = HOG.MatCollection(img1)
    MatCollection_1.set_magnitude_threshold(magnitude_threshold)
    MatCollection_1.make_gradient_map()
    MatCollection_1.calculate_gradient_direction()
    MatCollection_1.calculate_gradient_magnitude()

    MatCollection_2 = HOG.MatCollection(img2)
    MatCollection_2.set_magnitude_threshold(magnitude_threshold)
    MatCollection_2.make_gradient_map()
    MatCollection_2.calculate_gradient_direction()
    MatCollection_2.calculate_gradient_magnitude()

    ## HOG
    HOG_feature_list_1 = HOG.HOG(MatCollection_1, moravec_img1).compute_all(vector_len=feature_len)
    HOG_feature_list_2 = HOG.HOG(MatCollection_2, moravec_img2).compute_all(vector_len=feature_len)

    ##Matching
    Matching_points = Matching.Matching(HOG_feature_list_1, HOG_feature_list_2, distance_threshold)
    Matching_points.matching_point_making()
    Matching_points = Matching_points.get_matching_points()

    #RANSAC

    Ransac1 = RANSAC.RANSAC(Matching_points)
    Ransac1.set_loop(loop)
    Ransac1.set_error_threshold(error_threshold)
    Ransac1.set_inlier_threshold(inlier_threshold)
    best_T = Ransac1.compute_RANSAC()

    print("best_T : ", best_T)
    stitching_imgs = Stitching.Stitch(img1, img2, Matching_points)
    stitching_imgs.compute_all()
    stitching_imgs.set_best_T(best_T)
    stitching_imgs.create_panorama()

if __name__=="__main__":
    moravec_threshold = 40000
    feature_len = 8
    magnitude_threshold = 200
    distance_threshold = 900
    loop = 10
    inlier_threshold = 100000
    error_threshold = 1000
    set_threshold(moravec_threshold, magnitude_threshold, feature_len, distance_threshold, loop, inlier_threshold, error_threshold)