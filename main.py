import numpy as np
import cv2
import math
import HOG
import RANSAC
import Moravec
import Stitching
from numpy.linalg import inv
img1, img2 = cv2.imread('girl1.jpg', cv2.IMREAD_COLOR), cv2.imread('girl2.jpg', cv2.IMREAD_COLOR)

moravec_threshold = 40000
feature_len = 8
magnitude_threshold = 200

### Moravec
moravec_img1 = Moravec.Moravec(img1)
moravec_img1.set_threshold(moravec_threshold)
moravec_img1.calculate_moravec()
moravec_feature_map1 = moravec_img1.get_featureMap()

moravec_img2 = Moravec.Moravec(img2)
moravec_img2.set_threshold(moravec_threshold)
moravec_img2.calculate_moravec()
moravec_feature_map2 = moravec_img2.get_featureMap()

# cv2.imshow("feature_map1", moravec_feature_map1)
# cv2.imshow("feature_map2", moravec_feature_map2)
# cv2.waitKey(0)

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

matching_point_list = []
# sorted_matching_point_list = []
print(len(HOG_feature_list_1), len(HOG_feature_list_2))
for feature_1 in HOG_feature_list_1:
    y1, x1, vector_1 = feature_1
    min=1e+7
    matching_point = None
    for feature_2 in HOG_feature_list_2:
        y2, x2, vector_2 = feature_2
        distance = np.linalg.norm(vector_1 - vector_2)
        if distance < min:
            min = distance
            matching_point = ((y1, x1), (y2, x2), distance)
    if min < 1100:
        matching_point_list.append(matching_point)
matching_point_list = sorted(matching_point_list, key=lambda distance: distance[2])

print(len(matching_point_list))
print(matching_point_list[:10])
stitching_imgs = Stitching.Stitch(img1, img2, matching_point_list)
stitching_imgs.compute_all()
### RANSAC
loop = 10
sampling_number = 3
inlier_threshold = 10000
inlier_ratio = 0.7
matching_error = 100
# print(matching_point_list[:10])
best_T = RANSAC.RANSAC(matching_point_list).compute_RANSAC()

print("best_T : ", best_T)
stitching_imgs.set_best_T(best_T)
stitching_imgs.create_panorama()