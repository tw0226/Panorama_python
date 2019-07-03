import numpy as np
import cv2

img1, img2 = cv2.imread('girl1.jpg', cv2.IMREAD_COLOR), cv2.imread(cv2.IMREAD_COLOR)
gray_img1, gray_img2 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
