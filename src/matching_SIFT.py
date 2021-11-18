import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time

img1 = cv.imread('../img/1.png') # queryImage
img2 = cv.imread('../img/2.png') # trainImage

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
print(len(kp1))
print(len(kp1))
#bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
bf = cv.BFMatcher()

tic1 = time.clock()
#matches = bf.match(des1,des2)
matches = bf.knnMatch(des1,des2,k=2)
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
toc1 = time.clock()
print(toc1-tic1)
print(len(matches))

#img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10], None,flags=2)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good[:10], None,flags=2)
plt.imshow(img3),plt.show()

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

#keypoints
sift = cv.xfeatures2d.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

img_1 = cv.drawKeypoints(gray1,keypoints_1,img1)
plt.imshow(img_1)
