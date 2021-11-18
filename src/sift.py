import numpy as np
import cv2 as cv

img = cv.imread('../img/2.png')
height, width, channels = img.shape
print(height, " ", width, " ", channels)
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)

img=cv.drawKeypoints(gray,kp,img)
#img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()
#cv.imwrite('sift_keypoints.jpg',img)