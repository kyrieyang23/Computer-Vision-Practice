import cv2
import numpy as np

org = cv2.imread('Box_images/doc_741_0_0.jpg')

scale = 1
org = cv2.resize(org,None,fx=scale,fy=scale)

configWindowName = 'img'

cv2.namedWindow(configWindowName)
def callback(value): pass

cv2.createTrackbar('H_low'	,configWindowName	,0	,180,callback)
cv2.createTrackbar('H_high'	,configWindowName	,180,180,callback)
cv2.createTrackbar('S_low'	,configWindowName	,0	,255,callback)
cv2.createTrackbar('S_high'	,configWindowName	,255,255,callback)
cv2.createTrackbar('V_low'	,configWindowName	,0	,255,callback)
cv2.createTrackbar('V_high'	,configWindowName	,255,255,callback)

while True:
	img = org.copy()

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower = np.array([
		cv2.getTrackbarPos('H_low',configWindowName),
		cv2.getTrackbarPos('S_low',configWindowName),
		cv2.getTrackbarPos('V_low',configWindowName)])
	upper = np.array([
		cv2.getTrackbarPos('H_high',configWindowName),
		cv2.getTrackbarPos('S_high',configWindowName),
		cv2.getTrackbarPos('V_high',configWindowName)])
	mask = cv2.inRange(hsv, lower, upper)
	print(upper)

	img = cv2.bitwise_and(img, img, mask = mask)

	cv2.imshow('img',img)
	if cv2.waitKey(1) != -1:
		cv2.destroyAllWindows()
		break
