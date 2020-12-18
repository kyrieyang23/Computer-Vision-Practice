import cv2
import numpy as np
import matplotlib.pyplot as plt

font = cv2.FONT_HERSHEY_SIMPLEX

count = 0
fontScale = 3
color = (0, 255, 0)
thickness = 10

img = cv2.imread("../Box_images/doc_632_0_0.jpg")

height = img.shape[0]
width = img.shape[1]

output = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = cv2.inRange(img, np.array([90, 20, 20]), np.array([140, 255, 255]))
blurImg = cv2.GaussianBlur(img, (3,3), 0)
kernel = np.ones((6, 6),np.uint8)
img_morp = cv2.morphologyEx(blurImg, cv2.MORPH_CLOSE, kernel)
edges = cv2.Canny(img_morp, 100, 250, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
r, theta = lines[4][0]
cos = np.cos(theta)
sin = np.sin(theta)
x0 = cos * r
y0 = sin * r
x1 = int(x0 + 100 * (-sin))
y1 = int(y0 + 100 * (cos))
x2 = int(x0 - 1000 * (-sin))
y2 = int(y0 - 1000 * (cos))
cv2.line(output, (x1, y1), (1000,y1), (255, 0, 0), 10)
cv2.line(output, (x1, y1), (x2,y2), (0, 0, 255), 10)
#cv2.line(output, (x1, y1), (x2,y1), (0, 0, 255), 10)
slope1 = (y2-y1)/(x2-x1)
slope2 = 0
angle = abs(np.arctan((slope1-slope2)/(1+(slope1*slope2))))
cv2.putText(output, 'Angle: ' + str(round((angle),2)) , (0, 100), font,  fontScale, color, thickness, cv2.LINE_AA)
imgplot = plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))