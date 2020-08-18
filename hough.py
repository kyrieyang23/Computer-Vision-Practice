import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("test.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 100, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
for line in lines:
    r, theta = line[0]
    cos = np.cos(theta)
    sin = np.sin(theta)
    x0 = cos * r
    y0 = sin * r
    x1 = int(x0 + 200 * (-sin))
    y1 = int(y0 + 200 * (cos))
    x2 = int(x0 - 400 * (-sin))
    y2 = int(y0 - 400 * (cos))
    cv2.line(img, (x1, y1), (x2,y2), (0, 0, 255), 2)
imgplot = plt.imshow(img)