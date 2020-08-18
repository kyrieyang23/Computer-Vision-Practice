import matplotlib.pylab as plt
import cv2
import numpy as np

area = []
font = cv2.FONT_HERSHEY_SIMPLEX

count = 0
fontScale = 0.7
color = (255, 0, 0)
thickness = 2



lower = np.array([5,0,0])
upper = np.array([255,255,255])

img = cv2.imread("ex2.jpg")
output = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = cv2.inRange(img, lower, upper)
blurImg = cv2.GaussianBlur(img, (3,3), 0)

kernel = np.ones((3, 3),np.uint8)
img_morp = cv2.morphologyEx(blurImg, cv2.MORPH_OPEN, kernel)
img_morp = cv2.morphologyEx(img_morp, cv2.MORPH_CLOSE, kernel)
img_morp = cv2.erode(img, kernel)
kernel = np.ones((10, 10),np.uint8)
img_morp = cv2.morphologyEx(blurImg, cv2.MORPH_OPEN, kernel)
img_morp = cv2.morphologyEx(img_morp, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((5, 5),np.uint8) 
img_morp = cv2.morphologyEx(img_morp, cv2.MORPH_GRADIENT, kernel)
img_morp = cv2.erode(img, kernel)

contours, hierarchy = cv2.findContours(img_morp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect

        if cv2.contourArea(c) > 200:
            cv2.drawContours(output, contours, -1, (0,0,0), 1)
            cv2.rectangle(output, (x, y), (x+w, y+h), (0,255,0), 2)
            area.append(np.pi*((w*3)/2)*((h*3)/2))
cv2.putText(output, 'Area1: ' + str(round(area[0])) + " mm^2" , (0, 20), font,  fontScale, color, thickness, cv2.LINE_AA)
cv2.putText(output, 'Area2: ' + str(round(area[1])) + " mm^2", (0, 50), font,  fontScale, color, thickness, cv2.LINE_AA)
cv2.putText(output, 'Area3: ' + str(round(area[2])) + " mm^2", (0, 80), font,  fontScale, color, thickness, cv2.LINE_AA)

cv2.imwrite('ex2A.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 50])
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))