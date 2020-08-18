import matplotlib.pylab as plt
import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

count = 0
fontScale = 3
color = (0, 255, 0)
thickness = 10

lower = np.array([0,50,200])
upper = np.array([100,255,255])

img = cv2.imread("ex5.jpg")
output = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = cv2.inRange(img, lower, upper)
blurImg = cv2.GaussianBlur(img, (3,3), 0)

kernel = np.ones((10, 10),np.uint8)
img_morp = cv2.morphologyEx(blurImg, cv2.MORPH_OPEN, kernel)
img_morp = cv2.morphologyEx(img_morp, cv2.MORPH_CLOSE, kernel)

circles = cv2.HoughCircles(img_morp, cv2.HOUGH_GRADIENT, 1, 60, 
                           param1=50, param2=30, minRadius=150, maxRadius=600)
detected_circle = np.uint16(np.around(circles))
for (x,y,r) in detected_circle[0,:]:
    cv2.circle(output, (x,y), r, (0, 255, 0), 3)
    cv2.circle(output, (x,y), 2, (0, 255, 0), 1)
    count = count + 1
cv2.putText(output, 'Count: ' + str(count) , (0, 100), font,  fontScale, color, thickness, cv2.LINE_AA)
#cv2.imshow("Circle",output)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#plt.imshow(cv2.cvtColor(blurImg, cv2.COLOR_BGR2RGB))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))