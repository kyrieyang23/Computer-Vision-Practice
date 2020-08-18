import matplotlib.pylab as plt
import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

count = 0
fontScale = 0.7
color = (0, 255, 0)
thickness = 2

img = cv2.imread("ex4.jpg")
output = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray,5)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, 
                           param1=50, param2=30, minRadius=20, maxRadius=40)
detected_circle = np.uint16(np.around(circles))
for (x,y,r) in detected_circle[0,:]:
    cv2.circle(output, (x,y), r, (0, 0, 255), 3)
    cv2.circle(output, (x,y), 2, (0, 0, 255), 1)
    count = count + 1

cv2.putText(output, 'Count: ' + str(count) , (0, 20), font,  fontScale, color, thickness, cv2.LINE_AA)
#cv2.imshow("Circle",output)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))