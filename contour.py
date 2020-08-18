import numpy as np
import cv2
import matplotlib.pyplot as plt

def PostDetect(img, preImg):
    biImg = cv2.Canny(preImg, 20, 50)
    contours, hierarchy = cv2.findContours(biImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_draw = img.copy()
    
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect

        if cv2.contourArea(c) > 600:
            cv2.drawContours(img_draw, contours, -1, (0,0,0), 1)
            cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0,255,0), 2)
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(img_draw, (cx, cy), 1, (0,255,0), 2)
    return img_draw

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.inRange(img, np.array([100, 50, 100]), np.array([250, 280, 360]))
    blurImg = cv2.GaussianBlur(img, (3,3), 0)
    kernel = np.ones((6, 6),np.uint8)
    img_morp = cv2.morphologyEx(blurImg, cv2.MORPH_CLOSE, kernel)
    img_morp = cv2.morphologyEx(img_morp, cv2.MORPH_OPEN, kernel)
    return img_morp

if __name__ == "__main__":
    img = cv2.imread("img.jpg")
    preImg = preprocessing(img)
    drawImg = PostDetect(img,preImg)
    cv2.imshow("Input Image", img) 
    cv2.imshow("Preprocessing Image", preImg) 
    cv2.imshow("Draw image", drawImg) 
    cv2.waitKey() 
    cv2.destroyAllWindows()