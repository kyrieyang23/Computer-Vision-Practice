import numpy as np
import cv2
import matplotlib.pyplot as plt

def PostDetect(img, preImg):
    l = []
    biImg = cv2.Canny(preImg, 50, 100)
    contours, hierarchy = cv2.findContours(biImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_draw = img.copy()
    print(len(contours))
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
#        cv2.drawContours(img_draw, contours, -1, (0,0,0), 1)
#        if cv2.contourArea(c) > 10000:
##            cv2.drawContours(img_draw, contours, -1, (0,0,0), 2)
#            cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0,0,255), 2)
##            img_draw = img_draw[y:y+h, x:x+w]
#            M = cv2.moments(c)
#            cx = int(M['m10']/M['m00'])
#            cy = int(M['m01']/M['m00'])
#            cv2.circle(img_draw, (cx, cy), 1, (0,0,255), 2)
        if (w*h) > 1000 and (w*h) < 7000 and (w/h) < 1.3 and (h/w) < 1.3:
            cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0,255,0), 2)
            l.append((x,y,w,h))
    return img_draw, l

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.inRange(img, np.array([114, 56, 109]), np.array([120, 255, 255]))
    blurImg = cv2.GaussianBlur(img, (3,3), 0)
    kernel = np.ones((3, 3),np.uint8)
    img_morp = cv2.morphologyEx(blurImg, cv2.MORPH_CLOSE, kernel)
#    img_morp = cv2.Canny(img_morp, 50, 100)
#    img_morp = cv2.morphologyEx(img_morp, cv2.MORPH_OPEN, kernel)
#    img_morp = cv2.morphologyEx(img_morp, cv2.MORPH_OPEN, kernel)
    return img_morp

if __name__ == "__main__":
    for i in range(120):
        img = cv2.imread("../Box_images/doc_"+ str(631 + i)+ "_0_0.jpg")
        preImg = preprocessing(img)
        drawImg = PostDetect(img,preImg)
        cv2.imwrite(str(631 + i)+".jpg",drawImg[0])
#    cv2.imshow("Input Image", img) 
#        cv2.imshow("Preprocessing Image"+ str(631 + i), preImg) 
#        cv2.imshow("Draw image"+ str(631 + i), drawImg) 
#    cv2.waitKey() 
#    cv2.destroyAllWindows()