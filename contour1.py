# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:31:12 2020

@author: ThanaphatJ
"""

import numpy as np
import cv2

color_drawContour = (0, 0, 0)
color_drawRect = (0, 0, 255)

lower_ํYellow = np.array([100, 50, 100])
upper_Yellow = np.array([250, 280, 360])
count_yellow = 0
radius = 1

def draw_text_on_image(img_draw, count_yellow):
    # rectangle(img_draw, (200, 0), (350, 90), (0,0,0), -1)

    cv2.putText(img_draw,'Pink Count : ' + str(count_yellow), 
        (10,50),                  # bottomLeftCornerOfText
        cv2.FONT_HERSHEY_SIMPLEX, # font 
        1.0,                      # fontScale
        (0, 0, 0),            # fontColor
        4)                        # lineType
    return img_draw

def dectectPostIt(img, lower_clolor, upper_color):
    global count_yellow
    
    mask = cv2.inRange(img_hsv, lower_clolor, upper_color)
    
    img_gaussian = cv2.GaussianBlur(mask, (3, 3), 0)
    
    kernel = np.ones((6, 6),np.uint8)
    img_morp = cv2.morphologyEx(img_gaussian, cv2.MORPH_CLOSE, kernel)
    img_morp = cv2.morphologyEx(img_morp, cv2.MORPH_OPEN, kernel)
    
    img_edge = cv2.Canny(img_morp, 20, 50)
    
    contours, hierarchy = cv2.findContours(img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_draw = img.copy()
    
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect

        if cv2.contourArea(c) > 600:
            cv2.drawContours(img_draw, contours, -1, color_drawContour, 1)
            cv2.rectangle(img_draw, (x, y), (x+w, y+h), color_drawRect, 2)
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(img_draw, (cx, cy), radius, color_drawRect, 2)
            count_yellow += 1
    
    img_draw = draw_text_on_image(img_draw, count_yellow)
    
    cv2.imshow('Original', img)
    cv2.imshow('mask', img_morp)
    cv2.imshow('img_draw', img_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    path = 'img.jpg'
    
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    img = cv2.resize(img,None,fx=0.5,fy=0.5)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    dectectPostIt(img, lower_ํYellow, upper_Yellow)
    



    