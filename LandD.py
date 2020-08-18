#import matplotlib.pylab as plt
import cv2
import numpy as np

#image = cv2.imread('lane.png')

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
#    channel_count = img.shape[2]
    match_mask_color = 255 #* channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def process(image):
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    height = image.shape[0]
    width = image.shape[1]
    img_out = image.copy()
    
    region_of_interest_vertices = [
    (0, height),
    (width/2-200, height/2),
    (width/2, height/2),
    (width-100, height)
    ]
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 20, 150)
    cropped_image = region_of_interest(edges,np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image, 1, np.pi / 180, 100,minLineLength=100,maxLineGap=1000)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_out, (x1, y1), (x2,y2), (0, 255, 0), 3)
    return img_out

if __name__ == "__main__":
    cap = cv2.VideoCapture("lane.mp4")
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = process(frame)
        cv2.imshow("Lane Detection",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
#plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
#plt.show()
