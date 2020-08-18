import numpy as np
import cv2
import csv
from datetime import datetime


VIDEO_PATH = "test.mp4"
lower_blue = np.array([75,40,130])
upper_blue = np.array([140,255,255])

font = cv2.FONT_HERSHEY_SIMPLEX

fontScale = 1
color = (255, 0, 0)
thickness = 2

cap = cv2.VideoCapture(VIDEO_PATH)

breathData = []
high = []
count = 0
click = 0
volume_change = [0]
timestamp = [0]
resRate = [0]

def breath_count():
    global count
    global high
    global volume_change
    change = 0
    if (breathData[0] - min(breathData)) > 10 and (breathData[16] - min(breathData)) > 10:
        if min(breathData) == breathData[8] and breathData[8] != breathData[7] and breathData[8] != breathData[6]:
            count += 1
            change = (((breathData[0]*high[0]) - (breathData[8]*high[8]))/(breathData[8]*high[8]))*100
            volume_change.append(change)
            timestamp.append(datetime.now())
            resRate.append(count/60)
            print("breath count = " + str(count))
            print(breathData)
            print("bound = " + str(breathData[0] - min(breathData)) + " " + str(breathData[16] - min(breathData)))
            print("in = " + str(breathData[0]) + " " + str(min(breathData)) + " " + str(breathData[16]))
            print()
#       print(str(breathData[11] - breathData[8]) + " " + str(breathData[5] - breathData[8]))


def breath_update(new , newh):
    del breathData[0]
    breathData.append(new)
    del high[0]
    high.append(newh)


def Detect_img(img, preImg):
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
            cv2.circle(img_draw, (x, y), 1, (255,0,0), 2)
            cv2.circle(img_draw, (x, y+h), 1, (255,0,0), 2)

    return [img_draw,h,w]

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.inRange(img, lower_blue, upper_blue)
    blurImg = cv2.GaussianBlur(img, (3,3), 0)

    kernel = np.ones((40, 40),np.uint8)
    img_morp = cv2.morphologyEx(blurImg, cv2.MORPH_OPEN, kernel)
    img_morp = cv2.morphologyEx(img_morp, cv2.MORPH_CLOSE, kernel)

    im_floodfill = img_morp.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = img_morp.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = img_morp | im_floodfill_inv
    return im_out

if __name__ == '__main__':
    while True:
        ret,frame = cap.read()
        try:
            frame = frame[:,150:-100]
        except:
            break
        img = frame
        frame = preprocessing(frame)
#        cv2.imshow('frame', frame)
        result = Detect_img(img , frame)
        result[0] = cv2.putText(result[0], 'Count: ' + str(count) + " " + str(timestamp[count]), (50, 50), font,  fontScale, color, thickness, cv2.LINE_AA)
        result[0] = cv2.putText(result[0], 'Average respiratory rate: ' + str(resRate[count]), (50, 100), font,  fontScale, color, thickness, cv2.LINE_AA)
        result[0] = cv2.putText(result[0], '% of the volume changing: ' + str(volume_change[-1]) + "%", (50, 150), font,  fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('The Breath Watcher', result[0])
#        print(result[1])
        if len(breathData) < 17:
            breathData.append(result[1])
            high.append(result[2])
        if len(breathData) == 17:
            breath_update(result[1],result[2])
            breath_count()
#        print(breathData)
#        time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            print(breathData)
            print("bound = " + str(breathData[0] - min(breathData)) + " " + str(breathData[16] - min(breathData)))
            print("in = " + str(breathData[0]) + " " + str(min(breathData)) + " " + str(breathData[16]))
#            print(str(breathData[11] - breathData[8]) + " " + str(breathData[5] - breathData[8]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    with open('breath.csv', 'w', newline='') as f:
        thewriter = csv.writer(f)
        thewriter.writerow(timestamp)
        thewriter.writerow(resRate)
