# OpenCV-inRange-Picker

This repository is an example code of cv2.inRange with trackbar that easy to see the preview in real-time

![alt text](https://github.com/earthsaharat/OpenCV-inRange-Picker/blob/master/inRange.gif "Demo")

## Installation

Download only file `inRange.py`

### Requirement
- Python3
- OpenCV
- NumPy

### Run
Run like a general python file

## Configuration

### Change preview image
```python
org = cv2.imread('color.jpg')
```
You can change `color.jpg` to your image that you want

### Scaling preview image
```python
scale = 1
```
You can change `1` to any scale that you want (scale up or down)

For example, if image size is 100x100
- scale = 2 -> image size is 200x200
- scale = 1 -> image size is 100x100
- scale = 0.5 -> image size is 50x50
- scale = 0.2 -> image size is 20x20

### Separate config menu from the preview
```python
configWindowName = 'img'
```
You just change `'img'` to another name (any name)

## Practical use
```python
# Convert an input image from BGR image to HSV image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Change upper and lower to the value that you got from the picker 
# These arrays are ordered in [H,S,V]
lower = np.array([130, 128, 0]) 
upper = np.array([255, 255, 255])

# The result image is called "mask"
# mask is a binary image (contain value only 0 and 255)
mask = cv2.inRange(hsv, lower, upper)

# If you want the result image that contain original image
result = cv2.bitwise_and(img, img, mask = mask) 
```
