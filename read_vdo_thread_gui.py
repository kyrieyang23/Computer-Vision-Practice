import sys, time, threading, cv2
import queue as Queue
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import numpy as np

IMG_FORMAT  = QImage.Format_RGB888
DISP_SCALE  = 4                 # Image scale
DISP_MSEC   = 50                # Display loops
VIDEO_PATH = "test.mp4"

camera_num  = 1
image_queue = Queue.Queue()
capturing_flag   = True

def grab_images(path, queue):
    cap = cv2.VideoCapture(path)
    while capturing_flag:
        if cap.grab():
            retval, image = cap.retrieve(0)
            if image is not None and queue.qsize() < 2:
                queue.put(image)
            else:
                time.sleep(DISP_MSEC / 1000.0)
        else:
            print("Error: can't grab camera image")
            break
    cap.release()


class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        self.setMinimumSize(image.size())
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()


class MyWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.central     = QWidget(self)
        self.layout      = QVBoxLayout()        # Window layout

        # Layout display
        self.layout_disp = QHBoxLayout()
        self.disp        = ImageWidget(self)
        self.layout_disp.addWidget(self.disp)

        # Layout menu
        self.layout_menu = QVBoxLayout()
        self.button1 = QPushButton('Start')
        self.button1.released.connect(self.on_button1_released)
        self.button2 = QPushButton('Stop')
        self.button2.released.connect(self.on_button2_released)
        self.layout_menu.addWidget(self.button1)
        self.layout_menu.addWidget(self.button2)

        self.layout.addLayout(self.layout_disp)
        self.layout.addLayout(self.layout_menu)
        self.central.setLayout(self.layout)
        self.setCentralWidget(self.central)

        exitAction = QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.triggered.connect(self.close)

    def start(self):
        self.timer = QTimer(self)           # Timer to trigger display
        self.timer.timeout.connect(lambda:
                    self.show_image(image_queue, self.disp, DISP_SCALE))
        self.timer.start(DISP_MSEC)
        self.capture_thread = threading.Thread(target=grab_images,
                    args=(VIDEO_PATH, image_queue))
        self.capture_thread.start()         # Thread to grab images

    def stop(self):
        self.timer.stop()

    # Queue > display
    def show_image(self, imageq, display, scale):
        if not imageq.empty():
            image = imageq.get()
            image = self.process(image)
            if image is not None and len(image) > 0:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.display_image(img, display, scale)

    # Display
    def display_image(self, img, display, scale=1):
        disp_size = img.shape[1]//scale, img.shape[0]//scale
        disp_bpl = disp_size[0] * 3
        if scale > 1:
            img = cv2.resize(img, disp_size,interpolation=cv2.INTER_CUBIC)
        qimg = QImage(img.data, disp_size[0], disp_size[1], disp_bpl, IMG_FORMAT)
        display.setImage(qimg)

    # Processing
    def process(self, img):
        # TODO


        return img

    def on_button1_released(self):
        self.start()
    def on_button2_released(self):
        self.stop()

    def closeEvent(self, event):
        global capturing_flag
        capturing_flag = False
        self.capture_thread.join()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    win.setWindowTitle("VDO")
    sys.exit(app.exec_())

#EOF
