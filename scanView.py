import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import time
import requests
import json
from PIL import Image

class Demo(QWidget):
    libcam = None
    lb = None
    w, h = 600, 400
    width = 0
    height = 0
    ip = '127.0.0.1'
    port = 9001
    cycleTime = 1000
    def __init__(self):
        super().__init__()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.operate)
        self.timer.start(self.cycleTime)
        self.initUI()

    def initUI(self):
        resp = requests.post(f'http://{self.ip}:{self.port}/getImage', data=json.dumps({'imageInfo':1}))
        self.width = resp.json()['imageWidth']
        self.height = resp.json()['imageHeight']
        self.w = self.width/2
        self.h = self.height/2
        self.lb = QLabel(self)
        self.lb.setGeometry(0,0,self.w,self.h)
        self.lb.setStyleSheet("border: 1px solid red")
        #lb.setPixmap(pix)

        self.resize(self.w, self.h)
        self.move(300, 300)
        self.setWindowTitle('ScanView')
        self.show()

    def operate(self):
        try:
            resp = requests.post(f'http://{self.ip}:{self.port}/getImage', data=json.dumps({'imageInfo':0}))
            img_dat = Image.frombytes(mode='RGB', size=(self.width, self.height), data = resp.content)
            img = QImage(img_dat.convert("RGBA").tobytes("raw", "BGRA"), img_dat.size[0], img_dat.size[1], QImage.Format_ARGB32)
            #pix = QPixmap.fromImage(img)
            pix = QPixmap.fromImage(img.scaled(QSize(self.w, self.h), Qt.IgnoreAspectRatio))
            self.lb.setPixmap(pix)
        except:
            print("http error")

if __name__ == '__main__':
    app = QApplication(sys.argv)

    d = Demo()
    sys.exit(app.exec_())
