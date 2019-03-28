import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import requests
import json
from PIL import Image

class Demo(QWidget):
    ip = '127.0.0.1'
    port = 9001
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        try:
            resp = requests.post(f'http://{self.ip}:{self.port}/getImage', data=json.dumps({'imageInfo':1}))
            self.width = resp.json()['imageWidth']
            self.height = resp.json()['imageHeight']
            self.w = self.width/2
            self.h = self.height/2
            self.lb = QLabel(self)
            self.lb.setGeometry(0,40,self.w,self.h)
            #self.lb.setStyleSheet("border: 1px solid red")
        except:
            print("http error")

        self.btn = QPushButton("Corr", self)
        self.btn.move(20, 20)
        self.btn.clicked.connect(self.corr)

        self.resize(300, 400)
        self.move(300, 300)
        self.setWindowTitle('CCDCorrect')
        self.show()

    def picture(self):
        img_dat = None
        try:
            resp = requests.post(f'http://{self.ip}:{self.port}/getImage', data=json.dumps({'imageInfo':0}))
            img_dat = Image.frombytes(mode='RGB', size=(self.width, self.height), data = resp.content)
            img = QImage(img_dat.convert("RGBA").tobytes("raw", "BGRA"), img_dat.size[0], img_dat.size[1], QImage.Format_ARGB32)
            pix = QPixmap.fromImage(img.scaled(QSize(self.w, self.h), Qt.IgnoreAspectRatio))
            self.lb.setPixmap(pix)
        except:
            print("http error")
        return img_dat

    def move(self, direct, length):
        resp = requests.post(f'http://{self.ip}:{self.port}/debug', data=json.dumps({'cmd':f'move {direct} {length}'}))

    def corr(self):
        img1 = self.picture()
        if img1 == None:
            print('picture img1 fail')
            return

        self.move('x', 1000000)

        img2 = self.picture()
        if img2 == None:
            print('picture img2 fail')
            return

        m = calc_matrix(img1, img2)
        img_corr = transf(m, img2)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    d = Demo()
    sys.exit(app.exec_())
