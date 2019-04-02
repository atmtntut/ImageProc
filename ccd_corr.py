import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import requests
import json
from image_match import *
from PIL import Image

class Demo(QWidget):
    ip = '127.0.0.1'
    port = 9001
    width = 300
    height = 300
    zoo = 3
    w = width / zoo
    h = height / zoo
    #0,x方向;1,y方向
    direct = 0
    delta = 300000
    x_step = 0
    y_step = 0
    img_corr = None
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        #layout = QGridLayout()
        self.btn = QPushButton("Corr", self)
        self.btn.move(10, 10)
        self.btn.clicked.connect(self.getCorrImg)

        self.cb = QComboBox(self)
        self.cb.addItem('X')
        self.cb.addItem('Y')
        self.cb.move(120, 10)

        self.txt = QTextEdit(self)
        self.txt.setGeometry(180, 10, 80, 30)

        self.out = QLabel(self)
        self.out.setGeometry(300,10, 350, 30)
        self.out.setText('log')

        try:
            resp = requests.post(f'http://{self.ip}:{self.port}/getImage', data=json.dumps({'imageInfo':1}))
            self.width = resp.json()['imageWidth']
            self.height = resp.json()['imageHeight']
            self.w = self.width/self.zoo
            self.h = self.height/self.zoo
            self.lb = QLabel(self)
            self.lb.setGeometry(0,60,self.w,self.h)
            #self.lb.setStyleSheet("border: 1px solid red")
        except:
            self.mprint("initUI http error")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.operate)
        self.timer.start(1000)

        self.resize(self.w, self.h+60)
        self.move(300, 300)
        self.setWindowTitle('CCDCorrect')
        self.show()

    def mprint(self, string):
        try:
            self.out.setText(string)
        except Exception as e:
            print(e)
        print(string)

    def operate(self):
        try:
            resp = requests.post(f'http://{self.ip}:{self.port}/getImage', data=json.dumps({'imageInfo':0}))
            img_dat = Image.frombytes(mode='RGB', size=(self.width, self.height), data = resp.content)
            if self.img_corr != None:
                img_dat = Image.blend(img_dat, self.img_corr, 0.5)
            img = QImage(img_dat.convert("RGBA").tobytes("raw", "BGRA"), img_dat.size[0], img_dat.size[1], QImage.Format_ARGB32)
            pix = QPixmap.fromImage(img.scaled(QSize(self.w, self.h), Qt.IgnoreAspectRatio))
            self.lb.setPixmap(pix)
        except:
            self.mprint("operate http error")

    def picture(self):
        img_dat = None
        try:
            resp = requests.post(f'http://{self.ip}:{self.port}/getImage', data=json.dumps({'imageInfo':0}))
            img_dat = Image.frombytes(mode='RGB', size=(self.width, self.height), data = resp.content)
            img = QImage(img_dat.convert("RGBA").tobytes("raw", "BGRA"), img_dat.size[0], img_dat.size[1], QImage.Format_ARGB32)
            pix = QPixmap.fromImage(img.scaled(QSize(self.w, self.h), Qt.IgnoreAspectRatio))
            self.lb.setPixmap(pix)
        except:
            self.mprint("picture http error")
        return img_dat

    def moveGls(self, direct, length):
        resp = requests.post(f'http://{self.ip}:{self.port}/debug', data=json.dumps({'cmd':f'move {direct} {length}'}))
        self.mprint(resp.json())

    def getCorrImg(self):
        img1 = self.picture()
        if img1 == None:
            self.mprint('picture img1 fail')
            return

        #time.sleep(1)
        try:
            ret = int(self.txt.getText())
            self.delta = ret
        except:
            print('fail delta')
        if self.cb.currentText()=='X':
            self.direct = 0
        else:
            self.direct = 1
        self.moveGls(self.cb.currentText(), self.delta)
        time.sleep(1)

        img2 = self.picture()
        if img2 == None:
            self.mprint('picture img2 fail')
            return

        #time.sleep(1)

        Ma = calcMatrix(img1, img2)
        R, X, Y = calcRT(Ma)
        if self.direct == 0:
            self.x_step = abs(int(self.width * self.delta / X))
        else:
            self.y_step = abs(int(self.height * self.delta / Y))
        self.mprint(f'R: {R:.4f}, T: {X:.2f}, {Y:.2f}, step: {self.x_step:d}, {self.y_step:d}')

        self.img_corr = img2.rotate(-R/2)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    d = Demo()
    sys.exit(app.exec_())
