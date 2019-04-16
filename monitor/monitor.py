import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import requests
import json

class Demo(QWidget):
    width = 300
    height = 300
    ip = '192.168.10.139'
    port = 9001
    def __init__(self):
        super().__init__()
        self.initUI()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_curve)
        self.timer.start(300)
        self.img_timer = QTimer(self)
        self.img_timer.timeout.connect(self.update_img)
        self.img_timer.start(1000)

    def initUI(self):
        layout = QGridLayout()

        self.lb = QLabel(self)
        self.lb.setStyleSheet("border: 1px solid red")
        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)

        layout.addWidget(self.lb)
        layout.addWidget(self.canvas)

        try:
            resp = requests.post(f'http://{self.ip}:{self.port}/getImage', data=json.dumps({'imageInfo':1}))
            self.w = resp.json()['imageWidth']
            self.h = resp.json()['imageHeight']
            self.width, self.height = self.w//4, self.h//4
            print('w: {}, h: {}'.format(self.w, self.h))
            self.lb.resize(self.width, self.height)
        except Exception as e:
            print(e)

        #self.l = []
        #self.index = 0
        self.resize(self.width, self.height+400)
        self.move(300, 300)
        self.setWindowTitle('Monitor')
        self.setLayout(layout)
        self.show()

    def update_img(self):
        try:
            resp = requests.post(f'http://{self.ip}:{self.port}/getImage', data=json.dumps({'imageInfo':0}))
            img_dat = Image.frombytes(mode='RGB', size=(self.w, self.h), data = resp.content)
            img = QImage(img_dat.convert("RGBA").tobytes("raw", "BGRA"), img_dat.size[0], img_dat.size[1], QImage.Format_ARGB32)
            #pix = QPixmap.fromImage(img)
            pix = QPixmap.fromImage(img.scaled(QSize(self.width, self.height), Qt.IgnoreAspectRatio))
            self.lb.setPixmap(pix)
        except Exception as e:
            print(e)

    def update_curve(self):
        coststrs = ['cost0', 'cost1', 'cost2', 'cost3', 'cost4', 'cost5', 'cost6', 'cost7', 'cost8', 'cost9', 'cost10', 'cost11', 'cost12', 'cost13', 'cost14']
        try:
            resp = requests.post(f'http://{self.ip}:{self.port}/heartbeat', data=json.dumps({}))
            #print(resp.json())
            #fs = [ 'D1802198_103700000_27500000.json',
            #    'D1802198_98500000_22300000.json',
            #    'D1802198_98500000_32700000.json',
            #    'D1802198_93300000_27500000.json',
            #    'D1802198_98500000_27500000.json']
            #self.index = (self.index+1)%len(fs)
            #with open(fs[self.index]) as fp:
            #    dat = json.loads(fp.read())
            datas = []
            labels = []
            for coststr in coststrs:
                x,y = self.calc_curvedata(resp.json()['FocusInfo'], coststr)
                if y is None:
                    continue
                elif y.max() != 0:
                    labels.append(coststr)
                    datas.append((x, y))
            #x2,y2 = self.calc_curvedata(resp.json()['FocusInfo'], 'cost5')
            #x3,y3 = self.calc_curvedata(resp.json()['FocusInfo'], 'cost10')
            #x1,y1 = self.calc_curvedata(dat, 'cost0')
            #x2,y2 = self.calc_curvedata(dat, 'cost5')
            #x3,y3 = self.calc_curvedata(dat, 'cost10')
            #datas = [(x1, y1), (x2, y2), (x3, y3)]

            plt.subplot(1, 1, 1)
            plt.clf()
            l = []
            for x, y in datas:
                tl, = plt.plot(x, y, '-o')
                l.append(tl)
                #for i,tl in enumerate(self.l):
                #    tl, = plt.plot(x, y, '-o')
                #    self.l.append(tl)
                #    tl.set_data(datas[i][0], datas[i][1])
            #self.l2, = plt.plot(x2, y2, '-o')
            #self.l3, = plt.plot(x3, y3, '-o')
            plt.legend(l, coststr, loc = 'upper right')
            self.canvas.draw()
        except Exception as e:
            print(e)

    def draw_curve(self):
        #draw curve
        with open('cost.json') as fp:
            x,y = self.calc_curvedata(json.loads(fp.read()), 'cost10')
            plt.subplot(1, 1, 1)
            l, = plt.plot(x, y, '-o')
            plt.legend([l], ['cost10'], loc = 'upper right')
            self.canvas.draw()

    def calc_curvedata(self, dat, filter):
        if len(dat.get('costs'))>1:
            x, y = zip(*[(c.get('ZPos'), c.get(filter)) for c in dat['costs'] if c.get('ZPos')])
            y = np.array(y)
            if y.max() != 0:
                y = y/y.max()
            return np.array(x),y
        #print("return None")
        return None, None

    def calc_meshdata(self, dat):
        v_b = [(p.get('X'), p.get('Y'), p.get('Z')) for p in dat['points'] if p.get('X')]
        v_t = [(p.get('focusX'), p.get('focusY'), p.get('focusZ')) for p in dat['points'] if p.get('focusX')]
        V = np.array(v_b + v_t)

        V = preprocessing.MinMaxScaler().fit_transform(V)
        v_num = V.shape[0]
        return V

if __name__ == '__main__':
    app = QApplication(sys.argv)

    d = Demo()
    sys.exit(app.exec_())
