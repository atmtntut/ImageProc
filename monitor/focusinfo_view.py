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
    width = 800
    height = 800
    ip = '192.168.10.139'
    port = 9001
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QGridLayout()

        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)

        layout.addWidget(self.canvas)

        self.resize(self.width, self.height)
        self.move(300, 300)
        self.setWindowTitle('Monitor')
        self.setLayout(layout)
        self.show()

    def update_curve(self, f):
        coststrs = ['cost0', 'cost1', 'cost2', 'cost3', 'cost4', 'cost5', 'cost6', 'cost7', 'cost8', 'cost9', 'cost10', 'cost11', 'cost12', 'cost13', 'cost14']
        try:
            with open(f) as fp:
                dat = json.loads(fp.read())

            for i,cost in enumerate(dat['FocusInfo']['curve']):
                datas = []
                labels = []
                for coststr in coststrs:
                    x,y = self.calc_curvedata(cost, coststr)
                    if y is None:
                        continue
                    elif y.max() != 0:
                        labels.append(coststr)
                        datas.append((x, y))

                l = []
                for x, y in datas:
                    plt.subplot(len(dat['FocusInfo']['curve']), 1, i+1)
                    plt.subplots_adjust(bottom=0.0, top=0.95, hspace=0.4)
                    tl, = plt.plot(x, y, '-o')
                    l.append(tl)
                    plt.legend(l, coststr, loc = 'upper right')
            self.canvas.draw()
        except Exception as e:
            print(e)

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
    if len(sys.argv)>1:
        f = sys.argv[1]
        app = QApplication(sys.argv)

        d = Demo()
        d.update_curve(f)
        sys.exit(app.exec_())
    else:
        print('usge: python3 focusinfo_view.py test_focus_err.json')
