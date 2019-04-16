import PyQt5
from pyqtgraph.Qt import QtCore, QtGui
#import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import sys
import time
import json
from sklearn import preprocessing
import matplotlib.pyplot as plt

widget = None
m = None

def draw(vertexs, faces, edges):
    global m
    # wireframe
    md = gl.MeshData(vertexes, faces, edges)
    m = gl.GLMeshItem(meshdata=md, smooth=False, drawFaces=False, drawEdges=True, edgeColor=(1,1,1,1))
    m.translate(0,0,0)
    widget.addItem(m)

def calc_meshdata(dat):
    v_b = [(p.get('X'), p.get('Y'), p.get('Z')) for p in dat['points'] if p.get('X')]
    v_t = [(p.get('focusX'), p.get('focusY'), p.get('focusZ')) for p in dat['points'] if p.get('focusX')]
    V = np.array(v_b + v_t)

    V = preprocessing.MinMaxScaler().fit_transform(V)
    #V = preprocessing.scale(V)*5
    print(V)
    v_num = V.shape[0]

    e = [(i,i+v_num//2) for i in range(0, v_num//2)]
    E = np.array(e)
    print(E)

    F = np.array([
        [0,1,2],
        [0,2,3],
        [0,3,4],
        [0,1,4],
        [5,6,7],
        [5,7,8],
        [5,8,9],
        [5,6,9],
        ])
    return V, F, E

def calc_curvedata(dat, filter):
    x, y = zip(*[(c.get('ZPos'), c.get(filter)) for c in dat['costs'] if c.get('ZPos')])
    #x2, y2 = zip(*[(c.get('ZPos'), c.get('cost5')) for c in dat['costs'] if c.get('ZPos')])
    #x3, y3 = zip(*[(c.get('ZPos'), c.get('cost10')) for c in dat['costs'] if c.get('ZPos')])
    y = np.array(y)
    if y.max() != 0:
        y = y/y.max()
    return np.array(x),y


def update():
    return
    global m
    #md = gl.MeshData(v, f, None)
    #m.setMeshData(meshdata=md)

def gui_init():
    global widget
    widget = gl.GLViewWidget()
    widget.show()
    widget.setWindowTitle('ZJScan Focus Monitor')
    widget.setCameraPosition(distance=4)

    g = gl.GLGridItem()
    g.scale(1,1,1)
    widget.addItem(g)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    #app = QtGui.QApplication([])
    #gui_init()
    try:
        #with open('plane.json') as fp:
        #    dat = json.loads(fp.read())

        #vertexes, faces, edges = calc_meshdata(dat)
        #draw(vertexes, faces, edges)
        #fs = ['D1802198_103700000_27500000.json',
        #    'D1802198_98500000_22300000.json',
        #    'D1802198_98500000_32700000.json',
        #    'D1802198_93300000_27500000.json',
        #    'D1802198_98500000_27500000.json']
        fs = ['cost.json']
        plt.figure()
        for i,f in enumerate(fs):
            with open(f) as fp:
                dat = json.loads(fp.read())

            x1,y1 = calc_curvedata(dat, 'cost0')
            x2,y2 = calc_curvedata(dat, 'cost5')
            x3,y3 = calc_curvedata(dat, 'cost10')
            plt.subplot(len(fs), 1, i+1)
            plt.title(f)
            #l1, = plt.plot(x1,y1, '-o')
            #l2, = plt.plot(x2,y2, '-o')
            l3, = plt.plot(x3,y3, '-o')
            plt.grid(ls='--')
            #plt.legend([l1, l2, l3], ['cost0', 'cost5', 'cost10'], loc = 'upper right')
            plt.legend([l3], ['cost10'], loc = 'upper right')
        plt.show()
    except Exception as e:
        print(e)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(1000)

    #sys.exit(app.exec_())
