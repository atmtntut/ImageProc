from PyQt5.QtWidgets import *
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
from sklearn import preprocessing
import sys
import json

class Demo(QWidget):
    width = 300
    height = 600
    def __init__(self, f):
        super().__init__()
        self.initUI(f)

    def initUI(self, f):
        layout = QVBoxLayout()
        #layout = QGridLayout()

        #draw curve
        #pw = pg.PlotWidget()
        #with open(f) as fp:
        #    dat = json.loads(fp.read())
        #    x, y = self.calc_curvedata(dat, 'cost10')
        #pw.plot(x, y)
        #layout.addWidget(pw)

        #draw mesh
        glw = gl.GLViewWidget()
        glw.setCameraPosition(distance=4)
        g = gl.GLGridItem()
        g.scale(1,1,1)
        glw.addItem(g)
        with open(f) as fp:
            dat = json.loads(fp.read())
            v, f, e = self.calc_meshdata(dat['FocusInfo'])
        md = gl.MeshData(v, f, e)
        m = gl.GLMeshItem(meshdata=md, smooth=False, drawFaces=False, drawEdges=True, edgeColor=(1,1,1,1))
        m.translate(0,0,0)
        glw.addItem(m)
        layout.addWidget(glw)

        self.resize(self.width, self.height)
        self.move(300, 300)
        self.setWindowTitle('Monitor')
        self.setLayout(layout)
        self.show()

    def calc_curvedata(self, dat, filter):
        x, y = zip(*[(c.get('ZPos'), c.get(filter)) for c in dat['costs'] if c.get('ZPos')])
        y = np.array(y)
        if y.max() != 0:
            y = y/y.max()
        return np.array(x),y

    def calc_meshdata(self, dat):
        v_b = [(p.get('X'), p.get('Y'), p.get('Z')) for p in dat['points'] if p.get('X')]
        v_t = [(p.get('focusX'), p.get('focusY'), p.get('focusZ')) for p in dat['points'] if p.get('focusX')]
        V = np.array(v_b + v_t)

        V = preprocessing.MinMaxScaler().fit_transform(V)
        #V = preprocessing.scale(V)*5
        print(V)
        v_num = V.shape[0]

        e = [(i,i+v_num//2) for i in range(0, v_num//2)]
        E = np.array(e)

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

if __name__ == '__main__':
    app = QApplication(sys.argv)

    d = Demo(sys.argv[1])
    sys.exit(app.exec_())
