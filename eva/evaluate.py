import numpy as np
import os
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import functools
import collections
from ctypes import *

def print_func_time(function):
    @functools.wraps(function)
    def func_time(*args, **kwargs):
        t0 = time.clock()
        result = function(*args, **kwargs)
        t1 = time.clock()
        print("{} running time: {:.5f} s".format(function, t1 - t0))
        return result
    return func_time

@print_func_time
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

@print_func_time
def tenegrad(gray):
    #tenegrad 标准
    row, col = gray.shape
    print(gray.shape)
    S=0.0;
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Gy = np.array([[ 1, 2, 1],
                   [ 0, 0, 0],
                   [-1,-2,-1]])
    for x in range(1, row-1):
        for y in range(1, col-1):
            #当前邻域：
            #pre_row[y-1],pre_row[y],pre_row[y+1];
            #cur_row[y-1],cur_row[y],cur_row[y+1];
            #nex_row[y-1],nex_row[y],nex_row[y+1];
            #Gx =-1,0,1       Gy =1, 2, 1
            #    -2,0,2           0, 0, 0
            #    -1,0,1          -1,-2,-1
            Sx = (Gx * gray[x-1:x+2, y-1:y+2]).sum()
            Sy = (Gy * gray[x-1:x+2, y-1:y+2]).sum()
            S += Sx**2 + Sy**2
    return S/(row-2)/(col-2)

@print_func_time
def getImgFromDir(path):
    ret = os.walk(path)
    for root,dir,files in ret:
        imgs = [os.path.join(root,f) for f in files if f.find('.JPG')>0]
        if len(imgs)>0:
            break
    return sorted(imgs, key=lambda f: f[-12:-4])

@print_func_time
def calcEva(evas, gray, evaFunc):
    h,w = gray.shape
    graymap = (c_char*(h*w))()
    n = 0
    for i in gray.reshape(1,-1)[0]:
        graymap[n] = int(i)
        n+=1

    func = evaFunc.tenegrad
    func.restype = c_double
    eva = func(graymap, w, h, w)
    evas['tenegrad'].append(eva)

    func = evaFunc.laplace
    func.restype = c_double
    eva = func(graymap, w, h, w)
    evas['laplace'].append(eva)

    func = evaFunc.laplaceAdvance
    func.restype = c_double
    eva = func(graymap, w, h, w)
    evas['laplaceAdvance'].append(eva)

    func = evaFunc.energyVariance
    func.restype = c_double
    eva = func(graymap, w, h, w)
    evas['energyVariance'].append(eva)

    func = evaFunc.grayBrenner
    func.restype = c_double
    eva = func(graymap, w, h, w)
    evas['grayBrenner'].append(eva)

    func = evaFunc.grayEntropy
    func.restype = c_double
    eva = func(graymap, w, h, w)
    evas['grayEntropy'].append(eva)

    func = evaFunc.grayVariance
    func.restype = c_double
    eva = func(graymap, w, h, w)
    evas['grayVariance'].append(eva)

@print_func_time
def calcEvalutions(imgs, evaFunc):
    evas = collections.defaultdict(list)
    for f in imgs:
        img = mpimg.imread(f)
        gray = rgb2gray(img)
        #print(gray.shape)
        calcEva(evas, gray, evaFunc)

    with open('data.txt', 'a+') as fp:
        for k,v in evas.items():
            fp.write(k+'\n')
            fp.write(f'{v}'+'\n')

if __name__ == '__main__':
    evaFunc = CDLL('./eva.so')
    if len(sys.argv)>1:
        imgs = getImgFromDir(sys.argv[1])
    else:
        imgs = getImgFromDir('.')
    #drawEvalution(imgs, evaFunc.laplace)
    calcEvalutions(imgs, evaFunc)

