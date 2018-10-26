import numpy as np
import os
import sys
import matplotlib.pyplot as plt

with open('data.txt', 'r') as fp:
    #fig = plt.figure()
    for l in fp.readlines():
        info = l.split('[')
        if len(info)<2:
            continue
        name = info[0]
        data = [float(x) for x in info[1][:-2].split(',')]
        maxdata = max(data)
        data = [x/maxdata for x in data]
        index = list(range(len(data)))
        print(name)
        print(type(data))
        plt.plot(index, data, linewidth=0.5 ,label=name)
        plt.legend()
    #ax = fig.add_subplot(131)
    #ax.imshow(img[:, :, 0])
    #ax = fig.add_subplot(132)
    #ax.imshow(img[:, :, 1], cmap='gray')
    #ax = fig.add_subplot(133)
    #ax.imshow(gray, cmap='gray')
    #plt.imshow(gray, cmap='gray')
    #index = list(range(0, len(evas)))
    #plt.plot(index, evas)
        plt.show()

