import os
import sys
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import namedtuple
Info = namedtuple('Info', 'id row col x y z')

infos = []
def calcTimeFromDir(path):
    outstrs = []
    ret = os.walk(path)
    for root,dir,files in ret:
        times = [os.path.getctime(os.path.join(root,f)) for f in files if f.find('.JPG')>0]
        if len(times)==0:
            continue
        base = min(times)
        times = sorted([t-base for t in times])
        totaltime = max(times)
        count = len(times)
        outstrs.append('{} count:{:>4}, time:{:>7.3f} s, {:>5.3f} p/sec, {:>7.3f} p/min, {:.0f}ms/p'.format(root, count, totaltime, count/totaltime, count/totaltime*60, totaltime/count*1000))
    outstrs.sort()
    for s in outstrs:
        print(s)

def getDataFromDir(path):
    ret = os.walk(path)

    for _,_,files in ret:
        for file in files:
            l = file[:-4].split('_')
            if len(l)>6:
                info = Info(int(l[1]), int(l[2][1:]), int(l[3][1:]), int(l[4]), int(l[5]), int(l[6]))
                infos.append(info)
                print(info)
            
def getDataFromLog():
    with open('scan.log', 'r') as fp:
        for i,line in enumerate(fp.readlines()):
            strs = re.finditer(r'[xyzij]:\d+', line)
            l = [i]
            for str in strs:
                str = str.group()[2:]
                l.append(str)
            info = Info(l[0], int(l[4]), int(l[5]), int(l[1]), int(l[2]), int(l[3]))
            infos.append(info)

def draw():
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')
    for info in infos:
        ax1.add_patch(
            patches.Rectangle(
                (info.x/(10**8), info.y/(10**8)),   # (x,y)
                1.2/100,          # width
                0.9/100,          # width
            )
        )
    plt.show()

def drawTime(times):
    index = [i for i in range(0, len(times))]
    plt.plot(index, times)
    plt.show()

if __name__ == '__main__':
    path = '.'
    if len(sys.argv)>1:
        path = sys.argv[1]

    times = calcTimeFromDir(path)
    #drawTime(times)
    #getDataFromLog()
    #draw()
