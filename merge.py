import os
from PIL import Image
import sys
from collections import namedtuple
Info = namedtuple('Info', 'id row col x y z file')

infos = []
def getTimeFromDir(path):
    ret = os.walk(path)
    for _,_,files in ret:
        first = os.path.getctime(files[0])
        times = [os.path.getctime(f)-first for f in files if f!='draw.py']
    return times

def getDataFromDir(path):
    ret = os.walk(path)

    for _,_,files in ret:
        for file in files:
            l = file[:-4].split('_')
            if len(l)>6:
                info = Info(int(l[1]), int(l[2][1:]), int(l[3][1:]), int(l[4]), int(l[5]), int(l[6]), file)
                infos.append(info)
                #print(info)

def getSize(file):
    (x, y) = (0, 0)
    with Image.open(file) as im:
        (x ,y) = im.size
    return (x ,y)

def zoom(im, scale):
    (x,y) = im.size
    x = int(x*scale)
    y = int(y*scale)
    return im.resize((x,y), Image.ANTIALIAS)

def getCount():
    (rows,cols) = zip(*[(info.row,info.col) for info in infos])
    return (max(rows)+1, max(cols)+1)
            
if __name__ == '__main__':
    if len(sys.argv)>1:
        path = sys.argv[1]
        getDataFromDir(path)
    else:
        getDataFromDir('.')
    
    (row, col) = getCount()
    (width, height) = getSize(infos[0].file)
    width = width // 10
    height = height // 10

    bg = Image.new('RGBA',(col*width, row*height))

    for info in infos:
        pic = Image.open(info.file)
        tpic = zoom(pic, 0.1)
        pos = ((col-1-info.col)*width, (row-1-info.row)*height)
        #pos = (info.col*width, info.row*height)
        #pos = (info.row*height, info.col*width)
        bg.paste(tpic, pos)
        pic.close()
        print('info: id:{}, row:{}, col:{}, pos:{}'.format(info.id, info.row, info.col, pos))

    bg.save('merged.png')


