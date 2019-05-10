from rect_extra import *
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
index = 0
def threshold(gray):
    out = gray.copy()
    row, col = gray.shape
    T = 0.15
    s = 7
    for i in range(row):
        y1 = i - s
        y2 = i + s
        #if y1 < 0 or y2 >= row:
        #    continue
        if y1 < 0:
            y1 = 0
        if y2 >= row:
            y2 = row - 1

        for j in range(col):
            x1 = j - s
            x2 = j + s
            #if x1 < 0 or x2 >= col:
            #    continue
            if x1 < 0:
                x1 = 0
            if x2 >= col:
                x2 = col - 1
            count = (x2 - x1)*(y2 - y1)
            sum = int(gray[y2, x2]) + int(gray[y1, x1]) - int(gray[y1, x2]) - int(gray[y2, x1])
            #sum = gray[y2, x2] + gray[y1, x1] - gray[y1, x2] - gray[y2, x1]
            if sum < 0:
                sum = 256 + sum
            #print(f'{gray[i, j]} * {count}')
            #print(f'{sum} * {(1.0 - T)}')
            edge = 10
            if i < edge or i > row - edge:
                out[i, j] = 0
            elif j < edge or j > col - 80:
                out[i, j] = 0
            else:
                if (gray[i, j] < (sum * (1.0 - T))):
                    out[i, j] = 255
                else:
                    out[i, j] = 0
    return out

def extra_outline(gls, num):
    # 寻找物体的凸包并绘制凸包的轮廓
    for n in range(num):
        contours = cv2.findContours(gls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours = cv2.findContours(gls, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours[0]:
            hull = cv2.convexHull(cnt)
            length = len(hull)
            # 如果凸包点集中的点个数大于4
            if length > 4:
                # 绘制图像凸包的轮廓
                for i in range(length):
                    cv2.line(gls, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (255,0,0), 3)
            cv2.drawContours(gls, [cnt], -1, 255, cv2.FILLED)
        if False:
            cv2.imshow(f'gls{n}', gls)

    # 最后再将轮孔中内涂成白色
    contours = cv2.findContours(gls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(gls, contours[0], -1, 255, cv2.FILLED)
    return gls, contours[0]

def to_mask(img, row, col, level=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    gls = threshold(gray)
    # 膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
    gls = cv2.morphologyEx(gls, cv2.MORPH_DILATE, kernel)
    # 寻找凸包轮廓
    gls, cnts = extra_outline(gls, level)
    if True:
        global index
        for cnt in cnts:
            length = len(cnt)
            # 绘制图像凸包的轮廓
            for i in range(length):
                cv2.line(gray, tuple(cnt[i][0]), tuple(cnt[(i+1)%length][0]), (0,0,255), 1)
        cv2.imshow(f'gray{index}', gray)
        index += 1
    #print(gls.shape)
    mask = cv2.resize(gls, (int(row/gls.shape[0]*gls.shape[1]), row))
    diff_col = col - mask.shape[1]
    if diff_col > 0:
        mask = np.c_[mask, np.zeros((row, diff_col), np.uint8)]
    mask = np.clip(mask, 0, 1)
    #print(mask.shape)
    #cv2.imshow(f'mask', mask*255)
    return mask

if __name__=='__main__':
    images = glob.glob('image/gls*.jpg')
    for fname in images:
        #img = cv2.imread('image/gls0.jpg')
        img = cv2.imread(fname)
        mask = to_mask(img, 23, 54)
        cv2.imshow(f'mask{index-1}', mask*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
