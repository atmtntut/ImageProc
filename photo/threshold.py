from rect_extra import *
import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma(img, gama):
    fi = img/255.0
    ret = np.power(fi, gama)
    ret = ret*255
    ret = ret.astype(np.uint8)
    return ret

def threshold(gray):
    out = gray.copy()
    row, col = gray.shape
    #S = max(row, col) // 16
    T = 0.15
    #s = S // 2
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
            sum_ = gray[y2, x2] - gray[y1, x2] - gray[y2, x1] + gray[y1, x1]
            #print(f'{gray[i, j]} * {count}')
            #print(f'{sum_} * {(1.0 - T)}')
            edge = 10
            if i < edge or i > row - edge:
                out[i, j] = 0
            elif j < edge or j > col - 80:
                out[i, j] = 0
            else:
                if (gray[i, j] < (sum_ * (1.0 - T))):
                    out[i, j] = 255
                else:
                    out[i, j] = 0
            #break
    return out

img = cv2.imread('gls1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = gamma(gray, 5)
gls = threshold(gray)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
gls = cv2.morphologyEx(gls, cv2.MORPH_DILATE, kernel)

for n in range(2):
    # 寻找物体的凸包并绘制凸包的轮廓
    #contours = cv2.findContours(gls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cv2.findContours(gls, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = contours[0]
    for cnt in cnts:
        hull = cv2.convexHull(cnt)
        length = len(hull)
        # 如果凸包点集中的点个数大于5
        if length > 4:
            # 绘制图像凸包的轮廓
            for i in range(length):
                cv2.line(gls, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (255,0,0), 3)
    cv2.imshow(f'gls{n}', gls)

#contours = cv2.findContours(gls, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = contours[0]
for cnt in cnts:
    #print(cnt.shape)
    hull = cv2.convexHull(cnt)
    length = len(hull)
    # 如果凸包点集中的点个数大于5
    if length > 5:
        # 绘制图像凸包的轮廓
        for i in range(length):
            cv2.line(gray, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0,0,255), 2)
cv2.imshow('gray', gray)

cv2.imshow(f'gls', gls)
#cv2.imshow(f'gray', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
