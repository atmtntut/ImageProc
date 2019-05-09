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

def edgea_modify(img, length):
    pass

img = cv2.imread('gls1.jpg')

#img = resize(img, 0.5)
#Imin, Imax = cv2.minMaxLoc(img)[:2]
#Imin, Imax = np.min(img), np.max(img)
#Omin, Omax = 0, 255
## 计算a和b的值
#a = float(Omax - Omin) / (Imax - Imin)
#b = Omin - a * Imin
#gls = a * img + b
#gls = gls.astype(np.uint8)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = gamma(gray, 5)
gray = cv2.equalizeHist(gray)

blurred = cv2.GaussianBlur(gray, (5, 5), 0) # 高斯模糊去噪（设定卷积核大小影响效果）
#_, gls = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY_INV) # 设定阈值165（阈值影响开闭运算效果）
_, gls = cv2.threshold(blurred, 236, 255, cv2.THRESH_BINARY_INV) # 设定阈值165（阈值影响开闭运算效果）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # 定义矩形结构元素
#closed = cv2.morphologyEx(gls, cv2.MORPH_CLOSE, kernel) # 闭运算（链接块）
#opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) # 开运算（去噪点）

gls = cv2.erode(gls, kernel, iterations=2)
gls = cv2.dilate(gls, kernel, iterations=3)

cnts = cv2.findContours(gls.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0]
cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)

plt.figure()
#plt.subplot(231), plt.imshow(closed, 'gray')
plt.subplot(232), plt.imshow(gls, 'gray')
plt.subplot(233), plt.imshow(gray, 'gray')
#plt.subplot(234), plt.imshow(thresh, 'gray')
plt.subplot(235), plt.hist(img.ravel(), 256)
plt.subplot(236), plt.hist(gray.ravel(), 256)
plt.show()
#cv2.imshow(f'gls', resize(gls, 0.5))
#cv2.imshow(f'gray', resize(gray, 0.5))
#cv2.imshow(f'closed', resize(closed, 0.5))
#cv2.imshow(f'opened', resize(opened, 0.5))
cv2.imshow(f'gls', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

