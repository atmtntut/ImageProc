import cv2
import numpy as np
from matplotlib import pyplot as plt

top, bot, left, right = 100, 100, 0, 500
img1 = cv2.imread('a.jpg')
img2 = cv2.imread('b.jpg')
srcImg = cv2.copyMakeBorder(img1, top, bot, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
testImg = cv2.copyMakeBorder(img2, top, bot, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

#img1gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
#img2gray = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)
#sift = cv2.xfeatures2d.SIFT_create()
plt.subplot(121), plt.imshow(srcImg)
plt.subplot(122), plt.imshow(testImg)

plt.show()
