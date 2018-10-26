import cv2
import numpy as np
from matplotlib import pyplot as plt
filename = 'a.jpg'
##filename = 'test1.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = np.float32(gray)

#dst = cv2.cornerHarris(gray, 2,3,0.04)
#dst = cv2.dilate(dst, None)
#
#img[dst>0.02*dst.max()] = [0, 0, 255]
#
#cv2.imshow('dst', img)
#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()

#corners = cv2.goodFeaturesToTrack(gray, 25, 0.08, 10)
#for i in corners:
#    x,y = i.ravel()
#    cv2.circle(img, (x,y), 3, 255, -1)

#sift = cv2.xfeatures2d.SIFT_create()
#kp, des = sift.detectAndCompute(gray, None)
#dst = cv2.drawKeypoints(gray, kp, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#surf = cv2.xfeatures2d.SURF_create(5000, upright=True)
#kp, des = surf.detectAndCompute(gray, None)
#print(len(kp))
#dst = cv2.drawKeypoints(gray, kp, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#star = cv2.xfeatures2d.StarDetector_create()
#kp = star.detect(gray, None)
#dst = cv2.drawKeypoints(gray, kp, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(gray, None)
#dst = cv2.drawKeypoints(gray, kp, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
dst = cv2.drawKeypoints(gray, kp, np.array([]))
print(len(kp))

fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(gray)
ax = fig.add_subplot(122)
ax.imshow(dst)
plt.show()


#a = cv2.imread('a.jpg')
#b = cv2.imread('b.jpg')
#
##生成高斯图像金字塔，共6层
#G = a.copy()
#gpA = [G]
#for i in range(6):
#    G = cv2.pyrDown(G)
#    gpA.append(G)
#print(len(gpA))
#
#G = b.copy()
#gpB = [G]
#for i in range(6):
#    G = cv2.pyrDown(G)
#    gpB.append(G)
#
#def sameSize(imgBig, imgLitter):
#    rows, cols, dpt = imgLitter.shape
#    dst = imgBig[:rows,:cols]
#    return dst
##生成拉普拉斯金字塔
#lpA = [gpA[5]]
#for i in range(5, 0, -1):
#    GE = cv2.pyrUp(gpA[i])
#    print(GE.shape)
#    print(gpA[i-1].shape)
#    L = cv2.subtract(gpA[i-1], sameSize(GE,gpA[i-1]))
#    lpA.append(L)
#
#
##fig = plt.figure()
##ax = fig.add_subplot(121)
##ax.imshow(gpA[2])
##ax = fig.add_subplot(122)
##ax.imshow(lpA[5])
##plt.show()
#
#lpB = [gpB[5]]
#for i in range(5, 0, -1):
#    GE = cv2.pyrUp(gpB[i])
#    L = cv2.subtract(gpB[i-1], sameSize(GE,gpB[i-1]))
#    lpB.append(L)
#
#LS = []
#for la,lb in zip(lpA, lpB):
#    rows, cols, dpt = la.shape
#    print(la.shape)
#    print(lb.shape)
#    ls = np.hstack((la[:,:cols//2], lb[:,cols//2:]))
#    LS.append(ls)
#
#ls_ = LS[0]
#for i in range(1,6):
#    ls_ = cv2.pyrUp(ls_)
#    print(ls_.shape)
#    print(LS[i].shape)
#    ls_ = cv2.add(LS[i], sameSize(ls_, LS[i]))
#
#real = np.hstack((a[:, :cols//2], b[:, cols//2:]))
#cv2.imwrite('pyramid.jpg', ls_)
#cv2.imwrite('direct.jpg', real)
#
