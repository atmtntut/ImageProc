import cv2
import numpy as np
from matplotlib import pyplot as plt

file1 = 'a.jpg'
file2 = 'b.jpg'
#file1 = 'test1.jpg'
#file2 = 'test2.jpg'
img1 = cv2.imread(file1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(file2)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#orb = cv2.ORB_create()
surf = cv2.xfeatures2d.SURF_create(5000, upright=True)
kp1, des1 = surf.detectAndCompute(gray1, None)
kp2, des2 = surf.detectAndCompute(gray2, None)

#bf = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=True)
#matches = bf.match(des1, des2)
#matches = sorted(matches, key = lambda x:x.distance)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

#matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
good = []
for i, (m, n) in enumerate(matches):
    if m.distance < 0.70*n.distance:
        good.append(m)
#        matchesMask[i] = [1, 0]

##查询图像的特征描述子索引
#src_pts = np.array([kp1[m.queryIdx].pt for m in good])
##训练(模板)图像的特征描述子索引
#dst_pts = np.array([kp2[m.trainIdx].pt for m in good])
#H, mask = cv2.findHomography(src_pts, dst_pts)
#h1, w1 = gray1.shape[:2]
#h2, w2 = gray2.shape[:2]
#shft=np.array([[1.0,0,w1],[0,1.0,0],[0,0,1.0]])
#M=np.dot(shft,H[0]) 

#查询图像的特征描述子索引
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#训练(模板)图像的特征描述子索引
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
print(src_pts.shape)
#M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
h1, w1 = gray1.shape[:2]
#warpImg = cv2.warpPerspective(gray1, M, (w1, h1*2), flags=cv2.WARP_INVERSE_MAP)
warpImg = cv2.warpPerspective(img1, np.array(M), (w1+500, h1*2)) #, flags=cv2.WARP_INVERSE_MAP)
warpImg[0:h1, 0:w1] = img2
warpImg = cv2.cvtColor(warpImg, cv2.COLOR_BGR2RGB)
plt.imshow(warpImg)
plt.show()

#draw_params = dict(matchColor=(0, 255, 0),
#                   singlePointColor=(255, 0, 0),
#                   matchesMask=matchesMask,
#                   flags=0)
#ret = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, matches, None, **draw_params)
#ret = cv2.drawMatches(gray1, kp1, gray2, kp2, matches[:50], np.array([]))
#plt.imshow(ret)
#plt.show()

#if len(good)>10:
    #src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    #dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    #M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #matchesMask = mask.ravel().tolist()
    #h,w, _ = img1.shape

    #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #dst = cv2.perspectiveTransform(pts,M)
    #cv2.polylines(img2,[np.int32(dst)],True,255,10, cv2.LINE_AA)

    #draw_params = dict(matchColor=(0, 255, 0),
    #                singlePointColor=None,
    #                matchesMask=matchesMask,
    #                flags=2)
    #ret = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    #plt.imshow(ret)
    #plt.show()