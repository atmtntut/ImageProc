import time
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import math

def calcConerSift(gray1, gray2):
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    return kp1, des1, kp2, des2

def calcConerSurf(gray1, gray2):
    surf = cv2.xfeatures2d.SURF_create(400)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(gray1, None)
    kp2, des2 = surf.detectAndCompute(gray2, None)
    return kp1, des1, kp2, des2

def calcConerOrb(gray1, gray2):
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    return kp1, des1, kp2, des2

def knnMatch(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.knnMatch(des1, des2, k=1)
    return matches

def drawKnnMatch(gray1, kp1, gray2, kp2, matches):
    img3 = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, matches, gray2, flags=2)
    plt.imshow(img3, ), plt.show()

def flannMatch(des1, des2):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return matches

def calcMatchMatrix(kp1, kp2, matches):
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.3*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            matchesMask[i] = [1, 0]

    #查询图像的特征描述子索引
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    #训练(模板)图像的特征描述子索引
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) 
    return M, matchesMask 

def drawFlannMatch(gray1, kp1, gray2, kp2, matches):
    M, matchesMask = calcMatchMatrix(kp1, kp2, matches)
    print(M)
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img3 = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()

def drawMatch(img1, img2):
    if not isinstance(img1, np.ndarray):
        img1 = np.array(img1)
    if not isinstance(img2, np.ndarray):
        img2 = np.array(img2)
    print(img1.shape)
    print(img2.shape)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1, kp2, des2 = calcConerSurf(gray1, gray2)
    matches = flannMatch(des1, des2)
    #M, matchesMask = calcMatchMatrix(kp1, kp2, matches)
    #R, X, Y = calcRT(M)
    drawFlannMatch(gray1, kp1, gray2, kp2, matches)

def calcMatrix(img1, img2):
    if not isinstance(img1, np.ndarray):
        img1 = np.array(img1)
    if not isinstance(img2, np.ndarray):
        img2 = np.array(img2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1, kp2, des2 = calcConerSurf(gray1, gray2)
    matches = flannMatch(des1, des2)
    M, matchesMask = calcMatchMatrix(kp1, kp2, matches)
    return M

def calcRT(Ma):
    X, Y = Ma[0][2], Ma[1][2]
    ret = -Y/X 
    r = math.degrees(math.atan(ret))
    print('r: {}, t: {},{}'.format(r, X, Y))
    return r, X, Y

def test(img1gray, img2gray):
    #sift
    start = time.clock()
    kp1, des1, kp2, des2 = calcConerSift(img1gray, img2gray)
    end = time.clock()
    print(end - start)
    #surf
    start = end
    kp1, des1, kp2, des2 = calcConerSurf(img1gray, img2gray)
    end = time.clock()
    print(end - start)
    #orb
    start = time.clock()
    kp1, des1, kp2, des2 = calcConerOrb(img1gray, img2gray)
    end = time.clock()
    print(end - start)

def templateMatch(target, tpl):
    th, tw = tpl.shape[:2]
    md = cv2.TM_CCOEFF
    res = cv2.matchTemplate(target, tpl, md)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(min_loc, max_loc)
    return (max_loc[0], min_loc[1]), (min_loc[0], max_loc[1])

def testTemplateMatch(img1, img2):
    tpl = img2
    target = img1
    #methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    th, tw = tpl.shape[:2]
    for method in methods:
        bgn = time.time()
        md = eval(method)
        res = cv2.matchTemplate(target, tpl, md)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print(cv2.minMaxLoc(res))
        if md in [cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF]:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0]+tw, tl[1]+th)   #br是矩形右下角的点的坐标
        cv2.rectangle(target, tl, br, (0, 255, 0), 5)
        end = time.time()
        print(f'run time: {end-bgn}s')

        plt.subplot(121)
        plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(122)
        plt.imshow(target,cmap = 'gray')
        plt.title('Detected Point')
        plt.xticks([])
        plt.yticks([])
        plt.suptitle(method)
        plt.show()

if __name__ == '__main__':
    flag = False
    if len(sys.argv) == 4:
        f1 = sys.argv[1]
        f2 = sys.argv[2]
        flag = True
    elif len(sys.argv) == 3:
        f1 = sys.argv[1]
        f2 = sys.argv[2]
    else:
        files = (('a.jpg','b.jpg'),
            ('test.jpg','testR10ST.jpg'))
        f1, f2 = files[1]

    #top, bot, left, right = 0, 500, 0, 200
    #('test.jpg','testaaa.jpg'),
    #for f1, f2 in files:
    #    Ma = calcMatrix(f1, f2)
    #    R, X, Y = calcRT(Ma)

    img1 = cv2.imread(f1)
    h1,w1 = img1.shape[0:2]
    print(img1.shape)
    img2 = cv2.imread(f2)
    h2,w2 = img2.shape[0]//10, img2.shape[1]//10
    img2 = cv2.resize(img2, (w2, h2))
    #print(type(img1))
    #if flag:
    #drawMatch(img1, img2)
    minPos, maxPos = templateMatch(img1, img2)
    fPhySizePerPix = 479.8/1000
    print((h1+h2)/2)
    print((w1+w2)/2)
    print(maxPos)
    corrX = ((w1+w2)/2 - maxPos[0]) * fPhySizePerPix
    corrY = ((h1+h2)/2 - maxPos[1]) * fPhySizePerPix
    print(f'corrX: {corrX}, corrY: {corrY}')

    #testTemplateMatch(img1, img2)

    #else:
    #    Ma = calcMatrix(img1, img2)
    #    R, X, Y = calcRT(Ma)

    #    img = cv2.imread(f2)
    #    rows, cols = img.shape[:2]
    #    M1 = cv2.getRotationMatrix2D((cols/2, rows/2), -R/2, 1)
    #    img_c = cv2.warpAffine(img, M1, (cols, rows))

    #    img_corr = Image.fromarray(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB))
    #    img_dat = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #    img_m = Image.blend(img_corr, img_dat, 0.5)
    #    img_m.show()
    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.imshow(cv2.imread(f2))
    #plt.subplot(1,2,2)
    #plt.imshow(res1, )
    #plt.show()
    #cv2.imshow('res1', res1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

