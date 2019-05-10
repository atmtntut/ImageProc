#以灰度图读入
#腐蚀膨胀，闭合等操作
#二值化图像
#获取图像4个顶点
#形成变换矩阵
#透视变换
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

#def perspective_transf(src, dest, src_img):
def perspective_transf(box, src_img):
    w = math.sqrt((box[1][0]-box[2][0])**2 + (box[1][1]-box[2][1])**2)
    h = math.sqrt((box[0][0]-box[1][0])**2 + (box[0][1]-box[1][1])**2)
    src = np.float32(box)
    dest = np.float32([[0, h],[0,0],[w, 0],[w,h]])

    #生成透视变换矩阵
    M = cv2.getPerspectiveTransform(src, dest)
    #进行透视变换
    dst = cv2.warpPerspective(src_img, M, tuple(dest[-1]))
    return dst

def get_outline(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0) # 高斯模糊去噪（设定卷积核大小影响效果）
    _, RedThresh = cv2.threshold(blurred, 165, 255, cv2.THRESH_BINARY) # 设定阈值165（阈值影响开闭运算效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # 定义矩形结构元素
    closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel) # 闭运算（链接块）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel) # 开运算（去噪点）
    return img, gray, RedThresh, closed, opened

def box_rotate(box):
    pass

def get_vertex(img, opened, count):
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #a = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(a))
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # 计算最大轮廓的旋转包围盒
    #print(len(contours))
    boxs, draw_imgs = [], []
    for i in range(len(contours)):
        #print(f'i: {i}, count: {count}')
        if i >= count:
            break
        rect = cv2.minAreaRect(contours[i]) # 获取包围盒（中心点，宽高，旋转角度）
        box = np.int0(cv2.boxPoints(rect)) # box
        draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 3)
        if abs(box[0][0] - box[1][0]) > abs(box[0][1] - box[1][1]):
            box[[0,1,2,3], :] = box[[1,2,3,0], :]
        #print(f'{i}:\n{box}')
        boxs.append(box)
        draw_imgs.append(draw_img)
    return boxs,draw_imgs

def resize(img, per):
    return cv2.resize(img, (0, 0), fx=per, fy=per)

if __name__=='__main__':
    img = cv2.imread('corr.jpg')
    #img = cv2.imread('image/unfisheyeImage.jpg')
    #H_rows, W_cols= img.shape[:2]
    #H_rows, W_cols = 250, 500
    #print(H_rows, W_cols)

    ## 原图中书本的四个角点(左上、右上、左下、右下),与变换后矩阵位置
    #src_vertex = np.float32([[630, 873], [678, 594], [1314, 990], [1365, 717]])
    #dst_vertex = np.float32([[0, 0],[H_rows,0],[0, W_cols],[H_rows,W_cols]])

    per = 0.5
    _, gray, RedThresh, closed, opened = get_outline(img)
    boxs, draws = get_vertex(img, opened, 4)
    boxs.sort(key=lambda x: x[0][1], reverse=True)
    for i in range(len(boxs)):
        #print(boxs[i])
        dst = perspective_transf(boxs[i], img)
        cv2.imshow(f'result{i}', resize(dst,    per))
        cv2.imshow(f'draw{i}',   resize(draws[i],   per))
    #cv2.imshow("test",cv2.resize(img, (0,0), fx=0.5, fy=0.5))
    #cv2.imshow("result",cv2.resize(dst, (0,0), fx=0.5, fy=0.5))
    #cv2.imshow("test", img)
    #cv2.imshow("result", dst)
    #cv2.imshow('gray',   resize(gray,   per))
    #cv2.imshow('closed', resize(closed, per))
    #cv2.imshow('opened', resize(opened, per))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

