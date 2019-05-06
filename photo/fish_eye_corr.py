import cv2
assert cv2.__version__[0] == '3'
import numpy as np
import os
import glob

def get_K_and_D(w, h, imgsPath):
    CHECKERBOARD = (w, h)
    # 阈值
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
    objp = np.zeros((1, w*h, 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)

    _img_shape = None
    # 储存棋盘格角点的世界坐标和图像坐标对
    # 在世界坐标系中的三维点
    objpoints = [] 
    # 在图像平面的二维点
    imgpoints = [] 

    images = glob.glob(imgsPath + '/*.jpeg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        
        # 检测角点
        ret, corners = cv2.findChessboardCorners(gray, (w, h), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)

            # 将角点在图像上显示
            img = cv2.drawChessboardCorners(img, (w,h), corners, ret)
            cv2.imshow('findCorners',cv2.resize(img, (0, 0), fx=0.5, fy=0.5))
            cv2.waitKey()
    cv2.destroyAllWindows()

    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    # 开始标定
    rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
                                objpoints,
                                imgpoints,
                                gray.shape[::-1],
                                K,
                                D,
                                rvecs,
                                tvecs,
                                calibration_flags,
                                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                                )
    DIM = _img_shape[::-1]
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    
    return DIM, K, D

def undistort(img_path, DIM, K, D):
    img = cv2.imread(img_path)
    img = cv2.resize(img, DIM)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM,cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

    cv2.imshow('undistortImg',cv2.resize(undistorted_img, (0, 0), fx=0.5, fy=0.5))
    cv2.waitKey()
    cv2.destroyAllWindows()

    cv2.imwrite('unfisheyeImage.jpg', undistorted_img)

if __name__=='__main__':
    # 计算内参和矫正系数
    '''
    # checkerboard： 棋盘格的格点数目
    # imgsPath: 存放鱼眼图片的路径
    '''
    DIM, K, D = get_K_and_D(6,9, 'Image')
    undistort('Image/fisheye1.jpeg', DIM, K, D)


