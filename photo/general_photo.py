from fisheye import *
from rect_extra import *
from camera import *
from threshold import *

def camera_calibr(chess_path):
    DIM, K, D = get_K_and_D(4,7, chess_path)
    return DIM, K, D

def take_photo(camera):
    #camera.set_led('on')
    camera.snap()
    time.sleep(0.5)
    img = camera.get_picture()
    #camera.set_led('off')
    return img

def corr_photo(img, DIM, K, D):
    img = undistort(img, DIM, K, D)
    return img

def get_gls(img):
    glses = []
    _, gray, RedThresh, closed, opened = get_outline(img)
    boxs, _ = get_vertex(img, opened, 4)
    # 通过坐标对玻片排序，如果托盘摆放方向改变需要修改排序方式
    # 当前height值越大则玻片序号越小，即图片最低下是1号，最上面是4号
    boxs.sort(key=lambda x: x[0][1], reverse=True)
    for i, box in enumerate(boxs):
        gls = perspective_transf(box, img)
        glses.append(gls)
        cv2.imwrite(f'gls{i}.jpg', gls)
        cv2.imshow(f'gls{i}.jpg', gls)
    return glses

def img_to_mask(img):
    #获取标定信息
    with open('cfg.inf', 'r') as fp:
        param = eval(fp.read())
    #DIM, K, D = camera_calibr('chess')
    DIM = param['DIM']
    K = param['K']
    D = param['D']
    row = param['row']
    col = param['col']
    #矫正
    img = corr_photo(img, DIM, K, D)
    cv2.imshow('corr', img)
    cv2.imwrite('corr.jpg', img)
    #提取玻片图片
    glses = get_gls(img)
    maskes = []
    for i,gls in enumerate(glses):
        #将玻片图片转为掩码
        mask = to_mask(gls, row, col)
        cv2.imshow(f'mask{i}', mask * 255)
        maskes.append(mask)
    return maskes

if __name__=='__main__':
    camera = Camera()
    light = Light()
    light.light('on')
    #拍照
    time.sleep(5)
    img = take_photo(camera)
    light.light('off')
    #cv2.imshow('src', img)
    maskes = img_to_mask(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    light.close()
    camera.close()
