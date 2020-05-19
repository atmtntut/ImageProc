import os
import copy
import json
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_points(scan_info):
    points = []
    try:
        with open(scan_info, 'r') as fp:
            info = json.loads(fp.read())
            print(f'ScanID: {info["ScanID"]}')
            for rng in info['RangeInfo']:
                for grd in rng['GridInfo']:
                    for pnt in grd['FocusPnts']:
                        points.append({'x': pnt['FcsX'], 'y': pnt['FcsY'], 'z': pnt['FcsZ'], 'is_used': pnt['IsUsed'] == 1})
    except Exception as e:
        print(e)
    return points

def calc_plane_param(points):
    p1 = [points[0]['x'], points[0]['y'], points[0]['z']]
    p2 = [points[1]['x'], points[1]['y'], points[1]['z']]
    p3 = [points[2]['x'], points[2]['y'], points[2]['z']]
    a = ((p2[1]-p1[1])*(p3[2]-p1[2])-(p2[2]-p1[2])*(p3[1]-p1[1]));
    b = ((p2[2]-p1[2])*(p3[0]-p1[0])-(p2[0]-p1[0])*(p3[2]-p1[2]));
    c = ((p2[0]-p1[0])*(p3[1]-p1[1])-(p2[1]-p1[1])*(p3[0]-p1[0]));
    d = (0 - (a * p1[0] + b * p1[1] + c * p1[2]));
    return (a, b, c, d)

def calc_z(x, y, prm):
    a, b, c, d = prm[0], prm[1], prm[2], prm[3]
    if isinstance(x, int) or isinstance(x, float):
        Z = -1 * (a * x + b * y + d) / c
    elif isinstance(x, np.ndarray):
        z = []
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z.append(-1 * (a * x[i][j] + b * y[i][j] + d) / c)
        Z = np.array(z).reshape(x.shape)
    return Z

def calc_delta(points, prm):
    UP_TISSUE_PEAK_DISTANCE = 120000
    delta_sum = 0
    for pnt in points:
        z = calc_z(pnt['x'], pnt['y'], prm)
        delta1 = abs(z - pnt['z']);
        delta2 = abs(z - (pnt['z'] + UP_TISSUE_PEAK_DISTANCE))
        # Delta1 不是足够好, Delta2 足够好
        if (delta1 >= (UP_TISSUE_PEAK_DISTANCE // 4)) \
                and ((delta2 < delta1) and (delta2 < (UP_TISSUE_PEAK_DISTANCE // 3))):
            delta_sum += delta2
            #print('({}, {}, {}),  delta1: {:.1f}, #delta2: {:.1f}'.format(pnt['x'], pnt['y'], pnt['z'], delta1, delta2))
        else:
            delta_sum += delta1
            #print('({}, {}, {}), #delta1: {:.1f},  delta2: {:.1f}'.format(pnt['x'], pnt['y'], pnt['z'], delta1, delta2))
    return delta_sum, delta_sum / len(points)

def draw_3d_plane(ax, min_pos, max_pos, prm):
    x = np.arange(min_pos[0], max_pos[0], 500000, dtype=np.float)
    y = np.arange(min_pos[1], max_pos[1], 500000, dtype=np.float)
    X, Y = np.meshgrid(x, y)
    Z = calc_z(X, Y, prm)
    #print(Z.shape)
    ax.plot_surface(X, Y, Z, color='gray', alpha=0.5)

def draw_3d_points(ax, points):
    if len(points) > 0:
        X, Y, Z, C = [], [], [], []
        for pnt in points:
            X.append(pnt['x'])
            Y.append(pnt['y'])
            Z.append(pnt['z'])
            if pnt['is_used']:
                C.append('r')
            else:
                C.append('b')

        ax.scatter(X, Y, Z, c=C)

def is_line(points):
    EPSION = 0.0001;

    # 功能：判断3点是否在一条直线
    # 参数：fFirstPnt表示第一个点的指针，fFirstPnt[0]为点的X值，fFirstPnt[1]为点的Y值
    #       fSecondPnt表示第一个点的指针，fSecondPnt[0]为点的X值，fSecondPnt[1]为点的Y值
    #       fThirdPnt表示第一个点的指针，fThirdPnt[0]为点的X值，fThirdPnt[1]为点的Y值
    # 返回值：True：在一条直线
    #         False:不在一条直线
    # 用三个点组成两个向量，算叉积小于某个值，认为是在同一条直线
    # 数据归一化
    first, second, third = points[0], points[1], points[2]
    x0 = (first['x'] - second['x'])/first['x'];
    x1 = (first['x'] - third['x'] )/first['x'];
    y0 = (first['y'] - second['y'])/first['y'];
    y1 = (first['y'] - third['y'] )/first['y'];
    if math.fabs((x0*y1) - (x1*y0)) < EPSION:
        return True
    return False

def calc_all_plane(points):
    datas = []
    for first in range(len(points)):
        for second in range(first+1, len(points)):
            for third in range(second+1, len(points)):
                if (first == second) or (first == third) or (second == third): continue

                plane_pnts = [points[first], points[second], points[third]]
                if is_line(plane_pnts): continue

                prm = calc_plane_param(plane_pnts)
                a, b, c, d = prm
                if c == 0: continue

                # 计算另外点的偏差,如果小于阈值,则继续执行
                # 否则与当前最好情况比较,离群点点个数越小则平面越好,离群点越近则平面越好,如果比当前最好平面好则更新
                delta_sum, delta_avg = calc_delta(points, prm)
                data = {'delta_avg': delta_avg, 'prm': (a, b, c, d), 'plane_pnts': copy.deepcopy(plane_pnts)}
                datas.append(data)
    return datas

def print_focus(scan_info, plane_pnts):
    offset = 0
    try:
        with open(scan_info, 'r') as fp:
            info = json.loads(fp.read())
            for rng in info['RangeInfo']:
                for grd in rng['GridInfo']:
                    for pnt in grd['FocusPnts']:
                        if offset < len(plane_pnts):
                            if plane_pnts[offset]['x'] == pnt['FcsX'] \
                                    and plane_pnts[offset]['y'] == pnt['FcsY'] \
                                    and plane_pnts[offset]['z'] == pnt['FcsZ']:
                                print(pnt)
                                offset += 1
                        else:
                            return
    except Exception as e:
        print(e)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--all', default=False, help='calc all plane')
    parser.add_argument('-f', '--file', default='./zjscaninf.json', help='calc all plane')

    args = parser.parse_args()
    print(args)

    info_file = args.file
    points = get_points(info_file)
    min_pos = [points[0]['x'], points[0]['y']]
    max_pos = [points[0]['x'], points[0]['y']]
    plane_pnts = []
    for pnt in points:
        if pnt['is_used']:
            plane_pnts.append(copy.deepcopy(pnt))
            pnt['is_used'] = False
        if min_pos[0] > pnt['x']:
            min_pos[0] = pnt['x']
        if max_pos[0] < pnt['x']:
            max_pos[0] = pnt['x']
        if min_pos[1] > pnt['y']:
            min_pos[1] = pnt['y']
        if max_pos[1] < pnt['y']:
            max_pos[1] = pnt['y']
    min_pos[0] -= 1000000
    min_pos[1] -= 1000000
    max_pos[0] += 1000000
    max_pos[1] += 1000000

    if not args.all:
        prm = calc_plane_param(plane_pnts)
        delta_sum, delta_avg = calc_delta(points, prm)
    else:
        datas = calc_all_plane(points)
        datas.sort(key=lambda x:x['delta_avg'])
        prm = datas[0]['prm']
        delta_avg = datas[0]['delta_avg']
        for pnt in datas[0]['plane_pnts']:
            pnt['is_used'] = True
        plane_pnts = datas[0]['plane_pnts']
        for data in datas[:5]:
            print_focus(info_file, data['plane_pnts'])
            print('delta_avg: {:.1f}'.format(data['delta_avg']))
    print_focus(info_file, plane_pnts)
    print('delta_avg: {:.1f}'.format(delta_avg))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    draw_3d_points(ax, points + plane_pnts)
    draw_3d_plane(ax, min_pos, max_pos, prm)

    x, y = (min_pos[0]+max_pos[0])//2, (min_pos[1]+max_pos[1])//2
    ax.text(x, y, calc_z(x, y, prm), 'delta: {:05.1f}'.format(delta_avg), color='r')

    plt.show()
