import matplotlib.pyplot as plt
import numpy as np

#um/ms
maxspeed = 24
#um/ms^2
acc = 0.1142
#加速度变化时间 ms
acctime = 13

def calc_acc(currtime):
    if currtime < acctime:
        return currtime*acc/acctime
    else:
        return acc

def calc_speed(currtime):
    if currtime < acctime:
        speed = (currtime**2)*acc/(2*acctime)
    else:
        speed = acctime*acc/2 + (currtime-acctime)*acc
    if speed < maxspeed:
        return speed
    return maxspeed

def calc_long(currtime):
    t2 = maxspeed/acc + acctime/2
    #print(t2)
    if currtime < acctime:
        #由积分公式推导
        l1 = (currtime**4)*acc/(8*acctime)
        return l1
    elif currtime < t2:
        l1 = (acctime**3)*acc/8
        l2 = currtime*(currtime - acctime/2)*acc/2
        return l1+l2
    else:
        l1 = (acctime**3)*acc/8
        l2 = t2*(t2 - acctime/2)*acc/2
        l3 = (currtime - t2)*maxspeed
        return l1+l2+l3

def L1(l):
    return l
    max_val = max(l)
    return [i/max_val for i in l]

def set_para(maxV, A, t1):
    global maxspeed, acc, acctime
    #um/ms
    #global maxspeed = 24
    maxspeed = maxV/1000
    #um/ms^2
    #global acc = 0.1142
    acc = A/(10**6)
    #加速度变化时间 ms
    #global acctime = 13
    acctime = t1

if __name__=='__main__':
    #set_para(24000, 114200, 13)
    times = np.linspace(0,300,500)
    #speeds = [calc_speed(t) for t in times]
    #accs = [calc_acc(t) for t in times]
    XY = (24000, 114200, 13)
    Z = (1200, 5710, 13)
    ls = []
    ss = []
    set_para(Z[0], Z[1], Z[2])
    ls.append([calc_long(t) for t in times])
    ss.append([calc_speed(t) for t in times])

    set_para(Z[0], Z[1]*2, Z[2]/2)
    ls.append([calc_long(t) for t in times])
    ss.append([calc_speed(t) for t in times])

    set_para(Z[0]*2, Z[1]*3, Z[2]/2)
    ls.append([calc_long(t) for t in times])
    ss.append([calc_speed(t) for t in times])

    set_para(Z[0]*2, Z[1]*3, Z[2]/3)
    ls.append([calc_long(t) for t in times])
    ss.append([calc_speed(t) for t in times])

    #set_para(1200, 5710*2, 13)
    #ls.append([calc_long(t) for t in times])
    #print(calc_long(272))
    #print(calc_long(282))
    #print(calc_long(292))
    #print(calc_long(302))

    #plt.plot(times, L1(speeds), linewidth=0.5 ,label='speed')
    #plt.plot(times, L1(accs), linewidth=0.5 ,label='acc')
    plt.figure(1)
    plt.subplot(211)
    for i,l in enumerate(ls):
        plt.plot(times, L1(l), linewidth=1 ,label=f'L{i}')
    #plt.plot(times, L1(ls2), linewidth=0.5 ,label='L')
    plt.legend()
    plt.grid(True)

    plt.subplot(212)
    for i,s in enumerate(ss):
        plt.plot(times, L1(s), linewidth=1 ,label=f'S{i}')
    plt.legend()
    plt.grid(True)
    plt.show()
