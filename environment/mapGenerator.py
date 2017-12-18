import matplotlib
matplotlib.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random

def bigMap(w,h):
    source=[]
    hole=[]

    sy=1
    for i in range(h):
        source.append([0,sy])
        source.append([w-1,sy])
        sy+=2+i%2
        if sy>h:
            break

    if w%3==1:
        xbias=1
    else:
        xbias=0

    if h%3==2:
        ybias=1
    else:
        ybias=0
    hx = 2+xbias
    hy = 1+ybias

    while hy<h-1:
        while hx<w-2:
            hole.append([hx,hy])
            hx+=3
        hx=2+xbias
        hy+=3

    return source, hole


def smallMap(w,h):
    source = []
    hole = []

    sy = 1
    for i in range(h):
        source.append([0, sy])
        source.append([w-1, sy])
        sy += 2
        if sy > h:
            break

    hx = 2
    hy = 1
    while hy < h-1:
        while hx < w - 2:
            hole.append([hx, hy])
            hx += 2
        hx = 2
        hy += 2

    return source, hole


def generateMap(w,h):
    if w<=13 and h<=13:
        return smallMap(w,h)
    else:
        return bigMap(w,h)


# still have some problem
def randomCity(cityNum, holeNum):
    hole_city=[]
    for i in range(holeNum):
        hole_city.append(np.random.randint(0,cityNum))

    city_dis=[]
    for j in range(cityNum):
        city_dis.append(np.random.random())
    dis_sum = sum(city_dis)
    for j in range(cityNum):
        city_dis[j]/=dis_sum

    return hole_city, city_dis




def show_map(mapsize, conveyors, hole_pos):
    # only for test
    fig = plt.figure()

    ax = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=3)
    ax.grid(True, color="black", alpha=0.25, linestyle='-')

    for k in range(len(conveyors)):
        p = patches.Wedge(
            ((conveyors[k][0]), (conveyors[k][1])),
            1,
            0,
            60,
            width=1,
            facecolor='r',
            linewidth=0.5,
            linestyle='-',
            alpha=0.5
            )
        ax.add_patch(p)

    for i in range(len(hole_pos)):
        p = patches.Rectangle(
            ((hole_pos[i][0]), (hole_pos[i][1])),
            1,
            1,
            linewidth=0.5,
            linestyle='-'
        )
        ax.add_patch(p)

    # set ticks and spines
    ax.set_xticks(np.arange(0, mapsize[0]+1, 1))
    ax.set_xticklabels(())
    ax.set_yticks(np.arange(0, mapsize[1]+1, 1))
    ax.set_yticklabels(())

    plt.show()


# if __name__ == "__main__":
    # edge=10
    # a,b=bigMap(edge,edge)
    # randomCity(4,6)
    # a,b=generateMap(edge,edge)
    # show_map((edge,edge),a,b)

def hugeMap(w,h):
    source=[]
    hole=[]

    sy=2
    for i in range(h):
        source.append([2,sy])
        source.append([w-3,sy])
        sy+=3
        if sy>h-3:
            break

    if w%3==1:
        xbias=1
    else:
        xbias=0

    if h%3==2:
        ybias=1
    else:
        ybias=0
    hx = 5+xbias
    hy = 2+ybias

    while hy<h-3:
        while hx<w-5:
            hole.append([hx,hy])
            hx+=3
        hx=5+xbias
        hy+=3

    nb_hole = len(hole)
    citydis = [1.0 for i in range(nb_hole/4)]
    for i in range(nb_hole / 4):
        citydis[i]+=random.randint(1,4)
    dis_sum = sum(citydis)
    for i in range(nb_hole / 4):
        citydis[i]/=dis_sum

    hole_city = [-1 for i in range(nb_hole)]
    for i in range(nb_hole / 4):
        choice = random.randint(0, nb_hole-1)
        while hole_city[choice]!=-1:
            choice = random.randint(0, nb_hole - 1)
        hole_city[choice] = i
    for i in range(nb_hole):
        if hole_city[i]==-1:
            hole_city[i] = np.random.multinomial(1, citydis, size=1).tolist()[0].index(1)

    return source, hole, hole_city, citydis


def huge_map_city(nb_hole, citydis):
    hole_city = [-1 for i in range(nb_hole)]
    for i in range(nb_hole / 4):
        choice = random.randint(0, nb_hole - 1)
        while hole_city[choice] != -1:
            choice = random.randint(0, nb_hole - 1)
        hole_city[choice] = i
    for i in range(nb_hole):
        if hole_city[i] == -1:
            hole_city[i] = np.random.multinomial(1, citydis, size=1).tolist()[0].index(1)
    return hole_city
