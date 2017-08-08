import matplotlib

#matplotlib.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
import time
import os
import imageio


def randomcolor(count):
    colors = np.random.random((count+1, 3))
    return colors


def draw_map(mapsize, conveyors, hole_pos, hole_city, agent_pos, agent_city, colors,
             dir, filename, step, agent_reward, hole_reward, source_reward, city_dis):
    fig = plt.figure()

    # fontsize for texts
    fontsize = int(120.0/mapsize[1])

    # main graph
    ax = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=3)
    ax.grid(True, color="black", alpha=0.25, linestyle='-')

    ax1 = plt.subplot2grid((3, 4), (0, 3))
    ax1.text(0, 0.8, "map size: " + str(mapsize), size=12, weight="light")
    ax1.text(0, 0.6, "hole num: " + str(len(hole_pos)), size=12, weight="light")
    ax1.text(0, 0.4, "source num: " + str(len(conveyors)), size=12, weight="light")
    ax1.text(0, 0.2, "agent num: " + str(len(agent_pos)), size=12, weight="light")
    ax1.text(0, 0, "city distribution: ", size=12, weight="light")

    # pie graph
    ax2 = plt.subplot2grid((3, 4), (1, 3))
    ax2.pie(city_dis, colors=colors, radius=1.25, autopct='%1.1f%%')

    # timestep graph
    ax3 = plt.subplot2grid((3, 4), (2, 3))
    ax3.text(0, 0.7,"Timestep: "+str(step/2), size=12, weight="light")
    ax3.text(0, 0.5, "Pack num: " + str(sum(source_reward)), size=12, weight="light")
    ax3.text(0, 0.bottom'].set_color('none')

    ax1.spines['top'].set_color('none')
    ax1.spines['right'].set_color('none')
    ax1.spines['left'].set_color('none')
    ax1.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                    right='off', left='off', labelleft='off')

    ax2.spines['bottom'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.spines['right'].set_color('none')
    ax2.spines['left'].set_color('none')
    ax2.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                    right='off', left='off', labelleft='off')

    ax3.spines['bottom'].set_color('none')
    ax3.spines['top'].set_color('none')
    ax3.spines['right'].set_color('none')
    ax3.spines['left'].set_color('none')
    ax3.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                    right='off', left='off', labelleft='off')

    # plt.show()

    if not os.path.exists(dir):
        os.mkdir(dir)
    fig.savefig(dir + '/' + filename + str(step) + '.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


def save_video2(dir,filename,step):
    with imageio.get_writer(dir+'/'+filename+'.gif',mode='I',fps=10) as writer:
        for i in range(step):
            image = imageio.imread(dir+'/'+filename+str(i)+'.png')
            writer.append_data(image)


def save_video3(dir,filename,step):
    with imageio.get_writer(dir+'/'+filename+'.mp4',mode='I',fps=10) as writer:
        for i in range(step):
            image = imageio.imread(dir+'/'+filename+str(i)+'.png')
            writer.append_data(image)


def read_result():
    file = open('result/result', 'r')
    result = file.readlines()

    fig = plt.figure()

    x = range(1,50)
    y = [0]*49

    for point in result:
        point = point.split()
        plt.scatter(int(point[0]), int(point[1]),alpha=0.25,color=(0.2,0.2,1))
        y[int(point[0])-1] += int(point[1])/20.0
    # print point
    plt.plot(x,y,color='r',linewidth=1.5)

    plt.xlim(0,50)
    plt.ylim(0,1000)
    plt.xticks(np.arange(0, 50, 2))
    plt.yticks(np.arange(0, 1000, 50))
    plt.grid(True, color="black", alpha=0.25, linestyle='-')
    # plt.show()
    plt.savefig('result/' + 'average.png', dpi=120, bbox_inches='tight')
    plt.close()


def draw_log(dir, picdir, filename):
    log = open(dir)
    mapsize = eval(log.readline())
    city_dis = eval(log.readline())
    source_pos = eval(log.readline())
    hole_pos = eval(log.readline())
    hole_city = eval(log.readline())
    colors = randomcolor(len(city_dis))
    colors[len(city_dis)] = [0.9, 0.9, 0.9]

    step = 0
    agent_reward = eval(log.readline())
    source_reward = eval(log.readline())
    hole_reward = eval(log.readline())
    agent_pos = eval(log.readline())
    agent_city = eval(log.readline())
    draw_map(mapsize, source_pos, hole_pos, hole_city, agent_pos, agent_city, colors,
             picdir, filename, step, agent_reward, hole_reward, source_reward, city_dis)
    line = log.readline()
    while line!='end':
        step+=1
        agent_pos = eval(line)
        agent_city = eval(log.readline())
        agent_reward = eval(log.readline())
        source_reward = eval(log.readline())
        hole_reward = eval(log.readline())
        draw_map(mapsize, source_pos, hole_pos, hole_city, agent_pos, agent_city, colors,
                 picdir, filename, step, agent_reward, hole_reward, source_reward, city_dis)
        step+=1
        agent_pos = eval(log.readline())
        agent_city = eval(log.readline())
        draw_map(mapsize, source_pos, hole_pos, hole_city, agent_pos, agent_city, colors,
                 picdir, filename, step, agent_reward, hole_reward, source_reward, city_dis)
        line = log.readline()

    save_video3('logtest', "demo", step+1)

test_mapsize = [50, 31]
test_conveyors = [[0, 8], [0, 16], [49, 30]]
test_holes = [[7,16,1], [13,22,2], [22,25,3]]
test_robots = [[3,5,0], [0,17,0], [12,22,1], [47,21,2], [37,29,2], [22,25,3]]


if __name__ == "__main__":
    # citycount = len(test_holes)
    # colors = randomcolor(citycount)
    # for step in range(1000):
    #     for robot in test_robots:
    #         random = np.random.randint(0,4)
    #         if random==0:
    #             robot[0]+=1
    #         elif random==1:
    #             robot[0]-=1
    #         elif random==2:
    #             robot[1]+=1
    #         elif random==3:
    #             robot[1]-=1
    #     draw_map(test_mapsize, test_conveyors, test_holes, test_robots, colors, "test", step)
    # main()
    # save_video2("show", 50)
    # imageio.help()
    # read_result()
    # save_video3('mapOutput_0',"show", 987)
    # draw_log('result/mapInfo5_0.log', 'logtest', 'demo')
    pass

