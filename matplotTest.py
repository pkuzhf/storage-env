import matplotlib

#matplotlib.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
import time
import os
#import cv2
import imageio


def read_map(filepath, idx):
    # TODO read map info
    return


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
    ax3.text(0, 0.5, "Package num: " + str(sum(source_reward)), size=12, weight="light")
    ax3.text(0, 0.3, "Total Reward: " + str(sum(agent_reward)), size=12, weight="light")

    for k in range(len(conveyors)):
        p=patches.Polygon(
            [[conveyors[k][0],conveyors[k][1]],
             [conveyors[k][0]+1, conveyors[k][1]],
             [conveyors[k][0]+0.5, conveyors[k][1]+1]
             ],
            facecolor=(0.9,0.9,0.9),
            linewidth=0.5,
            linestyle='-'
        )
        ax.text(conveyors[k][0]+0.35, conveyors[k][1]+0.35, str(source_reward[k]),
                size=fontsize, weight="light")
        ax.add_patch(p)

    for i in range(len(hole_pos)):
        p = patches.Rectangle(
            ((hole_pos[i][0]), (hole_pos[i][1])),
            1,
            1,
            facecolor=(colors[hole_city[i]][0], colors[hole_city[i]][1], colors[hole_city[i]][2]),
            linewidth=0.5,
            linestyle='-'
        )
        ax.text(hole_pos[i][0]+0.35, hole_pos[i][1]+0.35, str(hole_reward[i]),
                size=fontsize, weight="light", color=(1,1,1))
        ax.add_patch(p)

    for j in range(len(agent_pos)):
        p = patches.Circle(
            ((agent_pos[j][0]+0.5), (agent_pos[j][1]+0.5)),
            0.5,
            facecolor=(colors[agent_city[j]][0], colors[agent_city[j]][1], colors[agent_city[j]][2]),
            linewidth=0.5,
            linestyle='-'
        )
        ax.text(agent_pos[j][0]+0.35, agent_pos[j][1]+0.35, str(agent_reward[j]),
                size=fontsize, weight="light", alpha=0.85)
        ax.add_patch(p)

    # set ticks and spines
    ax.set_xticks(np.arange(0, mapsize[0]+1, 1))
    ax.set_xticklabels(())
    ax.set_yticks(np.arange(0, mapsize[1]+1, 1))
    ax.set_yticklabels(())

    ax1.spines['bottom'].set_color('none')
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
    

def save_video(dir, filename, step):
    frame = cv2.imread(dir + '/' + filename + '0' + '.png')
    x = frame.shape[0]
    y = frame.shape[1]

    fourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D')
    videoWriter = cv2.VideoWriter(dir + '/' + filename + '.avi', fourcc, 10, (y, x))

    for i in range(step):
        frame = cv2.imread(dir+'/' + filename + str(i) + '.png')
        videoWriter.write(frame)
    videoWriter.release()


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
    pass

