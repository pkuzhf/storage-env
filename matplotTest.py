import matplotlib

#matplotlib.use('Agg')
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
import time


def read_map(filepath, idx):
    # TODO read map info
    return


def randomcolor(count):
    colors = np.random.random((count+1, 3))
    return colors


def draw_map(mapsize, conveyors, hole_pos, hole_city, agent_pos, agent_city, colors, filename, step):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.grid(True, color="black", alpha=0.25, linestyle='-')

    for pos in conveyors:
        p = patches.Wedge(
            ((pos[0]), (pos[1])),
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
            facecolor=(colors[hole_city[i]][0], colors[hole_city[i]][1], colors[hole_city[i]][2]),
            linewidth=0.5,
            linestyle='-'
        )
        ax.add_patch(p)

    for j in range(len(agent_pos)):
        p = patches.Circle(
            ((agent_pos[j][0]+0.5), (agent_pos[j][1]+0.5)),
            0.5,
            facecolor=(colors[agent_city[j]][0], colors[agent_city[j]][1], colors[agent_city[j]][2]),
            linewidth=0.5,
            linestyle='-'
        )
        ax.add_patch(p)

    plt.xticks(np.arange(0, mapsize[0]+1, 1), color="none")
    plt.yticks(np.arange(0, mapsize[1]+1, 1), color="none")
    # plt.show()
    fig.savefig('testoutput' + '/' + filename + str(step) + '.png', dpi=150, bbox_inches='tight')


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
    pass
