import gym
import numpy as np
from agent_gym import AGENT_GYM
import config, utils
import matplotTest as draw
import mapGenerator as MG
import pathfinding as PF
from sets import Set

# source_pos = [[0,0]]
# hole_pos= [[4,4],[0,4],[4,0]]
agent_num = 3
total_time = 200
# hole_city = [0,1,2]
# city_dis = [0.33, 0.33, 0.34]


# there are some problems in randomcity()
# so hole_city and city_dis are inited by hand
source_pos, hole_pos = MG.bigMap(10,10)
hole_city = [0,1,2,2,1,3]
color = draw.randomcolor(4)
color[4] = [1, 1, 1]
city_dis = [0.12,0.33,0.37,0.18]

env = AGENT_GYM(source_pos, hole_pos, agent_num, total_time, hole_city, city_dis)
env.seed(config.Game.Seed)

window = 10

def encode(pos, t):
    [x, y] = pos
    return str(x) + ' ' + str(y) + ' ' + str(t)

def isValidPath(pos, t, agent_pos, reserve):
    return pos not in agent_pos and pos not in source_pos and pos not in hole_pos and encode(pos, t) not in reserve

def AStar(start_pos, end_pos, agent_pos, reserve, t0):
    open_set = []
    close_set = []
    h = {}
    g = {}
    path = {}
    open_set.append([start_pos, t0])
    pos = start_pos
    g[encode(pos, t0)] = 0
    h[encode(pos, t0)] = utils.getDistance(start_pos, end_pos)
    path[encode(pos, t0)] = [[-1, -1], -1, -1]
    schedule = []
    while len(open_set) > 0:
        min_f = -1
        idx = -1
        for i in range(len(open_set)):
            [pos, t] = open_set[i]
            f = g[encode(pos, t)] + h[encode(pos, t)]
            if min_f == -1 or f < min_f:
                min_f = f
                idx = i
        [pos, t] = open_set[idx]
        open_set.pop(idx)
        close_set.append([pos, t])

        dirs = utils.dirs + [0, 0]
        for a in range(len(dirs)):
            [x, y] = pos
            x += dirs[a][0]
            y += dirs[a][1]
            new_pos = [x, y]
            if not isValidPath(new_pos, t + 1, agent_pos, reserve) or [new_pos, t + 1] in close_set:
                continue
            new_g = g[encode(pos, t)] + utils.getDistance(new_pos, pos)
            if [new_pos, t + 1] not in open_set:
                open_set.append([new_pos, t + 1])
                g[encode(new_pos, t + 1)] = new_g
                h[encode(new_pos, t + 1)] = utils.getDistance(new_pos, end_pos)
                path[encode(new_pos, t + 1)] = [pos, t, a]
            elif new_g < g[encode(new_pos, t + 1)]:
                g[encode(new_pos, t + 1)] = new_g
                path[encode(new_pos, t + 1)] = [pos, t, a]

        if g[encode(pos, t)] == window or len(open_set) == 0:
            while (path[encode(pos, t)] != [[-1, -1], -1, -1]):
                schedule.insert(0, path[encode(pos, t)])
                [pos, t, a] = path[encode(pos, t)]
            break
    
    return schedule

def pathfinding(reserve, agent_pos, agent_city, agent_id, t):
    start_pos = agent_pos[agent_id]
    city = agent_city[agent_id]
    min_distance = -1
    if city == -1:
        for i in range(len(source_pos)):
            distance = utils.getDistance(source_pos[i], start_pos)
            if min_distance == -1 or distance < min_distance:
                min_distance = distance
                end_pos = source_pos[i]
    else:
        for i in range(len(hole_pos)):
            if hole_city[i] != city:
                continue
            distance = utils.getDistance(hole_pos[i], start_pos)
            if min_distance == -1 or distance < min_distance:
                min_distance = distance
                end_pos = source_pos[i]   
    schedule = AStar(start_pos, end_pos, agent_pos, reserve, t)
    return schedule


for i_episode in range(1):
    observation = env.reset()
    print observation
    [agent_pos, agent_city, agent_reward, hole_reward, source_reward] = observation
    reserve = Set()
    schedule = [[]] * agent_num
    for t in range(total_time):
        env.render()

        # random action
        # action = []
        # for i in range(agent_num):
        #     r = np.random.random_integers(0, 4)
        #     if r < 4:
        #         action.append(utils.dirs[r])
        #     else:
        #         action.append([0,0])

        # pathfinding

        action = []
        for i in range(agent_num):
            if len(schedule[i]) == 0:
                schedule[i] = pathfinding(agent_pos, agent_city, i, t)
            else:
                [pos, t, a] = schedule[i][0]
                if pos != agent_pos[i]:
                    for [pos, t, a] in schedule[i]:
                        reserve.remove(encode(pos, t))
                    schedule[i] = pathfinding(reserve, agent_pos, agent_city, i, t)
                    for [pos, t, a] in schedule[i]:
                        reserve.add(encode(pos, t))
            [pos, t, a] = schedule[i][0]
            schedule[i] = schedule[i][1:]
            reserve.remove(encode(pos, t))
            action.append(a)


        print action
        observation, reward, done, info = env.step(action)
        [agent_pos, agent_city, agent_reward, hole_reward, source_reward] = observation


        # so many params...
        draw.draw_map([10,10], source_pos, hole_pos, hole_city, agent_pos, agent_city,
                      color, "show", t, agent_reward, hole_reward, source_reward, city_dis)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

draw.save_video("show", total_time)
