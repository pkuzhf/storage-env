import gym
import numpy as np
from agent_gym import AGENT_GYM
import config, utils
import matplotTest as draw
import mapGenerator as MG

# source_pos = [[0,0]]
# hole_pos= [[4,4],[0,4],[4,0]]
#agent_num = 20
total_time = 1000
# hole_city = [0,1,2]
# city_dis = [0.33, 0.33, 0.34]


# there are some problems in randomcity()
# so hole_city and city_dis are inited by hand
source_pos, hole_pos = MG.bigMap(config.Map.Height, config.Map.Width)
hole_city = [0,1,2,2,1,3]
color = draw.randomcolor(4)
color[4] = [1, 1, 1]
city_dis = [0.12,0.33,0.37,0.18]

window = 10

def encode(pos, t):
    [x, y] = pos
    return str(x) + ' ' + str(y) + ' ' + str(t)

def isValidPath(agent_id, pos, t, reserve, agent_pos, end_pos):
    if not utils.inMap(pos):
        #print 'not valid: out of map'
        return False
    elif encode(pos, t) in reserve and reserve[encode(pos, t)] != agent_id:
        #print 'not valid: reserved by ' + str(reserve[encode(pos, t)])
        return False
    elif encode(pos, t + 1) in reserve and reserve[encode(pos, t + 1)] != agent_id:
        #print 'not valid: reserved by ' + str(reserve[encode(pos, t + 1)])
        return False
    elif pos in agent_pos and agent_pos.index(pos) != agent_id:
        #print 'not valid: agent ' + str(agent_pos.index(pos))
        return False
    elif pos == end_pos:
        return True
    elif pos in source_pos or pos in hole_pos:
        #print 'not valid: source or hole'
        return False
    else:
        return True

def AStar(agent_id, start_pos, end_pos, agent_pos, reserve, t0):
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

        #dirs = utils.dirs + [[0, 0]]
        dirs = utils.dirs
        for i in range(len(dirs)):
            a = dirs[i]
            new_pos = [pos[0] + a[0], pos[1] + a[1]]
            #print new_pos
            if not isValidPath(agent_id, new_pos, t + 1, reserve, agent_pos, end_pos):
                continue
            if [new_pos, t + 1] in close_set:
                continue
            new_g = g[encode(pos, t)] + 1
            if [new_pos, t + 1] not in open_set:
                open_set.append([new_pos, t + 1])
                g[encode(new_pos, t + 1)] = new_g
                h[encode(new_pos, t + 1)] = utils.getDistance(new_pos, end_pos)
                path[encode(new_pos, t + 1)] = [pos, t, a]
            elif new_g < g[encode(new_pos, t + 1)]:
                g[encode(new_pos, t + 1)] = new_g
                path[encode(new_pos, t + 1)] = [pos, t, a]

        if pos == end_pos or g[encode(pos, t)] == window or len(open_set) == 0:
            while (path[encode(pos, t)] != [[-1, -1], -1, -1]):
                schedule.insert(0, path[encode(pos, t)])
                [pos, t, a] = path[encode(pos, t)]
            break
    if len(schedule) == 0:
        #print 'fail to schedule, stay'
        #print path
        schedule.append([start_pos, t, [0, 0]])
    [pos, t, a] = schedule[-1]
    new_pos = [pos[0] + a[0], pos[1] + a[1]]
    schedule.append([new_pos, t + 1, [0, 0]])
    #print 'AStar'
    #print ['start_pos', start_pos, 'end_pos', end_pos]
    #print ['schedule', schedule]
    #print 'End AStar'
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
                end_pos = hole_pos[i]
    #print 'pathfinding'
    #print ['city', city]
    schedule = AStar(agent_id, start_pos, end_pos, agent_pos, reserve, t)
    #print 'end pathfinding'
    return schedule

def removeSchedule(agent_id, reserve, schedule):
    #print 'remove schedule ' + str(agent_id)
    delset = {}
    for [pos, t, a] in schedule[agent_id]:
        delset[encode(pos, t)] = True
        delset[encode(pos, t + 1)] = True
    for entry in delset:
        del reserve[entry]
    schedule[agent_id] = []

def addReserve(entry, agent_id, reserve, schedule):
    if entry in reserve:
        i = reserve[entry]
        #print 'entry collision ' + str(entry) + ' new ' + str(agent_id) + ' old ' + str(i)
        if agent_id != i:
            removeSchedule(i, reserve, schedule)
    reserve[entry] = agent_id


for agent_num in range(1, 50):

    env = AGENT_GYM(source_pos, hole_pos, agent_num, total_time, hole_city, city_dis)
    env.seed(config.Game.Seed)

    #print 'agent_num: ' + str(agent_num)
    observation = env.reset()
    #print observation
    [agent_pos, agent_city, agent_reward, hole_reward, source_reward] = observation
    reserve = {}
    schedule = [[]] * agent_num
    for time in range(total_time):
        #print ['*** time ***', time]
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
            #print 'agent ' + str(i) + ' in pos ' + str(agent_pos[i])
            if len(schedule[i]) == 0:
                schedule[i] = pathfinding(reserve, agent_pos, agent_city, i, time)
                for [pos, t, a] in schedule[i]:
                    addReserve(encode(pos, t), i, reserve, schedule)
                    addReserve(encode(pos, t + 1), i, reserve, schedule)
            elif len(schedule[i]) == 1:
                removeSchedule(i, reserve, schedule)
                schedule[i] = pathfinding(reserve, agent_pos, agent_city, i, time)
                for [pos, t, a] in schedule[i]:
                    addReserve(encode(pos, t), i, reserve, schedule)
                    addReserve(encode(pos, t + 1), i, reserve, schedule)
            else:
                [pos, t, a] = schedule[i][0]
                if pos != agent_pos[i]:
                    removeSchedule(i, reserve, schedule)
                    schedule[i] = pathfinding(reserve, agent_pos, agent_city, i, time)
                    for [pos, t, a] in schedule[i]:
                        addReserve(encode(pos, t), i, reserve, schedule)
                        addReserve(encode(pos, t + 1), i, reserve, schedule)
            #print ['schedule', schedule[i]]
            [pos, t, a] = schedule[i][0]
            del schedule[i][0]
            del reserve[encode(pos, t)]
            if a != [0, 0]:
                del reserve[encode(pos, t + 1)]
            action.append(a)


        #print ['action', action]
        observation, reward, done, info = env.step(action)
        [agent_pos, agent_city, agent_reward, hole_reward, source_reward] = observation
        #print ['agent_pos', agent_pos]

        for i in range(agent_num):
            addReserve(encode(agent_pos[i], time + 1), i, reserve, schedule)
            addReserve(encode(agent_pos[i], time + 2), i, reserve, schedule)


        # so many params...
        #draw.draw_map([config.Map.Width,config.Map.Height], source_pos, hole_pos, hole_city, agent_pos, agent_city,
        #              color, "show", time, agent_reward, hole_reward, source_reward, city_dis)

        if done:
            #print("Episode finished after {} timesteps".format(time+1))
            break
    print str(agent_num) + '\t' + str(sum(agent_reward))
#draw.save_video("show", total_time)
#draw.save_video2("show", total_time)
