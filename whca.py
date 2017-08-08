import gym
import numpy as np
from agent_gym import AGENT_GYM
import config, utils
import matplotTest as draw
import mapGenerator as MG
import copy

def encode(pos, t):
    [x, y] = pos
    return str(x) + ' ' + str(y) + ' ' + str(t)

class WHCA:

    def isValidPath(self, agent_id, pos, t, agent_pos, end_pos):
        if not utils.inMap(pos):
            #print 'not valid: out of map'
            return False
        elif encode(pos, t) in self.reserve and self.reserve[encode(pos, t)] != agent_id:
            #print 'not valid: reserved by ' + str(self.reserve[encode(pos, t)])
            return False
        elif encode(pos, t + 1) in self.reserve and self.reserve[encode(pos, t + 1)] != agent_id:
            #print 'not valid: reserved by ' + str(self.reserve[encode(pos, t + 1)])
            return False
        elif pos in agent_pos and agent_pos.index(pos) != agent_id:
            #print 'not valid: agent ' + str(agent_pos.index(pos))
            return False
        elif pos == end_pos:
            return True
        elif pos in self.source_pos or pos in self.hole_pos:
            #print 'not valid: source or hole'
            return False
        else:
            return True

    def AStar(self, agent_id, start_pos, end_pos, agent_pos, t0):
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
                if not self.isValidPath(agent_id, new_pos, t + 1, agent_pos, end_pos):
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

            if pos == end_pos or g[encode(pos, t)] == self.window or len(open_set) == 0:
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
        #print 'self.AStar'
        #print ['start_pos', start_pos, 'end_pos', end_pos]
        #print ['schedule', schedule]
        #print 'End self.AStar'
        return schedule

    def pathfinding(self, agent_pos, agent_city, agent_id, t):
        start_pos = agent_pos[agent_id]
        city = agent_city[agent_id]
        min_distance = -1
        if city == -1:
            for i in range(len(self.source_pos)):
                distance = utils.getDistance(self.source_pos[i], start_pos)
                if min_distance == -1 or distance < min_distance:
                    min_distance = distance
                    end_pos = self.source_pos[i]
        else:
            for i in range(len(self.hole_pos)):
                if self.hole_city[i] != city:
                    continue
                distance = utils.getDistance(self.hole_pos[i], start_pos)
                if min_distance == -1 or distance < min_distance:
                    min_distance = distance
                    end_pos = self.hole_pos[i]
        #print 'self.pathfinding'
        #print ['city', city]
        schedule = self.AStar(agent_id, start_pos, end_pos, agent_pos, t)
        #print 'end self.pathfinding'
        return schedule

    def removeSchedule(self, agent_id):
        #print 'remove schedule ' + str(agent_id)
        delset = {}
        for [pos, t, a] in self.schedule[agent_id]:
            delset[encode(pos, t)] = True
            delset[encode(pos, t + 1)] = True
        for entry in delset:
            del self.reserve[entry]
        self.schedule[agent_id] = []

    def addReserve(self, entry, agent_id):
        if entry in self.reserve:
            i = self.reserve[entry]
            #print 'entry collision ' + str(entry) + ' new ' + str(agent_id) + ' old ' + str(i)
            if agent_id != i:
                self.removeSchedule(i)
        self.reserve[entry] = agent_id

    def __init__(self, window, source_pos, hole_pos, hole_city, agent_num):
        self.window = window
        self.source_pos = source_pos
        self.hole_pos = hole_pos
        self.hole_city = hole_city
        self.agent_num = agent_num
        self.reserve = {}
        self.schedule = [[]] * agent_num


    def getJointAction(self, agent_pos, agent_city, time):

        for i in range(self.agent_num):
            self.addReserve(encode(agent_pos[i], time + 1), i)
            self.addReserve(encode(agent_pos[i], time + 2), i)

        action = []
        for i in range(self.agent_num):
            #print 'agent ' + str(i) + ' in pos ' + str(agent_pos[i])
            if len(self.schedule[i]) == 0:
                self.schedule[i] = self.pathfinding(agent_pos, agent_city, i, time)
                for [pos, t, a] in self.schedule[i]:
                    self.addReserve(encode(pos, t), i)
                    self.addReserve(encode(pos, t + 1), i)
            elif len(self.schedule[i]) == 1:
                self.removeSchedule(i)
                self.schedule[i] = self.pathfinding(agent_pos, agent_city, i, time)
                for [pos, t, a] in self.schedule[i]:
                    self.addReserve(encode(pos, t), i)
                    self.addReserve(encode(pos, t + 1), i)
            else:
                [pos, t, a] = self.schedule[i][0]
                if pos != agent_pos[i]:
                    self.removeSchedule(i)
                    self.schedule[i] = self.pathfinding(agent_pos, agent_city, i, time)
                    for [pos, t, a] in self.schedule[i]:
                        self.addReserve(encode(pos, t), i)
                        self.addReserve(encode(pos, t + 1), i)
            #print ['schedule', self.schedule[i]]
            [pos, t, a] = self.schedule[i][0]
            del self.schedule[i][0]
            del self.reserve[encode(pos, t)]
            if a != [0, 0]:
                del self.reserve[encode(pos, t + 1)]
            action.append(a)

        return action



