import gym
import numpy as np
from agent_gym import AGENT_GYM
import config, utils
import matplotTest as draw
import mapGenerator as MG
import copy

def encode(pos):
    [x, y] = pos
    return str(x) + ' ' + str(y)

def decode(m):
    m = m.split()
    pos = [int(m[0]), int(m[1])]
    return pos

class WHCA:

    def setSchedule(self, agent_id, schedule):
        for [start_pos, action] in schedule:
            time = len(self.schedule[agent_id])
            self.addReserve(time, start_pos, agent_id)
            self.schedule[agent_id].append([start_pos, action])

    def removeSchedule(self, agent_id, start_time = 0):
        #print 'remove schedule ' + str(agent_id)
        for time in range(start_time, len(self.schedule[agent_id])):
            [start_pos, action] = self.schedule[agent_id][time]
            entry = encode(start_pos)
            for i in self.reserve_interval:
                t = time + i
                if entry in self.reserve[t]:
                    del self.reserve[t][entry]
        self.schedule[agent_id] = self.schedule[agent_id][:start_time]
        if len(self.schedule[agent_id]) <= 1:
            schedule = []
            schedule.append([self.agent_pos[agent_id], [0, 0]])
            schedule.append([self.agent_pos[agent_id], None])
            self.setSchedule(agent_id, schedule)
        else:
            self.schedule[agent_id][-1][1] = None

    def addReserve(self, time, pos, agent_id):
        entry = encode(pos)
        for i in self.reserve_interval:
            t = time + i
            if t < 0:
                continue
            while t >= len(self.reserve):
                self.reserve.append({})
            if entry in self.reserve[t]:
                [reserved_agent_id, schedule_time] = self.reserve[t][entry]
                #print 'entry collision ' + str(entry) + ' new ' + str(agent_id) + ' old ' + str(i)
                if agent_id != reserved_agent_id:
                    self.removeSchedule(reserved_agent_id, schedule_time)
            self.reserve[t][entry] = [agent_id, time]

    def getScheduleTable(self, total_time_step):
        time = min(total_time_step, len(self.reserve))
        table = [[[0 for t in range(time)] for y in range(config.Map.Width)] for x in range(config.Map.Height)]
        for t in range(time):
            for entry in self.reserve[t]:
                [x, y] = decode(entry)
                table[x][y][t] = 1
        return table

    def isValidPath(self, agent_id, pos, t, agent_pos, end_pos):
        if not utils.inMap(pos):
            #print 'not valid: out of map'
            return False
        for i in self.reserve_interval:
            if encode(pos) in self.reserve[t] and self.reserve[t][encode(pos)][0] != agent_id:
                #print 'not valid: reserved by ' + str(self.reserve[encode(pos, t)])
                return False
        if pos == end_pos:
            return True
        if pos in self.source_pos or pos in self.hole_pos:
            #print 'not valid: source or hole'
            return False
        return True

    def AStar(self, agent_id, start_pos, end_pos, agent_pos):
        open_set = []
        close_set = []
        h = [{} for i in range(self.window + 2)]
        g = [{} for i in range(self.window + 2)]
        path = [{} for i in range(self.window + 2)]
        open_set.append([start_pos, 0])
        pos = start_pos
        g[0][encode(pos)] = 0
        h[0][encode(pos)] = utils.getDistance(start_pos, end_pos)
        path[0][encode(pos)] = [[-1, -1], -1, -1]
        schedule = []
        while len(open_set) > 0:
            min_f = -1
            idx = -1
            for i in range(len(open_set)):
                [pos, t] = open_set[i]
                f = g[t][encode(pos)] + h[t][encode(pos)]
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
                new_g = g[t][encode(pos)] + 1
                if [new_pos, t + 1] not in open_set:
                    open_set.append([new_pos, t + 1])
                    g[t+1][encode(new_pos)] = new_g
                    h[t+1][encode(new_pos)] = utils.getDistance(new_pos, end_pos)
                    path[t+1][encode(new_pos)] = [pos, t, a]
                elif new_g < g[t+1][encode(new_pos)]:
                    g[t+1][encode(new_pos)] = new_g
                    path[t+1][encode(new_pos)] = [pos, t, a]

            if pos == end_pos or g[t][encode(pos)] == self.window or len(open_set) == 0:
                while (path[t][encode(pos)] != [[-1, -1], -1, -1]):
                    schedule.insert(0, path[t][encode(pos)])
                    [pos, t, a] = path[t][encode(pos)]
                break
                
        if len(schedule) == 0:
            #print 'fail to schedule, stay'
            #print path
            schedule.append([start_pos, t, [0, 0]])
        [pos, t, a] = schedule[-1]
        new_pos = [pos[0] + a[0], pos[1] + a[1]]
        schedule.append([new_pos, t + 1, None])
        #print 'self.AStar'
        #print ['start_pos', start_pos, 'end_pos', end_pos]
        #print ['schedule', schedule]
        #print 'End self.AStar'
        return schedule

    def findEndPos(self, city, agent_id):
        start_pos = self.agent_pos[agent_id]
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
    
        return end_pos

    def __init__(self, window, source_pos, hole_pos, hole_city, agent_num, reserve_interval = [0, 2]):
        self.window = window
        self.source_pos = source_pos
        self.hole_pos = hole_pos
        self.hole_city = hole_city
        self.agent_num = agent_num
        self.reserve = []
        self.schedule = [[]] * agent_num
        self.agent_pos = None
        self.reserve_interval = reserve_interval

    def getJointAction(self, agent_pos, agent_city, end_poses):

        self.agent_pos = agent_pos

        for i in range(self.agent_num):
            self.addReserve(0, agent_pos[i], i)

        action = []
        for i in range(self.agent_num):
            #print 'agent ' + str(i) + ' in pos ' + str(agent_pos[i])
            if len(self.schedule[i]) <= 1 or end_poses[i] != [-1, -1]:
                self.removeSchedule(i)
                if end_poses[i] == [-1, -1]:
                    end_pos = findEndPos(agent_city[i], i)
                else:
                    end_pos = end_poses[i]
                schedule = self.AStar(i, agent_pos[i], end_pos, agent_pos)
                self.setSchedule(i, schedule)
            if self.schedule[i][0][0] != agent_pos[i]:
                print 'unexpected agent pos'
                print i
                print self.schedule[i][0][0]
                print agenet_pos[i]
            #print ['schedule', self.schedule[i]]
            [pos, t, a] = self.schedule[i][0]
            del self.schedule[i][0]
            action.append(a)

        del self.reserve[0]
        return action


def main():
    window = 10
    source_pos = [0, 0]
    hole_pos = [3, 3]
    hole_city = [0]
    agent_num = 2
    whca = WHCA(window, source_pos, hole_pos, hole_city, agent_num)
    agent_pos = [[0, 0], [0, 1]]
    agent_city = [-1, -1]
    end_poses = [hole_pos, hole_pos]
    print whca.getJointAction(agent_pos, agent_city, end_poses)
    print whca.schedule

if __name__ == "__main__":
    main()