import numpy as np
import config, utils, copy

import gym
from gym import spaces
from rl.core import Processor
from gym.utils import seeding
from collections import deque

class AGENT_GYM(gym.Env):

    metadata = {'render.modes': ['human']}


    def init(self, hole_pos, source_pos, agent_num):
        self.time = 0
        self.agent_pos = []
        for i in range(agent_num):
            while True:
                x = np.random.random_integers(0, config.Map.Height-1)
                y = np.random.random_integers(0, config.Map.Width-1)
                if [x,y] not in hole_pos and [x,y] not in source_pos and [x,y] not in self.agent_pos:
                    self.agent_pos.append([x,y])
                    break
        self.agent_city = [-1] * agent_num
        self.agent_reward = [0] * agent_num
        self.hole_reward = [0] * len(hole_pos)
        self.source_reward = [0] * len(source_pos)

    def genCity(self, city_dis):
        return np.random.multinomial(1, city_dis, size=1).tolist()[0].index(1)

    def __init__(self, source_pos, hole_pos, agent_num, total_time, hole_city, city_dis):

        self._seed()

        self.source_pos = source_pos
        self.total_time = total_time
        self.hole_pos = hole_pos
        self.hole_city = hole_city
        self.agent_num = agent_num
        self.city_dis = city_dis

        self.init(self.hole_pos, self.source_pos, self.agent_num)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.init(self.hole_pos, self.source_pos, self.agent_num)
        return [self.agent_pos, self.agent_city, self.agent_reward, self.hole_reward, self.source_reward]

    def _step(self, action):

        agent_next_pos = []
        done = [False] * len(action)

        # invalid
        for i in range(self.agent_num):
            pos = self.agent_pos[i]
            a = action[i]
            agent_next_pos.append([pos[0] + a[0], pos[1] + a[1]])
            if a == [0, 0]:
                done[i] = True
            elif !utils.inMap(agent_next_pos[i]):
                agent_next_pos[i] = self.agent_pos[i]
                done[i] = True
            elif agent_next_pos[i] in self.source_pos and self.agent_city[i] != -1:
                agent_next_pos[i] = self.agent_pos[i]
                done[i] = True
            elif agent_next_pos[i] in self.hole_pos and self.agent_city[i] != self.hole_city[self.hole_pos.index[agent_next_pos[i]]]:
                agent_next_pos[i] = self.agent_pos[i]
                done[i] = True

        # circle
        for i in range(self.agent_num):
            if done[i]:
                continue
            circle = []
            j = i
            while not done[j] and j not in circle and agent_next_pos[j] in self.agent_pos:
                circle.append(j)
                j = self.agent_pos.index(agent_next_pos[j])
            if len(circle) > 0 and j == circle[0]:
                if len(circle) == 1:
                    print 'error: len(circle) == 1'
                if len(circle) == 2:
                    agent_next_pos[circle[0]] = self.agent_pos[circle[0]]
                    agent_next_pos[circle[1]] = self.agent_pos[circle[1]]
                    done[circle[0]] = True
                    done[circle[1]] = True
                else:
                    for k in range(len(circle)):
                        done[circle[k]] = True

        # line
        for i in range(self.agent_num):
            if done[i]:
                continue
            line = []
            j = i
            while not done[j] and agent_next_pos[j] in self.agent_pos:
                if j in line:
                    print 'error: duplicate in line'
                    print line
                    print self.agent_pos
                    print self.agent_next_pos
                    print done
                line.append(j)
                j = self.agent_pos.index(agent_next_pos[j])
            if done[j]:
                for k in range(len(line)):
                    if agent_next_pos[j] == self.agent_pos[j]:
                        agent_next_pos[k] = self.agent_pos[k]
                    done[circle[k]] = True
            else:
                line.append(j)
                collision = False
                for k in range(self.agent_num):
                    if done[k] and agent_next_pos[k] == agent_next_pos[j]:
                        collision = True
                        break
                for k in range(len(line)):
                    if collision:
                        agent_next_pos[k] = self.agent_pos[k]
                    done[circle[k]] = True
        
        if False in done:
            print 'error: False in done'
            print self.agent_pos
            print self.agent_next_pos
            print done
        
        self.agent_pos = agent_next_pos

        rewards = []

        for i in range(self.agent_num):
            rewards.append(0)
            pos = self.agent_pos[i]
            a = action[i]
            if a == [0, 0]:
                continue
            if pos in self.source_pos: # source
                self.agent_city[i] = self.genCity(self.city_dis)
                self.source_reward[source_idx] += 1
            elif pos in self.hole_pos: # hole
                hole_idx = self.hole_pos.index(pos)
                self.agent_city[i] = -1
                self.agent_reward[i] += 1
                self.hole_reward[hole_idx] += 1
                rewards[-1] = 1
            
        self.time += 1
        if self.time  == self.total_time:
            done = True
        else:
            done = False

        return [self.agent_pos, self.agent_city, self.agent_reward, self.hole_reward, self.source_reward], rewards, done, {}

