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

        rewards = []

        for i in range(self.agent_num):
            rewards.append(0)
            pos = self.agent_pos[i]
            a = action[i]
            if a == [0, 0]:
                continue
            pos = [pos[0] + a[0], pos[1] + a[1]]
            #print(['agent ', i, ' try to move to ', pos])
            if utils.inMap(pos):
                if pos in self.agent_pos:
                    #print('agent collision')
                    pass
                elif pos in self.source_pos: # source
                    source_idx = self.source_pos.index(pos)
                    if self.agent_city[i] == -1:
                        self.agent_pos[i] = pos
                        self.agent_city[i] = self.genCity(self.city_dis)
                        self.source_reward[source_idx] += 1
                        #print('enter source')
                    #else:
                        #print('source collision')
                elif pos in self.hole_pos: # hole
                    hole_idx = self.hole_pos.index(pos)
                    if self.agent_city[i] == self.hole_city[hole_idx]:
                        self.agent_pos[i] = pos
                        self.agent_city[i] = -1
                        self.agent_reward[i] += 1
                        self.hole_reward[hole_idx] += 1
                        rewards[-1] = 1
                        #print('enter hole')
                    #else:
                        #print('hole collision')
                else:
                    #print('move from ' + str(self.agent_pos[i]) + ' to ' + str(pos))
                    self.agent_pos[i] = pos
            #else:
                #print('out of map')
        self.time += 1
        if self.time  == self.total_time:
            done = True
        else:
            done = False



        return [self.agent_pos, self.agent_city, self.agent_reward, self.hole_reward, self.source_reward], rewards, done, {}
