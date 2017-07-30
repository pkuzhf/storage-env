import numpy as np
import config, utils, copy

import gym
from gym import spaces
from rl.core import Processor
from gym.utils import seeding
from collections import deque

class AGENT_GYM(gym.Env):

    metadata = {'render.modes': ['human']}


    def initAgent(self, hole_pos, source_pos, agent_num):
        agent_pos = []
        agent_city = []
        agent_reward = []
        for i in range(agent_num):
            while True:
                x = np.random.random_integers(0, config.Map.Height-1)
                y = np.random.random_integers(0, config.Map.Width-1)
                if [x,y] not in hole_pos and [x,y] not in source_pos and [x,y] not in agent_pos:
                    agent_pos.append([x,y])
                    agent_city.append(-1)
                    agent_reward.append(0)
                    break
        self.agent_pos = agent_pos
        self.agent_city = agent_city
        self.agent_reward = agent_reward

    def genCity(city_dis):
        return np.where(np.random.multinomial(1, city_dis, size=1) == 1)[0][0]

    def __init__(self, source_pos, hole_pos, agent_num, total_time, hole_city, city_dis):

        # t = ()
        # for i in range(agent_num):
        #     t += (spaces.Discrete(config.Game.AgentAction),)
        # self.action_space = spaces.Tuple(t)

        # t = ()
        # for i in range(config.Map.Height * config.Map.Width):
        #     t += (spaces.Discrete(utils.Cell.CellSize),)
        
        # n_hole = utils.calcHole(ini_map)

        # for i in range(n_hole):
        #     t += (spaces.Discrete(city_num),)

        # for i in range(agent_num):
        #     t += (spaces.Discrete(config.Map.Height),)
        #     t += (spaces.Discrete(config.Map.Width),)
        #     t += (spaces.Discrete(city_num+1),)

        # self.observation_space = spaces.Tuple(t)

        self._seed()

        self.source_pos = source_pos
        self.total_time = total_time
        self.hole_pos = hole_pos
        self.hole_city = hole_city
        self.agent_num = agent_num
        self.city_dis = city_dis

        print np.where(self.source_pos == [0,0])[0]
        print np.where(self.hole_pos == [4,0])[0]

        self.time = 0
        self.initAgent(self.hole_pos, self.source_pos, self.agent_num)


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.time = 0
        self.initAgent(self.hole_pos, self.source_pos, self.agent_num)
        return [self.agent_pos, self.agent_city, self.agent_reward]

    def _step(self, action):

        reward = 0

        for i in range(self.agent_num):
            r = 0
            [pos_x, pos_y] = self.agent_pos[i]
            [a_x, a_y] = action[i]
            pos = [pos_x + a_x, pos_y + a_y]
            if utils.inMap(pos):
                if len(np.where(self.source_pos == pos)[0]) > 0: # source
                    print([i, 'source'])
                    if self.agent_city[i] == -1:
                        self.agent_pos[i] = pos
                        self.agent_city[i] = genCity(self.city_dis)
                elif len(np.where(self.hole_pos == pos)[0]) > 0: # hole
                    hole_idx = np.where(self.hole_pos == pos)[0][0]
                    print([i, 'hole'])
                    print(self.agent_city[i], self.hole_city[hole_idx])
                    if self.agent_city[i] == self.hole_city[hole_idx]:
                        self.agent_pos[i] = pos
                        self.agent_city[i] = -1
                        self.agent_reward[i] += 1
                        reward += 1
                elif len(np.where(self.agent_pos == pos)[0]) == 0: # path (not agent)
                    self.agent_pos[i] = pos
                else:
                    print([i, 'agent'])

        self.time += 1
        if self.time  == self.total_time:
            done = True
        else:
            done = False



        return [self.agent_pos, self.agent_city, self.agent_reward], reward, done, {}

