import numpy as np
import config, utils, copy

import gym
from gym import spaces
from rl.core import Processor
from gym.utils import seeding
from collections import deque
from whca import WHCA
from result_visualizer import ResultVisualizer


class AGENT_GYM(gym.Env):

    metadata = {'render.modes': ['human']}


    def init(self, hole_pos, source_pos, agent_num):
        self.time = 0
        self.agent_pos = []
        for i in range(agent_num):
            while True:
                x = np.random.random_integers(0, config.Map.Width-1)
                y = np.random.random_integers(0, config.Map.Height-1)
                if [x,y] not in hole_pos and [x,y] not in source_pos and [x,y] not in self.agent_pos:
                    self.agent_pos.append([x,y])
                    break
        self.agent_city = [-1] * agent_num
        self.agent_reward = [0] * agent_num
        self.hole_reward = [0] * len(hole_pos)
        self.source_reward = [0] * len(source_pos)
        self.whca = WHCA(self.window, source_pos, hole_pos, self.hole_city, agent_num)
        # print self.agent_pos


    def genCity(self, city_dis):
        return np.random.multinomial(1, city_dis, size=1).tolist()[0].index(1)

    def __init__(self, source_pos, hole_pos, agent_num, total_time, hole_city, city_dis, window):

        self._seed()

        self.source_pos = source_pos
        self.total_time = total_time
        self.hole_pos = hole_pos
        self.hole_city = hole_city
        self.agent_num = agent_num
        self.city_dis = city_dis

        self.window = window

        self.init(self.hole_pos, self.source_pos, self.agent_num)

        self.whca = WHCA(window, source_pos, hole_pos, hole_city, agent_num)

        self.steps = 0
        self.visualizer = ResultVisualizer([config.Map.Width, config.Map.Height], source_pos, hole_pos,
                                           hole_city, city_dis, agent_num, "environment/result")

    def EnableResultVisualizer(self):
        self.visualizer.enable = True
        self.visualizer.wirte_static_info()


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.init(self.hole_pos, self.source_pos, self.agent_num)
        # return [self.agent_pos, self.agent_city, self.agent_reward, self.hole_reward, self.source_reward]
        if self.visualizer.enable:
            self.visualizer.write_ob(self.steps+1,self.agent_pos, self.agent_city, self.agent_reward,
                                     self.hole_reward, self.source_reward, 0)
        return self.format_ob()

    def _step(self, action):
        dir = [[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]]
        rewards = [-0.01]*self.agent_num

        end_pos = []
        for i in range(self.agent_num):
            if action[i]<5:
                end_pos.append([self.agent_pos[i][0]+dir[action[i]][0],self.agent_pos[i][1]+dir[action[i]][1]])
            elif action[i]<5+len(self.source_pos):
                if self.agent_city[i]!=-1:
                    rewards[i]-=1
                end_pos.append(self.source_pos[action[i]-5])
            else:
                if self.agent_city[i]==-1:
                    rewards[i]-=1
                end_pos.append(self.hole_pos[action[i]-len(self.source_pos)-5])


        astarAction = self.whca.getJointAction(self.agent_pos, self.agent_city, end_pos, self.steps % self.total_time)
        self.steps += 1

        for i in range(self.agent_num):
            pos = self.agent_pos[i]
            a = astarAction[i]

            if a == [0, 0]:
                continue
            pos = [pos[0] + a[0], pos[1] + a[1]]
            #print(['agent ', i, ' try to move to ', pos])
            if utils.inMap(pos):
                if pos in self.agent_pos:
                    rewards[i] -= 1
                    #print('agent collision')
                    pass
                elif pos in self.source_pos: # source
                    source_idx = self.source_pos.index(pos)
                    if self.agent_city[i] == -1:
                        self.agent_pos[i] = pos
                        self.agent_city[i] = self.genCity(self.city_dis)
                        self.source_reward[source_idx] += 1
                        rewards[i] += 10
                        #print('enter source')
                    else:
                        rewards[i] -= 1
                        #print('source collision')
                elif pos in self.hole_pos: # hole
                    hole_idx = self.hole_pos.index(pos)
                    if self.agent_city[i] == self.hole_city[hole_idx]:
                        self.agent_pos[i] = pos
                        self.agent_city[i] = -1
                        self.agent_reward[i] += 1
                        self.hole_reward[hole_idx] += 1
                        rewards[i] += 10
                        #print('enter hole')
                    else:
                        rewards[i] -= 1
                        #print('hole collision')
                else:
                    #print('move from ' + str(self.agent_pos[i]) + ' to ' + str(pos))
                    self.agent_pos[i] = pos
            else:
                rewards[i] -= 1
                # print('out of map')

        self.time += 1
        if self.time == self.total_time:
            done = True
        else:
            done = False

        if self.visualizer.enable:
            self.visualizer.write_ob(self.steps,self.agent_pos, self.agent_city, self.agent_reward,
                                     self.hole_reward, self.source_reward, sum(rewards))

        return self.format_ob(), np.array(rewards), done, {}
        # return [self.agent_pos, self.agent_city, self.agent_reward, self.hole_reward, self.source_reward], rewards, done, {}

    def format_ob(self):
        formated_ob = np.zeros((self.agent_num,config.Map.Width,config.Map.Height,8))
        table = self.whca.getScheduleTable4(self.steps % self.total_time)
        for i in range(self.agent_num):
            formated_ob[i] += table
            formated_ob[i][self.agent_pos[i][0]][self.agent_pos[i][1]][0] = 1
            # end
            if (self.agent_city[i]==-1):
                for j in range(len(self.source_pos)):
                    formated_ob[i][self.source_pos[j][0]][self.source_pos[j][1]][1] = 1
            else:
                for j in range(len(self.hole_pos)):
                    if self.hole_city[j] == self.agent_city[i]:
                        formated_ob[i][self.hole_pos[j][0]][self.hole_pos[j][1]][1] = 1
            # wall (can be combine with above)
            if (self.agent_city[i]==-1):
                for j in range(len(self.hole_pos)):
                    formated_ob[i][self.hole_pos[j][0]][self.hole_pos[j][1]][2] = 1
            else:
                for j in range(len(self.hole_pos)):
                    if self.hole_city[j] != self.agent_city[i]:
                        formated_ob[i][self.hole_pos[j][0]][self.hole_pos[j][1]][2] = 1
                for j in range(len(self.source_pos)):
                    formated_ob[i][self.source_pos[j][0]][self.source_pos[j][1]][2] = 1
            # others
            for j in range(self.agent_num):
                if i == j:
                    continue
                formated_ob[i][self.agent_pos[j][0]][self.agent_pos[j][1]][3] = 1

        return formated_ob
