import numpy as np
import config, utils, copy
import time as ttime

import gym
from gym import spaces
from rl.core import Processor
from gym.utils import seeding
from collections import deque
# from whca import WHCA
from result_visualizer import ResultVisualizer


class AGENT_GYM(gym.Env):
    metadata = {'render.modes': ['human']}

    def init(self, hole_pos, source_pos, agent_num):
        self.time = 0
        self.agent_pos = []
        for i in range(agent_num):
            while True:
                x = np.random.random_integers(0, config.Map.Width - 1)
                y = np.random.random_integers(0, config.Map.Height - 1)
                if [x, y] not in hole_pos and [x, y] not in source_pos and [x, y] not in self.agent_pos:
                    self.agent_pos.append([x, y])
                    break
        self.agent_city = [-1] * agent_num
        self.agent_reward = [0] * agent_num
        self.hole_reward = [0] * len(hole_pos)
        self.source_reward = [0] * len(source_pos)

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

        self.trans = [
            [[[1, 0]], [[1, 0], [0, 1]], [[1, 0]], [[0, 1]], [[0, 1]], [[0, 1]]],
            [[[1, 0]], [[1, 0]], [[0, -1], [1, 0]], [[0, 1], [1, 0]], [[0, 1]], [[0, 1], [-1, 0]]],
            [[[1, 0]], [[1, 0], [0, -1]], [[0, -1]], [[1, 0]], [[1, 0], [0, 1]], [[0, 1]]],
            [[[0, -1]], [[-1, 0], [0, -1]], [[-1, 0]], [[0, 1]], [[-1, 0], [0, 1]], [[-1, 0]]],
            [[[0, -1], [1, 0]], [[0, -1]], [[0, -1], [-1, 0]], [[0, 1], [-1, 0]], [[-1, 0]], [[-1, 0]]],
            [[[0, -1]], [[0, -1]], [[0, -1]], [[-1, 0]], [[-1, 0], [0, -1]], [[-1, 0]]]
        ]

        self.init(self.hole_pos, self.source_pos, self.agent_num)

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
        if self.visualizer.enable:
            self.visualizer.write_ob(self.steps + 1, self.agent_pos, self.agent_city, self.agent_reward,
                                     self.hole_reward, self.source_reward, 0)
        return self.format_ob()

    def _step(self, action):
        dir = [[1, 0], [0, 1], [-1, 0], [0, -1]]

        rewards = [-0.2]*self.agent_num
        hit_wall = 1
        illegal = 1
        pick_drop = 5

        agent_next_pos = []
        done = [False] * len(action)

        self.steps += 1

        # invalid
        for i in range(self.agent_num):
            pos = self.agent_pos[i]
            a = dir[action[i]]
            if a not in self.trans[self.agent_pos[i][1]][self.agent_pos[i][0]]:
                rewards[i] -= illegal
            # TODO simple resolution
            next_pos = [pos[0] + a[0], pos[1] + a[1]]
            if next_pos not in agent_next_pos:
                agent_next_pos.append(next_pos)
            else:
                agent_next_pos.append(pos)
            if pos == agent_next_pos[i]:
                done[i] = True
            elif not utils.inMap(agent_next_pos[i]):
                agent_next_pos[i] = self.agent_pos[i]
                done[i] = True
                rewards[i] -= hit_wall
            elif agent_next_pos[i] in self.source_pos and self.agent_city[i] != -1:
                agent_next_pos[i] = self.agent_pos[i]
                done[i] = True
                rewards[i] -= hit_wall
            elif agent_next_pos[i] in self.hole_pos and self.agent_city[i] != self.hole_city[
                self.hole_pos.index(agent_next_pos[i])]:
                agent_next_pos[i] = self.agent_pos[i]
                done[i] = True
                rewards[i] -= hit_wall

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
                    print i,j
                    print line
                    print self.agent_pos
                    print agent_next_pos
                    print done
                line.append(j)
                j = self.agent_pos.index(agent_next_pos[j])
            if not done[j]:
                line.append(j)
            collision = False
            for k in range(self.agent_num):
                if done[k] and agent_next_pos[k] == agent_next_pos[j]:
                    collision = True
                    break
            for k in range(len(line)):
                if collision:
                    agent_next_pos[line[k]] = self.agent_pos[line[k]]
                done[line[k]] = True

        if False in done:
            print 'error: False in done'
            print self.agent_pos
            print agent_next_pos
            print done

        self.agent_pos = agent_next_pos

        pack_count = []

        for i in range(self.agent_num):
            pack_count.append(0)
            pos = self.agent_pos[i]
            # a = action[i]
            # if a == [0, 0]:
            #     continue
            if pos in self.source_pos and self.agent_city[i]==-1:  # source
                source_idx = self.source_pos.index(pos)
                self.agent_city[i] = self.genCity(self.city_dis)
                self.source_reward[source_idx] += 1
                rewards[i] += pick_drop
            elif pos in self.hole_pos and self.agent_city[i]!=-1:  # hole
                hole_idx = self.hole_pos.index(pos)
                self.agent_city[i] = -1
                self.agent_reward[i] += 1
                self.hole_reward[hole_idx] += 1
                pack_count[-1] = 1
                rewards[i] += pick_drop

        self.time += 1
        if self.time == self.total_time:
            done = True
        else:
            done = False

        if self.visualizer.enable:
            self.visualizer.write_ob(self.steps,self.agent_pos, self.agent_city, self.agent_reward,
                                     self.hole_reward, self.source_reward, sum(pack_count))
        return self.format_ob(), np.array(rewards), done, {}
        # return [self.agent_pos, self.agent_city, self.agent_reward, self.hole_reward, self.source_reward], rewards, done, {}

    def format_ob(self):
        formated_ob = np.zeros((self.agent_num, 4, config.Map.Width, config.Map.Height, 1))
        for i in range(self.agent_num):
            formated_ob[i][0][self.agent_pos[i][0]][self.agent_pos[i][1]][0] = 1
            # end
            if (self.agent_city[i] == -1):
                for j in range(len(self.source_pos)):
                    formated_ob[i][1][self.source_pos[j][0]][self.source_pos[j][1]][0] = 1
            else:
                for j in range(len(self.hole_pos)):
                    if self.hole_city[j] == self.agent_city[i]:
                        formated_ob[i][1][self.hole_pos[j][0]][self.hole_pos[j][1]][0] = 1
            # wall (can be combine with above)
            if (self.agent_city[i] == -1):
                for j in range(len(self.hole_pos)):
                    formated_ob[i][2][self.hole_pos[j][0]][self.hole_pos[j][1]][0] = 1
            else:
                for j in range(len(self.hole_pos)):
                    if self.hole_city[j] != self.agent_city[i]:
                        formated_ob[i][2][self.hole_pos[j][0]][self.hole_pos[j][1]][0] = 1
                for j in range(len(self.source_pos)):
                    formated_ob[i][2][self.source_pos[j][0]][self.source_pos[j][1]][0] = 1
            # others
            for j in range(self.agent_num):
                if i == j:
                    continue
                formated_ob[i][3][self.agent_pos[j][0]][self.agent_pos[j][1]][0] = 1

        return formated_ob
