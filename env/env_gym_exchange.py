import copy, time
from collections import deque

import gym
from gym import spaces
from gym.utils import seeding

import config
import utils
from agent.agent_gym_exchange import AGENT_GYM as AGENT_GYM_HOLE
from policy import *
from myCallback import myTestLogger
import numpy as np
import os

class assign:
    def __init__(self, hole_city, hole_value):
        self.hole_city = hole_city
        self.hole_value = hole_value

class ENV_GYM(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        #
        # self.action_space = spaces.Discrete(len(config.Map.hole_pos))
        #
        # t = ()
        # for i in range(config.Map.Height * config.Map.Width):
        #     t += (spaces.Discrete(4),)
        # self.observation_space = spaces.Tuple(t)
        self._seed()

        self.env = None
        self.agent = None
        self.gamestep = 0
        self.hole_reward = None
        self.episode_count = 1

        self.used_agent = False

        trans = np.ones((config.Map.Width, config.Map.Height, 4), dtype=np.int8)
        for i in range(0, config.Map.Width):
            for j in range(0, config.Map.Height):
                if i % 2 == 0:
                    trans[i][j][1] = 0
                else:
                    trans[i][j][3] = 0
                if j % 2 == 0:
                    trans[i][j][2] = 0
                else:
                    trans[i][j][0] = 0
        self.trans = trans
        self.start_time = time.time()

        dis = open("distance20.txt", 'r')
        self.distance = eval(dis.readline())
        dis.close()

        self.nb_city = len(config.Map.city_dis)
        self.nb_hole = len(config.Map.hole_pos)
        self.np_city_dis = np.array(config.Map.city_dis)

        self.current_city = 0
        self.assigning = np.zeros((self.nb_hole, ), dtype=np.int32)
        self.occupied = np.zeros((self.nb_hole, ), dtype=np.int32)
        self.new_hole_city = self.nb_city * np.ones((self.nb_hole, ), dtype=np.int32)
        # TODO problem
        self.switch = [6, 11, 15, 18]

        self.mask = np.ones((self.nb_hole, ), dtype=np.int32)

    def _reset(self):
        self.gamestep = 0

        self.current_city = 0
        self.assigning = np.zeros((self.nb_hole,), dtype=np.int32)
        self.occupied = np.zeros((self.nb_hole,), dtype=np.int32)
        self.new_hole_city = self.nb_city * np.ones((self.nb_hole,), dtype=np.int32)

        self.mask = np.ones((self.nb_hole,), dtype=np.int32)

        return np.array([self.assigning, self.occupied], dtype=np.int32)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # print "action:", action

        if self.gamestep in self.switch:
            self.current_city += 1
            self.occupied += self.assigning
            self.assigning = np.zeros((self.nb_hole,), dtype=np.int32)

        done = (self.gamestep == self.nb_hole - 1)

        self.assigning[action] = 1
        self.new_hole_city[action] = self.current_city
        self.mask[action] = 0

        if done:
            # reward, hole_reward = self._get_reward_from_agent()
            reward = self.get_reward_from_distance(self.random_fill())
            v = self.get_reward_from_distance(self.random_fill())
            print "total time:", (time.time() - self.start_time) / 60
            print "env reward:", reward
            return np.array([self.assigning, self.occupied], dtype=np.int32), reward, done, v
        else:
            # while len(self.new_hole_city)<len(config.Map.hole_pos):
            #     self.new_hole_city.append(np.random.choice(range(len(config.Map.city_dis))))
            # reward = self._get_reward_from_agent()
            # self.new_hole_city = self.new_hole_city[:self.gamestep]
            if self.gamestep+1 in self.switch:
                v = 0
                for i in range(10):
                    # v2 = 1280.0/self.get_hole_dis(self.random_fill())
                    v2 = self.get_reward_from_distance(self.random_fill())
                    if v2>v:
                        v = v2
                # print "v:", v, self.gamestep, self.current_city
            else:
                v = 0
            reward = 0

        self.gamestep += 1
        return np.array([self.assigning, self.occupied], dtype=np.int32), reward, done, v

    def _get_reward_from_agent(self, city_hole):
        # if want to enable DQN agent, change self.use_agent to True
        agent_gym = AGENT_GYM_HOLE(config.Map.source_pos, config.Map.hole_pos, config.Game.AgentNum, config.Game.total_time,
                                   city_hole, config.Map.city_dis, self.trans, self.used_agent)
        agent_gym.agent = self.agent
        agent_gym.reset()

        if -1 in self.new_hole_city:
            print "fatal problem! hole not fully assigned!"
            print self.gamestep
            print self.gamestep % 210
            print "hole", self.new_hole_city
            print "mask", self.mask
            assert 0

        self.episode_count += 1

        self.agent.reward_his.clear()
        # we do not reset the agent network, to accelerate the training.

        if self.used_agent:
            self.agent.fit(agent_gym, nb_steps=10000, log_interval=10000, verbose=2)
        self.agent.reward_his.clear()
        testlogger = [myTestLogger()]
        self.agent.test_reward_his.clear()
        # print mazemap
        if self.used_agent:
            self.agent.test(agent_gym, nb_episodes=2, visualize=False, callbacks=testlogger, verbose=0)
        else:
            # if (self.episode_count)%10000 == 0:
            #     self.agent.test(agent_gym, nb_episodes=1, visualize=False, callbacks=testlogger, verbose=-1)
            #     print self.new_hole_city.tolist()
            # else:
            self.agent.test(agent_gym, nb_episodes=1, visualize=False, callbacks=testlogger, verbose=0)
                # print self.new_hole_city

        return np.mean(self.agent.test_reward_his)/1000, np.array(agent_gym.hole_reward)/1000

    def get_reward_from_distance(self, city_hole):
        max_distance = 50
        mins = max_distance * np.ones((config.Source_num, self.nb_city))
        for i in range(config.Source_num):
            for j in range(config.Hole_num):
                new_dis = self.distance[i][j]
                if new_dis<mins[i][city_hole[j]]:
                    mins[i][city_hole[j]] = new_dis
            mins[i] *= self.np_city_dis
        avg_dis = np.sum(mins) / config.Source_num
        total_step = config.Game.AgentNum * config.Game.total_time
        return total_step / avg_dis

    def random_fill(self):
        filled_city = np.zeros((self.nb_hole), dtype=np.int32)
        empty_hole = []
        for i in range(self.nb_hole):
            if self.new_hole_city[i]<self.nb_city:
                filled_city[i] = self.new_hole_city[i]
            else:
                empty_hole.append(i)
        current_city = 0
        for i in range(0, self.nb_hole):
            if i in self.switch:
                current_city += 1
            if i > self.gamestep and current_city < self.nb_city:
                hole_id = np.random.randint(len(empty_hole))
                filled_city[hole_id] = current_city
                del empty_hole[hole_id]
        return filled_city
