import copy, time
from collections import deque

import gym
from gym import spaces
from gym.utils import seeding

import config
import utils
from agent.agent_gym import AGENT_GYM
from agent.agent_gym_hole import AGENT_GYM as AGENT_GYM_HOLE
from policy import *
from myCallback import myTestLogger
from mcts import MyMCTS

class ENV_GYM(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.action_space = spaces.Discrete(len(config.Map.hole_pos))

        t = ()
        for i in range(config.Map.Height * config.Map.Width):
            t += (spaces.Discrete(4),)
        self.observation_space = spaces.Tuple(t)

        self._seed()

        self.env = None
        self.agent = None
        self.mask = None
        self.gamestep = 0
        self.invalid_count = 0
        self.conflict_count = 0
        self.hole_reward = None
        self.step_count = 0
        self.episode_count = 1

        self.used_agent = False

        self.new_city_hole = -np.ones((len(config.Map.hole_pos)), dtype=np.int8)
        self.onehot_city_hole = np.zeros((len(config.Map.hole_pos), len(config.Map.city_dis)), dtype=np.int8)
        self.mask = np.zeros((len(config.Map.hole_pos)), dtype=np.int8)
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

    def _reset(self):
        self.gamestep = 0
        self.invalid_count = 0
        self.conflict_count = 0
        self.new_city_hole = -np.ones((len(config.Map.hole_pos)), dtype=np.int8)
        self.onehot_city_hole = np.zeros((len(config.Map.hole_pos), len(config.Map.city_dis)), dtype=np.int8)
        self.mask = np.zeros((len(config.Map.hole_pos)), dtype=np.int8)
        return self.onehot_city_hole

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # print "action:", action
        done = (self.gamestep == (len(config.Map.hole_pos)-1))

        self.new_city_hole[action] = config.Map.hole_city[self.gamestep]
        self.mask[action] = 1
        self.onehot_city_hole[action][config.Map.hole_city[self.gamestep]] = 1

        self.step_count += 1
        self.gamestep += 1
        if done:
            reward = self._get_reward_from_agent()
            print "total time:", (time.time() - self.start_time) / 60
            print "env reward:", reward*10
        else:
            reward = 0

        return self.onehot_city_hole, reward, done, {}

    def _get_reward_from_agent(self):
        # TODO maybe could check valid map here
        # if want to enable DQN agent, change self.use_agent to True
        agent_gym = AGENT_GYM_HOLE(config.Map.source_pos, config.Map.hole_pos, config.Game.AgentNum, config.Game.total_time,
                              self.new_city_hole, config.Map.city_dis, self.trans, self.used_agent)
        agent_gym.agent = self.agent
        agent_gym.reset()

        self.episode_count += 1

        self.agent.reward_his.clear()
        # we do not reset the agent network, to accelerate the training.
        while True:
            if self.used_agent:
                self.agent.fit(agent_gym, nb_steps=10000, log_interval=10000, verbose=2)
            self.agent.reward_his.clear()
            testlogger = [myTestLogger()]
            self.agent.test_reward_his.clear()
            # print mazemap
            if self.used_agent:
                self.agent.test(agent_gym, nb_episodes=2, visualize=False, callbacks=testlogger, verbose=0)
            else:
                if (self.episode_count)%75 == 0:
                    self.agent.test(agent_gym, nb_episodes=1, visualize=False, callbacks=testlogger, verbose=-1)
                    print self.new_city_hole
                else:
                    self.agent.test(agent_gym, nb_episodes=1, visualize=False, callbacks=testlogger, verbose=0)
                    print self.new_city_hole
            self.hole_reward = agent_gym.hole_reward
            return np.mean(self.agent.test_reward_his)/10

    def get_mask(self):
        return self.mask
