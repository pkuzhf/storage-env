import copy, time
from collections import deque

import gym
from gym import spaces
from gym.utils import seeding

import config
import utils
from agent.agent_gym import AGENT_GYM
from policy import *
from myCallback import myTestLogger
from mcts import MyMCTS

class ENV_GYM(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._seed()

        self.env = None
        self.agent = None
        self.mask = None
        self.gamestep = 0
        self.invalid_count = 0
        self.conflict_count = 0
        self.reward_his = deque(maxlen=1000)
        self.step_count = 0

        self.used_agent = False

        self.actions_to_paths = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
                        [1,1,0,0],[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1],[0,0,1,1],
                        [1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1]])
        self.start_time = time.time()

        self.reward_history = open("reward_his",'w')
        self.reward_history.write('')
        self.reward_history.close()
        self.reward_history = open("reward_his",'a')

    def _reset(self):
        self.gamestep = 0
        self.invalid_count = 0
        self.conflict_count = 0
        self.pathes = np.zeros([config.Map.Width, config.Map.Height, 4], dtype=np.int32)
        self.grid_type = np.zeros([config.Map.Width, config.Map.Height, 3], dtype=np.int32)
        self.grid_type[0][0] = np.zeros((3, ))
        for i in range(config.Map.Width):
            for j in range(config.Map.Height):
                self.grid_type[i][j][0] = 1
        for sp in config.Map.source_pos:
            self.grid_type[sp[0]][sp[1]][1] = 1
            self.grid_type[sp[0]][sp[1]][0] = 0
        for hp in config.Map.hole_pos:
            self.grid_type[hp[0]][hp[1]][2] = 1
            self.grid_type[hp[0]][hp[1]][0] = 0

        self.last_pos = np.zeros([config.Map.Width, config.Map.Height, 1], dtype=np.int32)
        mid = np.concatenate((self.pathes, self.grid_type), axis=2)
        # return np.concatenate((mid, self.last_pos), axis=2)
        return self.pathes

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        print "action:", action
        done = (self.gamestep == config.Map.Width*config.Map.Height * 32)
        current_x = action/10/config.Map.Height
        current_y = (action/10)%config.Map.Height
        self.pathes[current_x][current_y] = self.actions_to_paths[action % 10 + 4]

        self.step_count += 1
        self.gamestep += 1

        pathes = copy.deepcopy(self.pathes)
        reward, ob = self._get_reward_from_agent(pathes)

        self.last_pos = np.zeros([config.Map.Width, config.Map.Height, 1], dtype=np.int32)
        for i in range(ob.shape[0]):
            self.last_pos += ob[i][0][:][:]

        print "total time:", (time.time()-self.start_time)/60
        print "reward:", reward

        mid = np.concatenate((self.pathes, self.grid_type), axis=2)

        self.reward_history.write(str(reward)+'\n')

        return self.pathes, reward, done, {}
        # return np.concatenate((mid, self.last_pos), axis=2), reward, done, {}

    def _get_reward_from_agent(self, mazemap):
        # if want to enable DQN agent, change self.use_agent to True

        agent_gym = AGENT_GYM(config.Map.source_pos, config.Map.hole_pos, config.Game.AgentNum, config.Game.total_time,
                              config.Map.hole_city, config.Map.city_dis, mazemap, self.used_agent)
        agent_gym.agent = self.agent
        agent_gym.reset()

        bonus = 0
        self.agent.reward_his.clear()
        # we do not reset the agent network, to accelerate the training.
        while True:
            if self.used_agent:
                self.agent.fit(agent_gym, nb_steps=10000, log_interval=10000, verbose=2)
            self.agent.reward_his.clear()
            np.random.seed(None)
            bonus += 5
            testlogger = [myTestLogger()]
            self.agent.test_reward_his.clear()
            # print mazemap
            if self.used_agent:
                self.agent.test(agent_gym, nb_episodes=2, visualize=False, callbacks=testlogger, verbose=0)
            else:
                if self.step_count % (3000) == 0:
                    self.agent.test(agent_gym, nb_episodes=1, visualize=False, callbacks=testlogger, verbose=-1)
                else:
                    self.agent.test(agent_gym, nb_episodes=1, visualize=False, callbacks=testlogger, verbose=0)
            return np.mean(self.agent.test_reward_his)/20, self.agent.last_ob

