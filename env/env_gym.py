import copy
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

        self.action_space = spaces.Discrete(config.Map.Height*config.Map.Width+1)

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
        self.max_reward = -1e20
        self.reward_his = deque(maxlen=1000)

        self.used_agent = False

        self.mcts = MyMCTS()

        self.actions_to_paths = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
                        [1,1,0,0],[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1],[0,0,1,1],
                        [1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1]])

    def _reset(self):
        self.gamestep = 0
        self.invalid_count = 0
        self.conflict_count = 0
        self.pathes = -np.ones([config.Map.Width, config.Map.Height, 4], dtype=np.int64)

        # if 'Masked' in type(self.env.policy).__name__ or 'Masked' in type(self.env.test_policy).__name__:
        #     self.mask = self._getmask(self.pathes)
        #     self.env.policy.set_mask(self.mask)
        #     self.env.test_policy.set_mask(self.mask)
        return self.pathes

    # def _getmask(self, mazemap):
    #     mask = np.zeros(self.action_space.n)
    #     for i in range(config.Map.Width):
    #         for j in range(config.Map.Height):
    #             if not mazemap[i, j, utils.Cell.Empty]:
    #                 mask[i*config.Map.Width+j] = 1
    #     return mask

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        print("action:", action)
        done = (self.gamestep == config.Map.Width*config.Map.Height-1)

        self.pathes[self.gamestep/config.Map.Width][self.gamestep%config.Map.Height] = self.actions_to_paths[action]

        if done:
            pathes = copy.deepcopy(self.pathes)
            # utils.displayMap(self.pathes)
            reward = self._get_reward_from_agent(pathes)
            # if self.used_agent:
            #     print('agent rewards: ' + utils.string_values(self.agent.reward_his))
                      # +'   agent qvalues: ' + utils.string_values(self.agent.q_values))
        else:
            # MCTS reward
            target_node = self.mcts.SEARCHNODE(self.pathes)
            # print self.pathes
            # print target_node.state.moves
            print "step: ", target_node.state.step
            end_node = self.mcts.TREEPOLICYEND(target_node)
            # print end_node.state.moves
            # print end_node.state.step
            pathes = self.mcts.MOVETOPATH(end_node.state)
            # assert 0
            reward = self._get_reward_from_agent(pathes)
            self.mcts.BACKUP(end_node, reward)
            # reward = 0

        self.gamestep += 1

        return self.pathes, reward, done, {}

    def _get_reward_from_agent(self, mazemap):
        self.used_agent = True

        agent_gym = AGENT_GYM(config.Map.source_pos, config.Map.hole_pos, config.Game.AgentNum, config.Game.total_time,
                              config.Map.hole_city, config.Map.city_dis, mazemap)
        agent_gym.agent = self.agent
        agent_gym.reset()

        bonus = 0
        # self.agent.max_reward = -1e20
        self.agent.reward_his.clear()
        #self.agent.memory.__init__(config.Training.AgentBufferSize, window_length=1)
        # we do not reset the agent network, to accelerate the training.
        while True:
            self.agent.fit(agent_gym, nb_steps=7000, log_interval=10000, verbose=2)

            # print('agent rewards: ' + utils.string_values(self.agent.reward_his))
                  # + '   agent qvalues: ' + utils.string_values(self.agent.q_values))
            self.agent.reward_his.clear()
            np.random.seed(None)
            bonus += 5
            testlogger = [myTestLogger()]
            self.agent.test_reward_his.clear()
            self.agent.test(agent_gym, nb_episodes=2, visualize=False, callbacks=testlogger, verbose=0)
            print mazemap
            return np.mean(self.agent.test_reward_his)
