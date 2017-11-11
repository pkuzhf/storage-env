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
        self.step_count = 0

        self.used_agent = False

        self.mcts = MyMCTS()

        self.actions_to_paths = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
                        [1,1,0,0],[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1],[0,0,1,1],
                        [1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1]])
        self.start_time = time.time()

    def _reset(self):
        self.gamestep = 0
        self.invalid_count = 0
        self.conflict_count = 0
        self.pathes = -np.ones([config.Map.Width, config.Map.Height, 4], dtype=np.int32)
        self.grid_type = -np.ones([config.Map.Width, config.Map.Height, 3], dtype=np.int32)
        self.grid_type[0][0] = np.zeros((3, ))
        if [0,0] in config.Map.source_pos:
            self.grid_type[0][0][1] = 1
        elif [0,0] in config.Map.hole_pos:
            self.grid_type[0][0][2] = 1
        else:
            self.grid_type[0][0][0] = 1
        return np.concatenate((self.pathes, self.grid_type), axis=2)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # print "action:", action
        done = (self.gamestep == config.Map.Width*config.Map.Height-1)
        current_x = self.gamestep/config.Map.Height
        current_y = self.gamestep%config.Map.Height
        self.pathes[current_x][current_y] = self.actions_to_paths[action]

        self.step_count += 1
        self.gamestep += 1
        if not done:
            current_x = self.gamestep/config.Map.Height
            current_y = self.gamestep%config.Map.Height
            self.grid_type[current_x][current_y] = np.zeros((3, ))
            if [current_x,current_y] in config.Map.source_pos:
                self.grid_type[current_x][current_y][1] = 1
            elif [current_x,current_y] in config.Map.hole_pos:
                self.grid_type[current_x][current_y][2] = 1
            else:
                self.grid_type[current_x][current_y][0] = 1

        if done:
            pathes = copy.deepcopy(self.pathes)
            reward = self._get_reward_from_agent(pathes)
        else:
            # MCTS reward
            # print self.pathes
            target_node = self.mcts.SEARCHNODE(self.pathes)
            # print target_node.state.moves
            # print "step: ", target_node.state.step
            end_node = self.mcts.TREEPOLICYEND(target_node)
            pathes = self.mcts.MOVETOPATH(end_node.state)
            reward = self._get_reward_from_agent(pathes)
            if reward > 20:
                print pathes
            self.mcts.BACKUP(end_node, reward)
            reward += 0.25 * target_node.reward / target_node.visits
            # reward = 0


        print "total time:", (time.time()-self.start_time)/60
        print "env reward:", reward

        return np.concatenate((self.pathes, self.grid_type),axis=2), reward, done, {}

    def _get_reward_from_agent(self, mazemap):
        # TODO maybe could check valid map here
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
                if self.step_count % (36 * 20) == 0:
                    self.agent.test(agent_gym, nb_episodes=1, visualize=False, callbacks=testlogger, verbose=-1)
                else:
                    self.agent.test(agent_gym, nb_episodes=1, visualize=False, callbacks=testlogger, verbose=0)
            return np.mean(self.agent.test_reward_his)/20

    def best_by_tree(self):
        end_node = self.mcts.TREEPOLICYEND(self.mcts.root)
        pathes = self.mcts.MOVETOPATH(end_node.state)
        reward = self._get_reward_from_agent(pathes)
        print "###############################################"
        print pathes
        print "tree got the reward:", reward
        print "###############################################"
