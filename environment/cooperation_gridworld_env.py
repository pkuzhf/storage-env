import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from collections import defaultdict

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from gym.utils import seeding
# from matplotlib.patches import Rectangle

class ShapedTuple(spaces.Tuple):
    """
    A tuple (i.e., product) of simpler spaces
    Example usage:
    observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
    """

    def __init__(self, spaces):
        self.spaces = spaces
        if type(spaces[0]) is gym.spaces.Discrete:
            self.shape = (len(spaces),1)
        else:
            self.shape = (len(spaces),) + spaces[0].shape
        super(ShapedTuple, self).__init__(spaces)


class CooperationGridWorld:
    def __init__(self, map_size=10, agent_num=4, max_steps=1000):
        self.map_size = map_size
        self.agent_num = agent_num
        self.init_position = (map_size / 2) * map_size + (map_size / 2)
        self.agent_postitions = [self.init_position] * self.agent_num
        self.map_status = [0] * (map_size**2)
        self.map_status[self.init_position] = agent_num
        self.max_steps = max_steps
        self.steps = 0
        self.action_space = ShapedTuple(tuple([spaces.Discrete(4) for _ in range(self.agent_num)]))
        self.observation_space = ShapedTuple(tuple([spaces.Discrete(map_size**2) for _ in range(self.agent_num)]))

    def reset(self):
        self.steps = 0
        self.agent_postitions = [self.init_position] * self.agent_num
        return np.array([[p] for p in self.agent_postitions])

    def get_reward(self, position):
        if position in [0, self.map_size - 1, (self.map_size - 1) * self.map_size, self.map_size**2 - 1]:
            return 1. / self.map_status[position]
        return 0.

    def get_rewards(self):
        return np.array([self.get_reward(position) for position in self.agent_postitions])

    def get_next_position(self, position, action):
        x, y = self.position2coord(position)
        coords = np.array([x, y])
        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        next_coords = np.clip(
            coords + increments[action],
            [0, 0],
            [self.map_size - 1, self.map_size - 1]
        )
        return next_coords[0] * self.map_size + next_coords[1]

    def is_done(self):
        if self.steps >= self.max_steps:
            return True
        count = 0
        for position in [0, self.map_size - 1, (self.map_size - 1) * self.map_size, self.map_size**2 - 1]:
            count += self.map_status[position]
        if count == self.agent_num:
            return True
        return False

    def position2coord(self, position):
        return position // self.map_size, position % self.map_size

    def step(self, actions):
        states = []
        for i, action in enumerate(actions):
            old_position = self.agent_postitions[i]
            new_position = self.get_next_position(old_position, action)
            self.agent_postitions[i] = new_position
            self.map_status[old_position] -= 1
            self.map_status[new_position] += 1
            states.append([new_position])
        self.steps += 1
        rewards = self.get_rewards()
        return np.array(states), rewards, self.is_done(), {}

    def render(self, plt_delay=.1):
        plt.matshow(np.zeros((self.map_size, self.map_size)),
                    cmap=plt.get_cmap('Greys'), fignum=1)
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        for position in self.agent_postitions:
            x, y = self.position2coord(position)
            plt.plot(x, y, "b.", markersize=10, alpha=0.6)
        for position in [0, self.map_size - 1, (self.map_size - 1) * self.map_size, self.map_size**2 - 1]:
            x, y = self.position2coord(position)
            plt.plot(x, y, "r+", markersize=3, alpha=0.9)
        plt.pause(plt_delay)
        plt.clf()


if __name__ == "__main__":
    print('Test')
    map_size=100
    agent_num=40
    env = CooperationGridWorld(map_size=map_size, agent_num=agent_num)
    states = env.reset()
    for _ in range(1000):
      env.render()
      actions = np.random.randint(4, size=agent_num)
      states, rewards, done, info = env.step(actions)
