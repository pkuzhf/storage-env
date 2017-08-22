from __future__ import absolute_import
import gym
from gym import spaces, utils
from gym.utils import seeding
import numpy as np
import sys
from six import StringIO
# from ..space import ShapedTuple
class ShapedTuple(spaces.Tuple):
    """
    A tuple (i.e., product) of simpler spaces
    Example usage:
    observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
    """
    def __init__(self, spaces):
        self.spaces = spaces
        self.shape = (len(spaces),) + spaces[0].shape
        super(ShapedTuple, self).__init__(spaces)

class GuessingGame(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    def __init__(self, agent_num=4, num_range=10, max_steps=10):
        self.agent_num = agent_num
        self.num_range = num_range
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
        self.target_numbers = self._generate_target_numbers()
        self.guess_count = 0
        self.guess_max = max_steps
        self._seed()
        self._reset()

    def _make_observation(self, differences):
        return np.array([(a, b) for a, b in zip(*(self.target_numbers, differences))])

    def _observation_space(self):
        obs_low = [-self.num_range, -self.num_range]
        obs_high = [self.num_range, self.num_range]
        obses = tuple([spaces.Box(np.array(obs_low), np.array(obs_high))
                           for _ in range(self.agent_num)])
        return ShapedTuple(obses)

    def _action_space(self):
        return spaces.Box(-self.num_range, self.num_range, self.agent_num)

    def _generate_target_numbers(self):
        return (np.random.rand(self.agent_num) - 0.5) * 2. * self.num_range

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, actions):
        # print(actions.shape)
        actions = actions.reshape(self.action_space.shape)
        assert self.action_space.contains(actions)
        self.last_actions = actions
        differences = actions - sum(self.target_numbers)

        rewards = -np.abs(differences)
        self.last_rewards = rewards
        done = False

        self.guess_count += 1
        if self.guess_count >= self.guess_max:
            done = True

        return self.observation, rewards, done, {"number": self.target_numbers, "guesses": self.guess_count}

    def _reset(self):
        self.target_numbers = self._generate_target_numbers()
        self.guess_count = 0
        self.observation = self._make_observation(np.zeros(self.agent_num))
        self.last_actions = np.zeros(self.agent_num)
        self.last_rewards = np.zeros(self.agent_num)
        return self.observation

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        content = 'Target numbers: \t{}\nactions:\t\t\t{}\nrewards:\t\t\t{}\n\n'.format(np.array_str(
            self.target_numbers), np.array_str(self.last_actions), np.array_str(self.last_rewards))
        outfile.write(content)
        if mode != 'human':
            return outfile

if __name__ == "__main__":
    print('Test')
    agent_num = 4
    env = GuessingGame(agent_num=agent_num)
    states = env.reset()
    for _ in range(1000):
        env.render()
        actions = env.action_space.sample()
        states, rewards, done, info = env.step(actions)
