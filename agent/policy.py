from __future__ import division
import numpy as np

from rl.util import *
from rl.policy import Policy


class EpsGreedyQPolicy(Policy):
    def __init__(self, eps=.1):
        super(EpsGreedyQPolicy, self).__init__()
        self.eps = eps

    def select_action(self, q_values):
        if self.eps > 0.1:
            self.eps *= 0.999995

        if q_values.ndim == 1:
            nb_actions = q_values.shape[0]

            if np.random.uniform() < self.eps:
                action = np.random.random_integers(0, nb_actions - 1)
            else:
                action = np.argmax(q_values)
            return action
        elif q_values.ndim == 2:
            nb_actions = q_values.shape[1]
            actions = []
            for q_value in q_values:
                if np.random.uniform() < self.eps:
                    action = np.random.random_integers(0, nb_actions - 1)
                else:
                    action = np.argmax(q_value)
                actions.append(action)
            return actions

    def get_config(self):
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config


class GreedyQPolicy(Policy):
    def select_action(self, q_values):
        if q_values.ndim == 1:
            action = np.argmax(q_values)
            return action
        elif q_values.ndim == 2:
            actions = []
            # print "------------------"
            # print q_values
            for q_value in q_values:
                action = np.argmax(q_value)
                actions.append(action)
            # print actions
            return actions
