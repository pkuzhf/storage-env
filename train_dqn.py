from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Conv2D
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Masking, concatenate, \
    Input, merge, Reshape, Lambda, TimeDistributed, LSTM, GRU, Bidirectional, SimpleRNN
from keras.optimizers import Adam
import keras.backend as K
from keras import initializers

from agent.dqn import DQNAgent
from agent.policy import EpsGreedyQPolicy, GreedyQPolicy, AstarPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from environment import AGENT_GYM
from environment import config
from environment import utils
from environment import mapGenerator as MG

from callback import myTestLogger

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
total_time = 1000
# source_pos, hole_pos = MG.bigMap(config.Map.Height, config.Map.Width)
# source_pos, hole_pos, hole_city, city_dis = MG.hugeMap(config.Map.Width, config.Map.Height)
city_dis = [0.26666666666666666, 0.13333333333333333, 0.2, 0.26666666666666666, 0.13333333333333333]
source_pos = [[2, 2], [17, 2], [2, 5], [17, 5], [2, 8], [17, 8], [2, 11], [17, 11], [2, 14], [17, 14], [2, 17],
              [17, 17]]
hole_pos = [[5, 3], [8, 3], [11, 3], [14, 3], [5, 6], [8, 6], [11, 6], [14, 6], [5, 9], [8, 9], [11, 9], [14, 9],
            [5, 12], [8, 12], [11, 12], [14, 12], [5, 15], [8, 15], [11, 15], [14, 15]]
hole_city = [0, 2, 2, 4, 2, 3, 1, 4, 4, 3, 1, 2, 3, 0, 0, 2, 3, 3, 0, 3]
agent_num = 10
window = 8
# nb_action = 5 + len(source_pos) + len(hole_pos)
nb_action = 4
w = config.Map.Width
h = config.Map.Height

# 1 is available. [right,up,left,down]. be careful to the mapping of the trans to map
# [1,1,1,0]
# TODO pathes here
# trans = [[[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]],
#          [[1,1,1,1], [1,1,1,1], [1,1,1,1], [0,0,0,1], [1,1,1,1], [1,1,1,1]],
#          [[1,1,1,1], [1,0,0,0], [1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]],
#          [[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1], [0,0,1,0], [1,1,1,1]],
#          [[1,1,1,1], [1,1,1,1], [0,1,0,0], [1,1,1,1], [1,1,1,1], [1,1,1,1]],
#          [[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]]]

# trans = None

# trans = [[[1,0,0,0], [1,0,0,0], [1,0,0,0], [0,0,0,1], [1,0,0,1], [0,0,0,1]],
#          [[1,1,0,0], [1,0,0,0], [1,0,0,1], [0,0,1,1], [0,0,0,1], [0,0,0,1]],
#          [[1,0,0,0], [1,0,0,1], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,0,1,0]],
#          [[0,1,0,0], [1,1,0,0], [1,0,0,0], [0,1,0,0], [0,1,1,0], [0,0,1,0]],
#          [[0,1,0,0], [0,1,0,0], [1,1,0,0], [0,1,1,0], [0,0,1,0], [0,0,1,1]],
#          [[0,1,0,0], [0,1,1,0], [0,1,0,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]]]
trans = np.ones((config.Map.Width,config.Map.Height,4), dtype=np.int8)
# TODO for highlight: comment the following lines to get full connected path
for i in range(0,config.Map.Width):
    for j in range(0, config.Map.Height):
        if i%2==0:
            trans[i][j][1] = 0
        else:
            trans[i][j][3] = 0
        if j%2==0:
            trans[i][j][2] = 0
        else:
            trans[i][j][0] = 0


for i in range(1):
    env = AGENT_GYM(source_pos, hole_pos, agent_num, total_time, hole_city, city_dis, window, trans=trans)
    # np.random.seed(123)
    env.seed(config.Game.Seed)
    # env2.seed(config.Game.Seed)
    qvalues_space_shape = (agent_num, nb_action)

    # print env.reset().shape
    print env.reset().shape

    # Next, we build our model. We use the same model that was described by Mnih et al. (2015).
    # observation_space_shape = env.observation_space.shape

    # print(qvalues_space_shape, observation_space_shape)

    ob_input = Input(shape=(agent_num,4,w,h,1), name='ob_input')

    x = TimeDistributed(Flatten())(ob_input)
    x = TimeDistributed(Dense(nb_action))(x)
    # x = TimeDistributed(Reshape((1,w*h*32)))(x)
    # x = TimeDistributed(LSTM(nb_action))(x)
    x = Reshape(qvalues_space_shape)(x)
    model = Model(inputs=ob_input, output=x)
    # print(model.summary())

    # # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # # even the metrics!
    memory = SequentialMemory(limit=400000, window_length=4)
    # processor = AtariProcessor()

    # Select a policy. We use eps-greedy action selection, which means that a random action is selected
    # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
    # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
    # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
    # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
    policy = EpsGreedyQPolicy(eps=0.5,end_eps=0.075,steps=90000)
    policy2 = GreedyQPolicy()
    astarPolicy = AstarPolicy()
    testlogger = [myTestLogger()]

    # # The trade-off between exploration and exploitation is difficult and an on-going research topic.
    # # If you want, you can experiment with the parameters or use a different policy. Another popular one
    # # is Boltzmann-style exploration:
    # # policy = BoltzmannQPolicy(tau=1.)
    # # Feel free to give it a try!
    # nb_actions=4

    dqn = DQNAgent(model=model, nb_actions=nb_action, policy=policy, memory=memory,
                   nb_steps_warmup=1000, gamma=.95, target_model_update=10000,
                   train_interval=4, delta_clip=1., test_policy=astarPolicy, agent_num=agent_num)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    # nb_step is reduced for test
    # dqn.fit(env, nb_steps=1000, log_interval=10000, verbose=2)

    # Finally, evaluate our algorithm for 10 episodes.
    # test episodes will be visualized
    # dqn.test(env2, nb_episodes=1, visualize=False, callbacks=testlogger, verbose=0)
    dqn.test(env, nb_episodes=1, visualize=False, callbacks=testlogger, verbose=0)
