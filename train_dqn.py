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

from agent.dqn import DQNAgent
from agent.policy import EpsGreedyQPolicy, GreedyQPolicy
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
# hole_city = [0,1,2,2,1,3]
# city_dis = [0.12,0.33,0.37,0.18]
source_pos = [[0,0]]
hole_pos = [[1,4],[4,1]]
city_dis = [0.5,0.5]
hole_city = [0,1]
agent_num = 4
window = 5
nb_action = 5 + len(source_pos) + len(hole_pos)
w = config.Map.Width
h = config.Map.Height

env = AGENT_GYM(source_pos, hole_pos, agent_num, total_time, hole_city, city_dis, window)
# np.random.seed(123)
env.seed(config.Game.Seed)
qvalues_space_shape = (agent_num, nb_action)

print env.reset().shape

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
# observation_space_shape = env.observation_space.shape

# print(qvalues_space_shape, observation_space_shape)


ob_input = Input(shape=(agent_num,w,h,8), name='ob_input')

# x = Reshape((agent_num,4*w*h))(ob_input)
# x = TimeDistributed(Dense(100))(x)
# x = Activation('tanh')(x)
# x = TimeDistributed(Dense(16))(ob_input)
x = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))(ob_input)
x = Activation(activation='elu')(x)
# x = TimeDistributed(Dense(8))(x)
x = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))(x)
x = Activation(activation='elu')(x)
# x = TimeDistributed(Dense(8))(x)
x = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))(x)
x = Activation(activation='elu')(x)
# x = TimeDistributed(Flatten())(x)
# x = TimeDistributed(Dense(nb_action))(x)
x = TimeDistributed(Reshape((1,w*h*32)))(x)
x = TimeDistributed(LSTM(nb_action))(x)
x = Reshape(qvalues_space_shape)(x)
model = Model(inputs=ob_input, output=x)
print(model.summary())


# # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# # even the metrics!
memory = SequentialMemory(limit=200000, window_length=4)
# processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = EpsGreedyQPolicy(eps=0.5,end_eps=0.075,steps=250000)
policy2 = GreedyQPolicy()
testlogger = [myTestLogger()]

# # The trade-off between exploration and exploitation is difficult and an on-going research topic.
# # If you want, you can experiment with the parameters or use a different policy. Another popular one
# # is Boltzmann-style exploration:
# # policy = BoltzmannQPolicy(tau=1.)
# # Feel free to give it a try!
# nb_actions=4
dqn = DQNAgent(model=model, nb_actions=nb_action, policy=policy, memory=memory,
               nb_steps_warmup=10000, gamma=.95, target_model_update=10000,
               train_interval=4, delta_clip=1., test_policy=policy2, agent_num=agent_num)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    # nb_step is reduced for test
    dqn.fit(env, nb_steps=300000, log_interval=10000, verbose=2)

    # Finally, evaluate our algorithm for 10 episodes.
    # test episodes will be visualized
    dqn.test(env, nb_episodes=10, visualize=False, callbacks=testlogger, verbose=0)
