import config, utils
from keras import backend as K
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, LeakyReLU, PReLU, merge, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Masking, LSTM
from keras.layers.merge import Concatenate
from keras.activations import relu, softmax
from keras.initializers import *

def get_env_actor_net():

    n = config.Map.Height
    m = config.Map.Width
    d = 4

    observation = Input(shape=(1, m, n, d), name='observation_input_actor')
    x = Reshape((m*n, d))(observation)
    x = Masking(mask_value=-1)(x)

    x = LSTM(4, dropout=0.2, recurrent_dropout=0.2)(x)

    x = Dense(14)(x)
    x = Activation(activation='elu')(x)
    x = Dense(14)(x)
    x = Activation(activation='elu')(x)
    actions = Activation(activation='softmax')(x)

    env_model_actor = Model(inputs=observation, outputs=actions, name='env_actor')

    print('env model:')
    print(env_model_actor.summary())
    return env_model_actor


def get_env_critic_net():

    n = config.Map.Height
    m = config.Map.Width
    d = 4

    action = Input(shape=(14,), name='action_input')
    observation = Input(shape=(1, m, n, d), name='observation_input_critic')
    flattened_observation = Flatten()(observation)
    x = Concatenate(axis=1)([action, flattened_observation])
    x = Masking(mask_value=-1)(x)
    x = Dense(64)(x)
    x = Activation(activation='elu')(x)
    x = Dense(32)(x)
    x = Activation(activation='elu')(x)
    x = Dense(32)(x)
    x = Activation(activation='elu')(x)
    x = Dense(1)(x)

    env_model_critic = Model(inputs=[action, observation], outputs=x, name='env_critic')

    print('env model:')
    print(env_model_critic.summary())
    return env_model_critic, action

