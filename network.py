from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, \
    Input, merge, Reshape, Lambda, TimeDistributed, LSTM, GRU, Bidirectional, SimpleRNN
from keras import backend as K
from functools import reduce
from agent.layer import EpisodicNoiseDense
import tensorflow as tf


def get_nb_actions(action_space_shape):
    return reduce(lambda a, b: a * b, action_space_shape)


def make_commnet(observation_space_shape, action_space_shape, agent_number):
    def cal_c(hs, agent_number=agent_number):
        return K.repeat(K.mean(hs, axis=1, keepdims=False), agent_number)
    actor_observation_input = Input(
        shape=(1,) + observation_space_shape, name='actor_observation_input')
    x = Reshape(observation_space_shape)(actor_observation_input)
    c = Lambda(cal_c)(x)
    c = TimeDistributed(Dense(observation_space_shape[1]))(c)
    x = TimeDistributed(Dense(observation_space_shape[1]))(x)
    x = merge([c, x], mode='sum')
    x = Activation('tanh')(x)
    c = Lambda(cal_c)(x)
    c = TimeDistributed(Dense(action_space_shape[1]))(c)
    x = TimeDistributed(Dense(action_space_shape[1]))(x)
    x = merge([c, x], mode='sum')
    x = Activation('tanh')(x)
    x = Reshape(action_space_shape)(x)
    actor = Model(input=actor_observation_input, output=x)
    print(actor.summary())

    action_input = Input(shape=action_space_shape, name='action_input')
    flattened_action = Flatten()(action_input)
    observation_input = Input(
        shape=(1,) + observation_space_shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = merge([flattened_action, flattened_observation], mode='concat')
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(agent_number)(x)
    x = Reshape((agent_number,))(x)
    critic = Model(input=[action_input, observation_input], output=x)
    print(critic.summary())
    return actor, critic, action_input


def make_independent(observation_space_shape,
                     action_space_shape, agent_number):
    actor_observation_input = Input(
        shape=(1,) + observation_space_shape, name='actor_observation_input')
    x = Reshape(observation_space_shape)(actor_observation_input)
    x = TimeDistributed(Dense(observation_space_shape[1]))(x)
    x = Activation('tanh')(x)
    x = TimeDistributed(EpisodicNoiseDense(action_space_shape[1]))(x)
    x = Reshape(action_space_shape)(x)
    actor = Model(input=actor_observation_input, output=x)
    print(actor.summary())

    action_input = Input(shape=action_space_shape, name='action_input')
    observation_input = Input(
        shape=(1,) + observation_space_shape, name='observation_input')
    shaped_observation = Reshape(observation_space_shape)(observation_input)
    x = merge([action_input, shaped_observation], mode='concat')
    x = TimeDistributed(Dense(observation_space_shape[1]))(x)
    x = Activation('tanh')(x)
    x = TimeDistributed(Dense(1))(x)
    x = Reshape((agent_number,))(x)
    x = Activation('linear')(x)
    critic = Model(input=[action_input, observation_input], output=x)
    print(critic.summary())
    return actor, critic, action_input


def make_dcn(observation_space_shape, action_space_shape, agent_number):
    scale = int(agent_number / 10 + 1)
    actor_observation_input = Input(
        shape=(1,) + observation_space_shape, name='actor_observation_input')
    x = Reshape(observation_space_shape)(actor_observation_input)
    x = Bidirectional(
        LSTM(1, return_sequences=True), merge_mode='sum')(x)
    # x = TimeDistributed(EpisodicNoiseDense(action_space_shape[1]))(x)
    x = Reshape(action_space_shape)(x)
    actor = Model(input=actor_observation_input, output=x)
    print(actor.summary())

    action_input = Input(shape=action_space_shape, name='action_input')
    shaped_action = Reshape(action_space_shape+(1,))(action_input)
    observation_input = Input(
        shape=(1,) + observation_space_shape, name='observation_input')
    shaped_observation = Reshape(observation_space_shape)(observation_input)
    x = merge([shaped_action, shaped_observation], mode='concat')
    x = Bidirectional(LSTM(32 * scale, return_sequences=True),
                      merge_mode='sum')(x)
    x = TimeDistributed(Dense(1))(x)
    x = Reshape((agent_number,))(x)
    critic = Model(input=[action_input, observation_input], output=x)
    print(critic.summary())
    return actor, critic, action_input

def get_model(observation_space_shape,
              action_space_shape, agent_number, model='dcn'):
    if model == 'dcn':
        return make_dcn(observation_space_shape,
                        action_space_shape, agent_number)
    elif model == 'commnet':
        return make_commnet(observation_space_shape,
                            action_space_shape, agent_number)
    elif model == 'independent':
        return make_independent(observation_space_shape,
                                action_space_shape, agent_number)
    else:
        raise Exception('model is unavailable')