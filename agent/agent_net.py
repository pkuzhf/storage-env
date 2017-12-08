import config, utils
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation, LeakyReLU, PReLU
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, TimeDistributed
from keras.initializers import *

def get_agent_net():

    h = config.Map.Height
    w = config.Map.Width
    agent_num = config.Game.AgentNum
    nb_action = config.Game.AgentAction

    qvalues_space_shape = (agent_num, nb_action)

    # ob_input = Input(shape=(agent_num, 4, w, h, 1), name='ob_input')
    #
    # x = TimeDistributed(TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), padding='same')))(ob_input)
    # x = Activation(activation='elu')(x)
    # x = TimeDistributed(TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same')))(x)
    # x = Activation(activation='elu')(x)
    # x = TimeDistributed(TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same')))(x)
    # x = Activation(activation='elu')(x)
    # x = TimeDistributed(Flatten())(x)
    # x = TimeDistributed(Dense(4 * w * h))(x)
    # x = Activation(activation='elu')(x)
    # x = TimeDistributed(Dense(nb_action))(x)
    # # x = TimeDistributed(Reshape((1,w*h*32)))(x)
    # # x = TimeDistributed(LSTM(nb_action))(x)
    # x = Reshape(qvalues_space_shape)(x)
    # model = Model(inputs=ob_input, output=x)
    # print(model.summary())
    ob_input = Input(shape=(agent_num, 4, w, h, 1), name='ob_input')

    x = TimeDistributed(Flatten())(ob_input)
    x = TimeDistributed(Dense(nb_action))(x)
    # x = TimeDistributed(Reshape((1,w*h*32)))(x)
    # x = TimeDistributed(LSTM(nb_action))(x)
    x = Reshape(qvalues_space_shape)(x)
    model = Model(inputs=ob_input, output=x)

    print('agent model:')
    print(model.summary())
    return model