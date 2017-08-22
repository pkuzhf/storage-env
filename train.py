from keras.optimizers import Adam, Nadam, RMSprop
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import ModelIntervalCheckpoint
# from callback import FileLogger

from environment import GuessingGame
from agent import MultiActorCriticAgent
from datetime import datetime
from network import get_model

import tensorflow as tf
from keras import backend as K

import argparse
import numpy as np
import gym


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', help='model name, including: dcn, mlp, \
         commnet, greedymdp, independent',
        default="dcn")
    parser.add_argument(
        '--ip', help='server ip, junwang1: 128.16.7.171', default="localhost")
    parser.add_argument('--port', help='server port', default="11111")
    parser.add_argument('--map_name', help='map name')
    parser.add_argument('--agent_number', help='agent number',
                        default="4", type=int)
    parser.add_argument('--enemy_agent_number',
                        help='enemy agent number', default="16", type=int)
    parser.add_argument('--max_episode_steps',
                        help='number of max episode steps',
                        default="500", type=int)
    parser.add_argument('--set_maximum_agent',
                        help='maxximum_agent',
                        default="False", type=bool)
    parser.add_argument('--gpu_memory_fraction',
                        help='maxximum_agent',
                        default="0.", type=float)
    parser.add_argument('--frame_skip',
                        help='frame skip',
                        default="9", type=int)
    parser.add_argument('--group_size',
                        help='group size',
                        default="0", type=int)
    args = parser.parse_args()

    if args.gpu_memory_fraction > 0.:
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        K.set_session(sess)
    else:
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        sess = tf.Session(config=config)
        K.set_session(sess)

    AGENT_NUMBER = args.agent_number
    ENEMY_AGENT_NUMBER = args.enemy_agent_number
    MAP_NAME = args.map_name
    nb_max_episode_steps = args.max_episode_steps

    if MAP_NAME is not None:
        AGENT_NUMBER, ENEMY_AGENT_NUMBER = extract_agent_numbers(MAP_NAME)
    ENV_NAME = 'multi_agent_starcraft'
    MODEL_NAME = '{}_{}'.format(
        args.model, datetime.utcnow().strftime('%Y%m%d%H%M%S'))

    if args.group_size > 0:
        import math
        MODEL_NAME = '{}_{}'.format(args.group_size, MODEL_NAME)
        AGENT_NUMBER = int(math.ceil(1. * AGENT_NUMBER / args.group_size)) * args.group_size


    file_prefix = '{}_{}_{}'.format(ENV_NAME, MAP_NAME, MODEL_NAME)

    gym.undo_logger_setup()
    # Get the environment and extract the number of actions.
    agent_num = 4
    env = GuessingGame(agent_num=agent_num)
    env.seed(123)
    np.random.seed(123)
    actor, critic, action_input = get_model(
        env.observation_space.shape,
        env.action_space.shape,
        AGENT_NUMBER,
        model=args.model
    )

    # Finally, we configure and compile our agent.
    # You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = None
    if args.model in {'dcn', 'dcn_1', 'commnet'}:
        random_process = OrnsteinUhlenbeckProcess(
            size=env.action_space.shape, theta=.25, mu=0., sigma=.1)


    weights_filename = './model_weights/{}_weights.h5f'.format(file_prefix)
    checkpoint_weights_filename = './model_weights/' + \
        file_prefix + '_weights_{step}.h5f'
    log_filename = './logs/{}_log.json'.format(file_prefix)
    callbacks = [ModelIntervalCheckpoint(
        checkpoint_weights_filename, interval=10000)]
    # callbacks += [FileLogger(log_filename, interval=1)]

    agent = MultiActorCriticAgent(
        agent_number=AGENT_NUMBER,
        actions_shape=env.action_space.shape,
        observation_shape=env.observation_space.shape,
        actor=actor,
        critic=critic,
        critic_action_input=action_input,
        memory=memory,
        nb_steps_warmup_critic=100,
        nb_steps_warmup_actor=100,
        random_process=random_process,
        gamma=.99,
        target_model_update=1e-3,
        batch_size=32
    )
    agent.compile(Nadam(), metrics=['mae'])

    # Okay, now it's time to learn something!
    # We visualize the training here for show, but this
    # slows down training quite a lot.
    # You can always safely abort the training prematurely using
    # Ctrl + C.
    history = agent.fit(
        env,
        nb_steps=50000,
        visualize=False,
        verbose=2,
        callbacks=callbacks,
        nb_max_episode_steps=nb_max_episode_steps
    )

    # After training is done, we save the final weights.
    agent.save_weights(weights_filename, overwrite=True)

    # rest the episodes number
    env.episodes = 0
    env.episode_wins = 0
    # Finally, evaluate our algorithm for 100 episodes.
    agent.test(env, nb_episodes=101, visualize=False,
               nb_max_episode_steps=nb_max_episode_steps, verbose=0)