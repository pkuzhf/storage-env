import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "" #0, 1, 2, 3, 4, 5, 6, 7"

import profile
from utils import *
from env.env_net import *
from env.env_gym import ENV_GYM
from env.mydqn import myDQNAgent as EnvDQN
from agent.agent_net import *
from agent.agent_gym import AGENT_GYM
from agent.agent_dqn import DQNAgent as AgentDQN
from keras.optimizers import *
from policy import *
from rl.memory import SequentialMemory

from env.myddpg import DDPGAgent

import keras.backend.tensorflow_backend as KTF
KTF.set_session(get_session())

# TODO net + observation, new env, 2x4
def main():
    np.random.seed(config.Game.Seed)

    # env part
    # ---------------------------------------------------------------------------
    env_gym = ENV_GYM()
    env_gym.seed(config.Game.Seed)

    # actor = get_env_actor_net()
    # critic, action_input = get_env_critic_net()
    env_memory = SequentialMemory(limit=100000, window_length=1)
    EpsGreedyPolicy = EpsGreedyQPolicy(0.3,0.15,1000)

    # env = DDPGAgent(nb_actions=14, actor=actor, critic=critic, critic_action_input=action_input,
    #                 memory=env_memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
    #                 gamma=.99, target_model_update=1e-3, policy=MultiDisPolicy())
    env = EnvDQN(name='env', model=get_env_net(), batch_size=config.Training.BatchSize, delta_clip=10, gamma=1.0,
               nb_steps_warmup=200, target_model_update=100,
               enable_dueling_network=True, policy=EpsGreedyPolicy, test_policy=GreedyQPolicy(),
               nb_actions=14, memory=env_memory)
    env.compile(Adam(lr=.0005), metrics=['mae'])

    # agent part
    # ---------------------------------------------------------------------------
    agent_gym = AGENT_GYM(config.Map.source_pos, config.Map.hole_pos,
                          config.Game.AgentNum, config.Game.total_time, config.Map.hole_city,
                          config.Map.city_dis)
    agent_gym.seed(config.Game.Seed)

    memory = SequentialMemory(limit=400000, window_length=4)
    policy = EpsGreedyQPolicy(eps=0.1, end_eps=0.1, steps=1000)
    policy2 = GreedyQPolicy2D()
    agent_model = get_agent_net()
    agent = AgentDQN(model=agent_model, nb_actions=config.Game.AgentAction, policy=policy, memory=memory,
               nb_steps_warmup=1000, gamma=.95, target_model_update=5000,
               train_interval=4, delta_clip=1., test_policy=policy2, agent_num=config.Game.AgentNum)
    agent.compile(Adam(lr=.00025), metrics=['mae'])

    # GAN part
    # ---------------------------------------------------------------------------
    env_gym.env = env
    env_gym.agent = agent

    agent_gym.agent = agent

    print(vars(config.Map))
    print(vars(config.Training))

    #run(agent, env, agent_gym, env_gym, task_name)
    run_env_path(env, env_gym)

    print(vars(config.Map))
    print(vars(config.Training))


def run_env_path(env, env_gym):

    nround = 100
    model_folder = config.Path.Models

    makedirs(model_folder)
    makedirs(config.Path.Logs)
    makedirs(config.Path.Figs)

    for round in range(nround):
        print "------------------------------------------------------"
        print('\n\nround train ' + str(round) + '/' + str(nround))
        print "------------------------------------------------------"
        # env.fit(env_gym, nb_steps=1000, visualize=False, verbose=2)
        env.fit(env_gym, nb_episodes=8, min_steps=80, visualize=False, verbose=2)
        # env.nb_steps_warmup = 0
        env.test(env_gym, nb_episodes=1, visualize=False, verbose=2)
        env_gym.best_by_tree()
        env.save_weights(model_folder + '/generator_model_weights_{}.h5f'.format(str(round)), overwrite=True)


if __name__ == "__main__":
    #main()
    profile.run("main()", sort=1)
