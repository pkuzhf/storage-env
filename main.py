import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "" #0, 1, 2, 3, 4, 5, 6, 7"

import profile
from utils import *
from env.env_net import *
from env.env_gym import ENV_GYM
from env.env_gym_hole import ENV_GYM as ENV_GYM_HOLE
from env.env_pg_gym import ENV_GYM as ENV_PG_GYM
from env.mydqn import myDQNAgent as EnvDQN
from env.homemadepg import pgAgent as EnvPG
from env.pg_hole import pgAgent as Env_PG_HOLE
from agent.agent_net import *
from agent.agent_gym import AGENT_GYM
from agent.agent_gym_hole import AGENT_GYM as AGENT_GYM_HOLE
from agent.agent_dqn import DQNAgent as AgentDQN
from keras.optimizers import *
from policy import *
from rl.memory import SequentialMemory

import keras.backend.tensorflow_backend as KTF
KTF.set_session(get_session())

# TODO net + observation, new env, 2x4
def main():
    np.random.seed(config.Game.Seed)
    config.Type_num = len(config.Map.city_dis) + 1
    config.Source_num = len(config.Map.source_pos)
    config.Hole_num = len(config.Map.hole_pos)
    # dqn env part
    # ---------------------------------------------------------------------------
    env_gym = ENV_GYM_HOLE()
    env_gym.seed(config.Game.Seed)
    #
    # env_memory = SequentialMemory(limit=100000, window_length=1)
    # EpsGreedyPolicy = EpsGreedyQPolicy(0.3,0.15,10000)
    # env = EnvDQN(name='env', model=get_env_hole_net(), batch_size=config.Training.BatchSize, delta_clip=10, gamma=1.0,
    #            nb_steps_warmup=200, target_model_update=100,
    #            enable_dueling_network=True, policy=EpsGreedyPolicy, test_policy=GreedyQPolicy(),
    #            nb_actions=len(config.Map.hole_pos), memory=env_memory)
    # env.compile(Adam(lr=.0005), metrics=['mae'])

    # pg env part
    # ---------------------------------------------------------------------------
    # env_gym = ENV_PG_GYM()
    # env_gym.seed(config.Game.Seed)
    # env = EnvPG(env_gym, nb_action=config.Map.Width*config.Map.Height*10, nb_warm_up=2000, policy=MultiDisPolicy(),
    #             testPolicy=MultiDisPolicy(), gamma=0.95, lr=0.00005, memory_limit=10000, batchsize=32, train_interval=8)
    env = Env_PG_HOLE(env_gym, nb_action=len(config.Map.hole_pos), nb_warm_up=len(config.Map.hole_pos), policy=MultiDisPolicy(),
                      testPolicy=MultiDisPolicy(), gamma=1.0, lr=0.001, memory_limit=10000, batchsize=32,
                      train_interval=len(config.Map.hole_pos))
    # agent part
    # ---------------------------------------------------------------------------
    # agent_gym = AGENT_GYM_HOLE(config.Map.source_pos, config.Map.hole_pos,
    #                       config.Game.AgentNum, config.Game.total_time, config.Map.hole_city,
    #                       config.Map.city_dis)
    # agent_gym.seed(config.Game.Seed)

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

    # agent_gym.agent = agent

    print(vars(config.Map))
    print(vars(config.Training))

    #run(agent, env, agent_gym, env_gym, task_name)
    run_env_path(env, env_gym)

    print(vars(config.Map))
    print(vars(config.Training))


def run_env_path(env, env_gym):

    nround = 1
    model_folder = config.Path.Models

    makedirs(model_folder)
    makedirs(config.Path.Logs)
    makedirs(config.Path.Figs)

    for round in range(nround):
        print "------------------------------------------------------"
        print('\n\n train ' + str(round) + '/' + str(nround))
        print "------------------------------------------------------"
        # env.fit(env_gym, nb_steps=1000, visualize=False, verbose=2)
        # env.fit(env_gym, nb_episodes=36, min_steps=80, visualize=False, verbose=2) # for dqn
        env.fit(40000)
        # env.nb_steps_warmup = 0
        # env.test(env_gym, nb_episodes=1, visualize=False, verbose=2) # for dqn
        env.test()
        # env_gym.best_by_tree()
        # env.save_weights(model_folder + '/generator_model_weights_{}.h5f'.format(str(round)), overwrite=True)


if __name__ == "__main__":
    #main()
    profile.run("main()", sort=1)
