import gym
import numpy as np
from agent_gym import AGENT_GYM
import config, utils

source_pos = [[0,0]]
hole_pos= [[4,4],[0,4],[4,0]]
agent_num = 3
total_time = 200
hole_city = [0,1,2]
city_dis = [0.33, 0.33, 0.34]

env = AGENT_GYM(source_pos, hole_pos, agent_num, total_time, hole_city, city_dis)
env.seed(config.Game.Seed)

for i_episode in range(1):
    observation = env.reset()
    print observation
    for t in range(total_time):
        env.render()
        action = []
        for i in range(agent_num):
            action.append(utils.dirs[np.random.random_integers(0, 3)])
        print action
        observation, reward, done, info = env.step(action)
        print(observation)
        #print(np.shape(observation))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break