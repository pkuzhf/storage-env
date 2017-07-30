import gym
import numpy as np
from agent_gym import AGENT_GYM


source_pos = [[0,0]]
hole_pos[[4,4],[0,4],[4,0]]
agent_num = 3
total_time = 100
hole_city = [0,1,2]

env = AGENT_GYM(source_pos, hole_pos, agent_num, total_time, hole_city)
env.seed(config.Game.Seed)

for i_episode in range(1):
    observation = env.reset()
    for t in range(10):
        env.render()
        action = []
        for i in range(agent_num):
            action.append(utils.dirs[np.random.random_integers(0, 3)])
        observation, reward, done, info = env.step(action)
        print(observation)
        #print(np.shape(observation))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break