import gym
import numpy as np
from agent_gym import AGENT_GYM
import config, utils
import matplotTest as draw

source_pos = [[0,0]]
hole_pos= [[4,4],[0,4],[4,0]]
agent_num = 3
total_time = 200
hole_city = [0,1,2]
city_dis = [0.33, 0.33, 0.34]

env = AGENT_GYM(source_pos, hole_pos, agent_num, total_time, hole_city, city_dis)
env.seed(config.Game.Seed)

color = draw.randomcolor(3)
color[3] = [1, 1, 1]

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
        [agent_pos, agent_city, agent_reward, hole_reward, source_reward] = observation

        # TODO: print picture according to source_pos, hole_pos, agent_pos, agent_city, hole_city

        draw.draw_map([config.Map.Width,config.Map.Height], source_pos, hole_pos, hole_city, agent_pos, agent_city, color, "show",t)

        print "///"
        print source_pos
        print hole_pos
        print hole_city
        print agent_pos
        print agent_city
        print "///"

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break