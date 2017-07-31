import gym
import numpy as np
from agent_gym import AGENT_GYM
import config, utils
import matplotTest as draw
import mapGenerator as MG
# source_pos = [[0,0]]
# hole_pos= [[4,4],[0,4],[4,0]]
agent_num = 3
total_time = 200
# hole_city = [0,1,2]
# city_dis = [0.33, 0.33, 0.34]


# there are some problems in randomcity()
# so hole_city and city_dis are inited by hand
source_pos, hole_pos = MG.bigMap(10,10)
hole_city = [0,1,2,2,1,3]
color = draw.randomcolor(4)
color[4] = [1, 1, 1]
city_dis = [0.12,0.33,0.37,0.18]

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
        [agent_pos, agent_city, agent_reward, hole_reward, source_reward] = observation

        # TODO: print picture according to source_pos, hole_pos, agent_pos, agent_city, hole_city

        # so many params...
        draw.draw_map([10,10], source_pos, hole_pos, hole_city, agent_pos, agent_city,
                      color, "show", t, agent_reward, hole_reward, source_reward, city_dis)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

draw.save_video("show", total_time)
