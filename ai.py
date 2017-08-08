import gym
import numpy as np
from agent_gym import AGENT_GYM
import config, utils
import matplotTest as draw
import mapGenerator as MG
import copy
from whca import WHCA

# source_pos = [[0,0]]
# hole_pos= [[4,4],[0,4],[4,0]]
# agent_num = 20
total_time = 1000
# hole_city = [0,1,2]
# city_dis = [0.33, 0.33, 0.34]


# there are some problems in randomcity()
# so hole_city and city_dis are inited by hand
source_pos, hole_pos = MG.bigMap(config.Map.Height, config.Map.Width)
hole_city = [0,1,2,2,1,3]
color = draw.randomcolor(4)
color[4] = [0.9, 0.9, 0.9]
city_dis = [0.12,0.33,0.37,0.18]

window = 10


for agent_num in range(1, 50):

    env = AGENT_GYM(source_pos, hole_pos, agent_num, total_time, hole_city, city_dis)
    env.seed(config.Game.Seed)

    #print 'agent_num: ' + str(agent_num)
    observation = env.reset()
    #print observation
    [agent_pos, agent_city, agent_reward, hole_reward, source_reward] = observation


    whca = WHCA(window, source_pos, hole_pos, agent_num)

    agent_city_old = copy.deepcopy(agent_city)
    agent_pos_old = copy.deepcopy(agent_pos)

    for time in range(total_time):
        #print ['*** time ***', time]
        env.render()

        # random action
        # action = []
        # for i in range(agent_num):
        #     r = np.random.random_integers(0, 4)
        #     if r < 4:
        #         action.append(utils.dirs[r])
        #     else:
        #         action.append([0,0])

        # whca

        action = whca.getJointAction(agent_pos, time)

        #print ['action', action]
        observation, reward, done, info = env.step(action)
        [agent_pos, agent_city, agent_reward, hole_reward, source_reward] = observation
        #print ['agent_pos', agent_pos]

        '''
        if True:
            # make pics
            for i in range(len(agent_pos)):
                agent_pos_old[i][0] = (agent_pos[i][0] + agent_pos_old[i][0]) / 2.0
                agent_pos_old[i][1] = (agent_pos[i][1] + agent_pos_old[i][1]) / 2.0

                # so many params...
            if not time == 0:
                draw.draw_map([config.Map.Width, config.Map.Height], source_pos, hole_pos, hole_city, agent_pos_old,
                              agent_city_old,color,"mapOutput", "show", 2 * time - 1, agent_reward, hole_reward, source_reward, city_dis)

            draw.draw_map([config.Map.Width, config.Map.Height], source_pos, hole_pos, hole_city, agent_pos, agent_city,
                          color, "mapOutput", "show", 2 * time, agent_reward, hole_reward, source_reward, city_dis)

            for i in range(len(agent_pos)):
                agent_city_old[i] = agent_city[i]
                agent_pos_old[i][0] = agent_pos[i][0]
                agent_pos_old[i][1] = agent_pos[i][1]
        '''

        if done:
            #print("Episode finished after {} timesteps".format(time+1))
            break
    print str(agent_num) + '\t' + str(sum(agent_reward))
# draw.save_video("mapOutput", "show", total_time) # avi with cv
# draw.save_video2("mapOutput", "show", total_time) # gif with imageio
# draw.save_video3("mapOutput", "show", total_time) # mp4 with imageio
