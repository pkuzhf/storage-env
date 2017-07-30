import gym
import numpy as np

env = gym.make('Maze-v0')
for i_episode in range(1):
    observation = env.reset()
    for t in range(10):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation)
        print(np.shape(observation))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break