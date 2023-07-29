import gym
import numpy as np
import gym_hybrid
from mlutils.env import EnvNeverDone

env = gym.make('Moving-v0')
env=EnvNeverDone(env)
obs = env.reset()
while 1:
    obs,reward,done,info=env.step([np.random.randint(0,3),[np.random.uniform(0,1),np.random.uniform(-1,1)]])
    # env.step(0)
    env.render()
    print(obs,reward,done)
    if done:
        input()
# print(obs)