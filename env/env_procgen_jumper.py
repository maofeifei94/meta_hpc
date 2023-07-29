import gym
import cv2
import numpy as np
import time


class Jumper():
    def __init__(self) -> None:
        self.env = gym.make('procgen:procgen-jumper-v0',start_level=0,num_levels=0,render=False)
    def reset(self):
        return self.env.reset()
    def step(self,action):
        return self.env.step(action)