import numpy as np
import cv2
import copy
import random

class GuessBox():
    def __init__(self) -> None:
        self.box_num=30
        self.target_num=1
        self.max_step_num=40
    def reset(self):
        obs=self.inner_reset()
        return obs
    def inner_reset(self):
        self.target_index=np.random.randint(0,self.box_num)
        obs=np.zeros([self.box_num],np.float32)
        self.step_num=0
        return obs
    
    def get_obs(self,action):
        if self.target_index==action:
            value=1
        else:
            value=-1
        obs=np.zeros([self.box_num],np.float32)
        obs[action]=value
        return obs
    def get_reward(self,action):
        return int(action==self.target_index)
    def get_inner_done(self,action):
        return self.step_num>=self.max_step_num
           
    def step(self,action):
        self.step_num+=1
        obs=self.get_obs(action)
        reward=self.get_reward(action)
        inner_done=self.get_inner_done(action)
        if inner_done:
            obs=self.inner_reset()
        
        return obs,reward,False,{}
