import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import os
import copy
import cv2
import gym
import numpy as np
import sys
import parl
import threading
# from attr import validate
# from scipy.fft import hfft2
# import hyperparam as hp
# from hyperparam import Hybrid_PPO_HyperParam as hp
import time
from paddle.jit import to_static
import parl
import paddle
import paddle.optimizer as optim
import paddle.nn as nn
from paddle.distribution import Categorical,Normal
import numpy as np
# from myf_ML_util import timer_tool
from mlutils.debug import timer_tool
import logging

"ppo_gru_hybrid"
class Model(parl.Model):
    def __init__(self, input_output_info,net_class_dict):
        super().__init__()

        actor_class=net_class_dict['actor_class']
        critic_class=net_class_dict['critic_class']

        self.actor_model = actor_class(input_output_info)
        self.critic_model =critic_class(input_output_info)
            

        if input_output_info['to_static']:
            self.eval()
        else:
            self.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        
    def train(self):
        self.actor_model.train()
        self.critic_model.train()       

    def reshape_obs(self,obs):
        obs_shape=obs.shape
        if len(obs_shape)==1:
            return obs.reshape([1,1,-1])
        elif len(obs_shape)==2:
            return obs.reshape([1,*obs_shape])
        else:
            return obs
    def reshape_img(self,img):
        img_shape=img.shape
        if len(img_shape)==3:
            return img.reshape([1,1,*img_shape])
        elif len(img_shape)==4:
            return img.reshape([1,*img_shape])
        else:
            return img
        pass
    def policy(self, input_dict,train,h_dict):
        tt=timer_tool("model policy")
        action_dict=self.actor_model(input_dict,train,h_dict)
        tt.end("all")
        return action_dict
    def value(self, input_dict,train,h_dict):
        tt=timer_tool("model value")
        value_dict=self.critic_model(input_dict,train,h_dict)
        tt.end("all")
        return value_dict
    def get_actor_params(self):
        return self.actor_model.parameters()
    def get_critic_params(self):
        return self.critic_model.parameters()