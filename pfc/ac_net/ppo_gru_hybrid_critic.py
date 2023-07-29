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

def nl_func():
    return nn.Tanh()
"ppo_gru_hybrid"
class Critic(parl.Model):
    def __init__(self, input_output_info):
        # print("gru_ppo init_param=",img_shape,obs_dim,int(img_shape[2]*img_shape[1]/8**2)+obs_dim)
        super().__init__()
        self.hp=input_output_info['hyperparam']
        self.gru_hid_dim=self.hp.PPO_GRU_HID_DIM
        self.obs_dim=input_output_info['obs_dim']


        "从obs到gru"
        self.obs_to_gru=nn.Sequential(
            nn.Linear(self.obs_dim,self.hp.PPO_CRITIC_ENCODER_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_CRITIC_ENCODER_FC_DIM,self.hp.PPO_CRITIC_ENCODER_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_CRITIC_ENCODER_FC_DIM,self.hp.PPO_CRITIC_ENCODER_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_CRITIC_ENCODER_FC_DIM,self.hp.PPO_CRITIC_ENCODER_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_CRITIC_ENCODER_FC_DIM,self.hp.PPO_CRITIC_ENCODER_FC_DIM),nl_func(),
            )

        self.gru=nn.GRU(input_size=self.hp.PPO_CRITIC_ENCODER_FC_DIM,hidden_size=self.gru_hid_dim)
        self.gru.flatten_parameters()
        
        "从gru到reward"
        self.gru_to_value=nn.Sequential(
            nn.Linear(self.gru_hid_dim,self.hp.PPO_CRITIC_HEAD_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_CRITIC_HEAD_FC_DIM,self.hp.PPO_CRITIC_HEAD_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_CRITIC_HEAD_FC_DIM,self.hp.PPO_CRITIC_HEAD_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_CRITIC_HEAD_FC_DIM,1)
            )

        self.reset()

    def reset(self):
        self.h=paddle.zeros([1,1,self.gru_hid_dim])

    def squeeze_time_layer(self,x,layer):
        "batch和time合并，通过全连接层或卷积层，再将batch和time分开"
        
        x_shape=x.shape
        # print("time layer xshape=",x.shape,[x_shape[0]*x_shape[1],*x_shape[2:]])
        x_no_time=paddle.reshape(x,[x_shape[0]*x_shape[1],*x_shape[2:]])
        x_after_layer=layer(x_no_time)
        x_after_layer_shape=x_after_layer.shape
        # print([x_shape[0],x_shape[1],*x_after_layer_shape[2:]])
        output=paddle.reshape(x_after_layer,[x_shape[0],x_shape[1],*x_after_layer_shape[1:]])
        # print("time layer outshape=",output.shape)
        return output

    def forward(self, input_dict,train,h_dict):
        """
        LSTM输入
        x=paddle.randn([batch_size,time_steps,input_size])
        pre_h=paddle.randn([num_layers*direction_num,batch_size,hid_dim])
        pre_c=paddle.randn([num_layers*direction_num,batch_size,hid_dim])
        输出
        y,(h,c)=LSTM(x,(pre_h,pre_c))

        """
        init_h=h_dict['critic_init_h']
        env_vec=input_dict['env_vec']

        batch_size,time_len=env_vec.shape[0],env_vec.shape[1]

       

        feature_to_gru=self.squeeze_time_layer(env_vec,self.obs_to_gru)

        "gru"
        if train:
            if init_h is None:
                raise("actor train mode must input init_h")
            pass
        else:
            init_h=self.h

        gru_out,h=self.gru(feature_to_gru,init_h)
        value_pred=self.squeeze_time_layer(gru_out,self.gru_to_value)
        value_dict={'value':value_pred}


        if train:
            pass
        else:
            self.h=h.detach()


        return value_dict
class Critic_Static(Critic):
    @to_static
    def forward(self,input_dict,train,h_dict):
        return super(Critic_Static,self).forward(input_dict,train,h_dict)