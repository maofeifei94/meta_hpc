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
"""
需要的配置信息
hyperparam
actor_gru_hid_dim
obs_dim
action_env_discrete_dim
action_env_continuous_dim
action_env_discrete_dim
action_env_continuous_dim
"""
"ppo_gru_hybrid"
class Actor(parl.Model):
    def __init__(self, input_output_info):
        super(Actor, self).__init__()
        self.hp=input_output_info['hyperparam']
        self.gru_hid_dim=self.hp.PPO_GRU_HID_DIM
        self.obs_dim=input_output_info['obs_dim']
        
        # self.action_env_dim=input_output_info['action_env_discrete_dim']+input_output_info['action_env_continuous_dim']
        self.action_env_discrete_dim=input_output_info['action_env_discrete_dim']
        self.action_env_continuous_dim=input_output_info['action_env_continuous_dim']
        "从obs到gru"
        self.obs_to_gru=nn.Sequential(
            nn.Linear(self.obs_dim,self.hp.PPO_ACTOR_ENCODER_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_ACTOR_ENCODER_FC_DIM,self.hp.PPO_ACTOR_ENCODER_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_ACTOR_ENCODER_FC_DIM,self.hp.PPO_ACTOR_ENCODER_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_ACTOR_ENCODER_FC_DIM,self.hp.PPO_ACTOR_ENCODER_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_ACTOR_ENCODER_FC_DIM,self.hp.PPO_ACTOR_ENCODER_FC_DIM),nl_func(),
            )

        self.gru=nn.GRU(input_size=self.hp.PPO_ACTOR_ENCODER_FC_DIM,hidden_size=self.gru_hid_dim)
        self.gru.flatten_parameters()
        
        "从gru到env_action"
        self.gru_to_env_discrete_action=nn.Sequential(
            nn.Linear(self.gru_hid_dim,self.hp.PPO_ACTOR_HEAD_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_ACTOR_HEAD_FC_DIM,self.hp.PPO_ACTOR_HEAD_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_ACTOR_HEAD_FC_DIM,self.hp.PPO_ACTOR_HEAD_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_ACTOR_HEAD_FC_DIM,self.action_env_discrete_dim)
            )
        self.gru_to_env_continuous_action=nn.Sequential(
            nn.Linear(self.gru_hid_dim,self.hp.PPO_ACTOR_HEAD_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_ACTOR_HEAD_FC_DIM,self.hp.PPO_ACTOR_HEAD_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_ACTOR_HEAD_FC_DIM,self.hp.PPO_ACTOR_HEAD_FC_DIM),nl_func(),
            nn.Linear(self.hp.PPO_ACTOR_HEAD_FC_DIM,self.action_env_continuous_dim*2)
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
    def forward(self,input_dict,train,h_dict):
        """
        若train=True，代表训练模式，需要指定h，h_dict！=None
        若train=False，代表与环境交互模式，不需要指定h，调用actor自身的h
        LSTM输入
        x=paddle.randn([batch_size,time_steps,input_size])
        pre_h=paddle.randn([num_layers*direction_num,batch_size,hid_dim])
        pre_c=paddle.randn([num_layers*direction_num,batch_size,hid_dim])
        输出
        y,(h,c)=LSTM(x,(pre_h,pre_c))
        y.shape=[batch_size,time_steps,input_size]

        
        """
        init_h=h_dict['actor_init_h']
        env_vec=input_dict['env_vec']

        batch_size,time_len=env_vec.shape[0],env_vec.shape[1]

        feature_to_gru=self.squeeze_time_layer(env_vec,self.obs_to_gru)

        "gru1"
        if train:
            if init_h is None:
                raise("actor train mode must input init_h")
            pass
        else:
            init_h=self.h
        gru_out,h=self.gru(feature_to_gru,init_h)


        "action"
        action_discrete_logits=self.squeeze_time_layer(gru_out,self.gru_to_env_discrete_action)
        action_discrete_probs=F.softmax(action_discrete_logits,axis=-1)

        action_continuous_logits=self.squeeze_time_layer(gru_out,self.gru_to_env_continuous_action)

        action_continuous_mean,action_continuous_std=action_continuous_logits[:,:,:self.action_env_continuous_dim],action_continuous_logits[:,:,self.action_env_continuous_dim:]
        action_continuous_mean=paddle.tanh(action_continuous_mean)*1.0

        action_continuous_std=F.sigmoid(action_continuous_std*self.hp.PPO_ACTOR_CONTINUOUS_SCALE_LOGITS)*self.hp.PPO_ACTOR_CONTINUOUS_SCALE_STD+self.hp.PPO_ACTOR_CONTINUOUS_MIN_STD
        # action_continuous_std=F.sigmoid(action_continuous_std)*0+0.4
        action_continuous_logstd=paddle.log(action_continuous_std)


        action_dict={
            # "action_discrete":action_discrete_logits,
            # "action_continuous":action_continuous_logits,
            "action_discrete_probs":action_discrete_probs,
            "action_continuous_mean":action_continuous_mean,
            "action_continuous_logstd":action_continuous_logstd
        }


        if train:
            pass
        else:
            self.h=h.detach()

        return action_dict

class Actor_Static(Actor):
    @to_static
    def forward(self,input_dict,train,h_dict):
        return super(Actor_Static,self).forward(input_dict,train,h_dict)
