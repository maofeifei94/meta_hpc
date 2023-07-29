import imp
from jinja2 import pass_environment
import paddle
import argparse
from collections import deque
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
import hyperparam as hp
import time
from paddle.jit import to_static
import parl
import paddle
import paddle.optimizer as optim
import paddle.nn as nn
from paddle.distribution import Normal
import numpy as np
from myf_ML_util import timer_tool

def nl_func():
    return nn.LeakyReLU(negative_slope=0.1)
def to_time_paddle(x,_type,b,t):
    if isinstance(x,list):
        x=np.array(x)
    if isinstance(x,np.ndarray):
        x=paddle.to_tensor(x)
    

    if _type=='vec':
        x=paddle.reshape(x,[b,t,x.shape[-1]])
        pass
    elif _type=="img":
        x_shape=x.shape
        x=paddle.reshape(x,[b,t,*x_shape[-3:]])
        pass
    return x



    


class Conv_Encoder_net(nn.Layer):
    def __init__(self,img_shape):
        super(Conv_Encoder_net,self).__init__()
        self.conv1=nn.Conv2D(
            in_channels=img_shape[0],out_channels=4,
            kernel_size=[3,3],stride=[2,2],padding='SAME')
        self.conv2=nn.Conv2D(
            in_channels=4,out_channels=4,
            kernel_size=[3,3],stride=[2,2],padding='SAME')
        self.conv3=nn.Conv2D(
            in_channels=4,out_channels=4,
            kernel_size=[3,3],stride=[2,2],padding='SAME')
        self.conv4=nn.Conv2D(
            in_channels=4,out_channels=4,
            kernel_size=[3,3],stride=[1,1],padding='SAME')
        self.conv5=nn.Conv2D(
            in_channels=4,out_channels=4,
            kernel_size=[3,3],stride=[1,1],padding='SAME') 
        self.conv_net=nn.Sequential(
            self.conv1,nl_func(),
            self.conv2,nl_func(),
            self.conv3,nl_func(),
            self.conv4,nl_func(),
            self.conv5,nl_func(),
        )

    def forward(self,img):
        return self.conv_net(img)
"""
input_output_info={
    'env_img_shape'=[4,64,64],
    'env_img_conv_vec_dim'=256,
    'env_vec_dim'=2,

    'hpc_img_shape'=[1,64,64],
    'hpc_img_conv_vec_dim'=256,
    'hpc_vec_dim'=2,

    'action_env_dim'=2,
    'action_hpc_dim'=2
}
input_output_info['env_img_shape']
input_output_info['env_img_conv_vec_dim']
input_output_info['env_vec_dim']

input_output_info['hpc_img_shape']
input_output_info['hpc_img_conv_vec_dim']
input_output_info['hpc_vec_dim']

input_output_info['action_env_dim']
input_output_info['action_hpc_dim']

input_output_info['actor_gru1_to_gru2_vec_dim']
input_output_info['actor_gru1_hid_dim']
input_output_info['actor_gru2_hid_dim']

input_dict={
    'env_img'
    'env_vec'
    'hpc_img'
    'hpc_vec'
    'action_env'
    'action_hpc'
}
h_dict={
    'actor_init_h1'
    'actor_init_h2'
    'critic_init_h'
}
"""
class pfc_actor(parl.Model):
    def __init__(self, input_output_info):
        super(pfc_actor, self).__init__()
        gru1_hid_dim=input_output_info['actor_gru1_hid_dim']
        gru2_hid_dim=input_output_info['actor_gru2_hid_dim']
        self.gru1_hid_dim,self.gru2_hid_dim=gru1_hid_dim,gru2_hid_dim
        gru1_to_gru2_vec_dim=input_output_info['actor_gru1_to_gru2_vec_dim']

        env_all_vec_dim=input_output_info['env_img_conv_vec_dim']+input_output_info['env_vec_dim']
        hpc_all_vec_dim=input_output_info['hpc_img_conv_vec_dim']+input_output_info['hpc_vec_dim']

        "从env_img place1 到obs1"
        self.env_img_to_obs1=Conv_Encoder_net(input_output_info['env_img_shape'])

        "从hpc_img place1 到obs2"
        self.hpc_img_to_obs2=Conv_Encoder_net(input_output_info['hpc_img_shape'])

        "从obs到gru1"
        self.obs_to_gru1=nn.Sequential(
            nn.Linear(env_all_vec_dim+hpc_all_vec_dim,128),
            nl_func(),
            nn.Linear(128,128),nl_func()
            )

        self.gru1=nn.GRU(input_size=128,hidden_size=gru1_hid_dim)

        "从gru1到gru2"
        self.gru1_to_gru2=nn.Sequential(
            nn.Linear(gru1_hid_dim+env_all_vec_dim,gru1_to_gru2_vec_dim),nl_func()
            )
        self.gru2=nn.GRU(input_size=gru1_to_gru2_vec_dim,hidden_size=gru2_hid_dim)

        "从gru2到env_action"
        self.gru2_to_env_action_mean=nn.Sequential(
            nn.Linear(gru2_hid_dim,input_output_info['action_env_dim'])
            )
        self.gru2_to_env_action_logstd=nn.Sequential(
            nn.Linear(gru2_hid_dim,input_output_info['action_env_dim'])
            )

        "从gru1到hpc_action"
        self.gru1_to_hpc_action_mean=nn.Sequential(
            nn.Linear(gru1_hid_dim,input_output_info['action_hpc_dim'])
            )
        self.gru1_to_hpc_action_logstd=nn.Sequential(
            nn.Linear(gru1_hid_dim,input_output_info['action_hpc_dim'])
            )

        self.reset()

    def reset(self):
        self.h1=paddle.zeros([1,1,self.gru1_hid_dim])
        self.h2=paddle.zeros([1,1,self.gru2_hid_dim])
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
        LSTM输入
        x=paddle.randn([batch_size,time_steps,input_size])
        pre_h=paddle.randn([num_layers*direction_num,batch_size,hid_dim])
        pre_c=paddle.randn([num_layers*direction_num,batch_size,hid_dim])
        输出
        y,(h,c)=LSTM(x,(pre_h,pre_c))
        y.shape=[batch_size,time_steps,input_size]
        """
        init_h1,init_h2=h_dict['actor_init_h1'],h_dict['actor_init_h2']

        env_img,env_vec,hpc_img,hpc_vec=input_dict['env_img'],input_dict['env_vec'],input_dict['hpc_img'],input_dict['hpc_vec']
        batch_size,time_len=env_img.shape[0],env_img.shape[1]

        env_img_vec=paddle.reshape(self.squeeze_time_layer(env_img,self.env_img_to_obs1),[batch_size,time_len,-1])
        env_all_vec=paddle.concat([env_img_vec,env_vec],axis=-1)

        hpc_img_vec=paddle.reshape(self.squeeze_time_layer(hpc_img,self.hpc_img_to_obs2),[batch_size,time_len,-1])
        hpc_all_vec=paddle.concat([hpc_img_vec,hpc_vec],axis=-1)
 
        hid_to_gru1=self.squeeze_time_layer(paddle.concat([env_all_vec,hpc_all_vec],axis=-1),self.obs_to_gru1)
        "gru1"
        if train:
            if init_h1 is None or init_h2 is None:
                raise("actor train mode must input init_h")
            pass
        else:
            init_h1=self.h1
            init_h2=self.h2
        gru1_out,h1=self.gru1(hid_to_gru1,init_h1)

        "gru2"
        hid_to_gru2=self.squeeze_time_layer(paddle.concat([gru1_out,env_all_vec],axis=-1),self.gru1_to_gru2)
        gru2_out,h2=self.gru2(hid_to_gru2,init_h2)
        
        "action"
        action_env_mean=self.squeeze_time_layer(gru2_out,self.gru2_to_env_action_mean)
        action_env_logstd=self.squeeze_time_layer(gru2_out,self.gru2_to_env_action_logstd)
        action_hpc_mean=self.squeeze_time_layer(gru1_out,self.gru1_to_hpc_action_mean)
        action_hpc_logstd=self.squeeze_time_layer(gru1_out,self.gru1_to_hpc_action_logstd)

        action_dict={
            'action_mean':paddle.concat([action_env_mean,action_hpc_mean],axis=-1),
            'action_logstd':paddle.concat([action_env_logstd,action_hpc_logstd],axis=-1),
            'action_env_mean':action_env_mean,
            'action_env_logstd':action_env_logstd,
            'action_hpc_mean':action_hpc_mean,
            'action_hpc_logstd':action_hpc_logstd
            }

        if train:
            pass
        else:
            self.h1=h1
            self.h2=h2
        
        return action_dict



class pfc_critic(parl.Model):
    def __init__(self, input_output_info):
        # print("gru_ppo init_param=",img_shape,obs_dim,int(img_shape[2]*img_shape[1]/8**2)+obs_dim)
        super(pfc_critic, self).__init__()
        gru_hid_dim=input_output_info['critic_gru_hid_dim']
        self.gru_hid_dim=gru_hid_dim

        env_all_vec_dim=input_output_info['env_img_conv_vec_dim']+input_output_info['env_vec_dim']
        hpc_all_vec_dim=input_output_info['hpc_img_conv_vec_dim']+input_output_info['hpc_vec_dim']

        "从env_img place1 到obs1"
        self.env_img_to_obs1=Conv_Encoder_net(input_output_info['env_img_shape'])

        "从hpc_img place1 到obs2"
        self.hpc_img_to_obs2=Conv_Encoder_net(input_output_info['hpc_img_shape'])

        "从obs到gru"
        self.obs_to_gru=nn.Sequential(
            nn.Linear(env_all_vec_dim+hpc_all_vec_dim,128),nl_func(),
            nn.Linear(128,128),nl_func()
            )

        "gru"
        self.gru=nn.GRU(input_size=128,hidden_size=gru_hid_dim)

        "从gru到value"
        self.gru_to_value=nn.Sequential(
            nn.Linear(gru_hid_dim,128),nl_func(),
            nn.Linear(128,128),nl_func(),
            nn.Linear(128,1)
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

        env_img,env_vec,hpc_img,hpc_vec=[input_dict[_name] for _name in ['env_img','env_vec','hpc_img','hpc_vec']]
        batch_size,time_len=env_img.shape[0],env_img.shape[1]

        env_img_vec=paddle.reshape(self.squeeze_time_layer(env_img,self.env_img_to_obs1),[batch_size,time_len,-1])
        env_all_vec=paddle.concat([env_img_vec,env_vec],axis=-1)

        hpc_img_vec=paddle.reshape(self.squeeze_time_layer(hpc_img,self.hpc_img_to_obs2),[batch_size,time_len,-1])
        hpc_all_vec=paddle.concat([hpc_img_vec,hpc_vec],axis=-1)

        hid_to_gru=self.squeeze_time_layer(paddle.concat([env_all_vec,hpc_all_vec],axis=-1),self.obs_to_gru)

        "gru"
        if train:
            if init_h is None:
                raise("actor train mode must input init_h")
            pass
        else:
            init_h=self.h

        gru_out,h=self.gru(hid_to_gru,init_h)

        if train:
            pass
        else:
            self.h=h

        value_pred=self.squeeze_time_layer(gru_out,self.gru_to_value)
        value_dict={'value':value_pred}
        return value_dict

class pfc_actor_static(pfc_actor):
    @to_static
    def forward(self,input_dict,train,h_dict):
        return super(pfc_actor_static,self).forward(input_dict,train,h_dict)

class pfc_critic_static(pfc_critic):
    @to_static
    def forward(self,input_dict,train,h_dict):
        return super(pfc_critic_static,self).forward(input_dict,train,h_dict)

class GRU_PPO_Model(parl.Model):
    def __init__(self, input_output_info):
        super(GRU_PPO_Model, self).__init__()
        if input_output_info['to_static']:
            self.actor_model = pfc_actor_static(input_output_info)
            self.critic_model = pfc_critic_static(input_output_info)
        else:
            self.actor_model = pfc_actor(input_output_info)
            self.critic_model =pfc_critic(input_output_info)

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

"PPO负责组装"
class GRUPPO(parl.Algorithm):
    def __init__(self,
                 model:GRU_PPO_Model,
                 clip_param,
                 value_loss_coef,
                 entropy_coef,
                 initial_lr,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):
        """ PPO algorithm
        Args:
            model (parl.Model): model that contains both value network and policy network
            clip_param (float): the clipping strength for value loss clipping
            value_loss_coef (float): the coefficient for value loss (c_1)
            entropy_coef (float): the coefficient for entropy (c_2)
            initial_lr (float): initial learning rate.
            eps (None or float): epsilon for Adam optimizer
            max_grad_norm (float): threshold for grad norm clipping
            use_clipped_value_loss (bool): whether use value loss clipping
        """
        super().__init__(model)
        self.clip_param = clip_param

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        clip = nn.ClipGradByNorm(max_grad_norm)

        self.optimizer = optim.Adam(
            parameters=model.parameters(),
            learning_rate=initial_lr,
            epsilon=eps,
            grad_clip=clip)

    def learn(self, train_dict):
        """ update the value network and policy network parameters.
        """
        # for key in train_dict:
        #     print(key,train_dict[key].shape)
        
        # img_batch,obs_batch, actions_batch, value_preds_batch, return_batch,old_action_log_probs_batch, adv_targ,train_control,actor_init_h,critic_init_h
        # for key in train_dict.keys():
        #     print(['train_dict',key,train_dict[key].shape])
        def sprint(x,name):
            pass
            # print(name,x.shape)
        value_dict = self.model.value(train_dict,train=True,h_dict=train_dict)
        values=value_dict['value']
        sprint(values,"values")
        

        # log std so the std is always positive after e^{log_std}
        action_dict = self.model.policy(train_dict,train=True,h_dict=train_dict)
        sprint(action_dict['action_mean'],'action_mean')
        sprint(action_dict['action_logstd'],'action_logstd')
        mean,logstd=action_dict['action_mean'],action_dict['action_logstd']
        dist = Normal(mean, logstd.exp())


        # print(dist.shape)
        # input()

        
        # Continuous actions are usually considered to be independent,
        # so we can sum components of the ``log_prob`` or the entropy.
        action_log_probs = dist.log_prob(train_dict['actions_batch']).sum(
            axis=-1, keepdim=True)
        sprint(action_log_probs,'action_log_probs')
        sprint(dist.entropy(),"dist.entropy")
        # sprint(dist.entropy().sum(axis=-1)*train_dict['train_control'],"dist.entropy*train_control no keepdim")
        dist_entropy = (dist.entropy().sum(axis=-1,keepdim=True)*train_dict['train_control']).mean()
        # sprint(dist.entropy().sum(axis=-1,keepdim=True)*train_dict['train_control'],"dist.entropy*train_control keepdim")
        sprint(dist_entropy,'dist_entropy')

        ratio = paddle.exp(action_log_probs - train_dict['old_action_log_probs_batch'])
        sprint(ratio,'ratio')
        ratio_in_clip=((ratio-(1-self.clip_param))*(ratio-(1+self.clip_param)))<0

        surr1 = ratio * train_dict['adv_targ']
        surr2 = paddle.clip(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * train_dict['adv_targ']
        sprint(train_dict['adv_targ'],'adv_targ')
        action_loss = -(paddle.minimum(surr1, surr2)*train_dict['train_control']).mean()
        sprint(train_dict['train_control'],'train_control')
        # input()
        # print(action_loss)

        # calculate value loss using semi gradient TD
        if self.use_clipped_value_loss:
            value_pred_clipped = train_dict['value_preds_batch'] + \
                (values - train_dict['value_preds_batch']).clip(-self.clip_param, self.clip_param)
            value_losses = (values - train_dict['returns_batch']).pow(2)
            value_losses_clipped = (value_pred_clipped - train_dict['returns_batch']).pow(2)
            value_loss = (0.5 * paddle.maximum(value_losses,value_losses_clipped)*train_dict['train_control']).mean()
        else:
            value_loss = (0.5 * (train_dict['returns_batch'] - values).pow(2)*train_dict['train_control']).mean()

        (value_loss * self.value_loss_coef + action_loss -dist_entropy * self.entropy_coef).backward()
        self.optimizer.step()
        self.optimizer.clear_grad()

        return value_loss.numpy(), action_loss.numpy(), dist_entropy.numpy()

    def sample(self, input_dict,train,h_dict):
        """ Sample action from parameterized policy
        """
        # obs=np.reshape(obs,[])
        # tt=timer_tool("agent sample")
        value_dict = self.model.value(input_dict,train,h_dict)
        value=value_dict['value']

        action_dict= self.model.policy(input_dict,train,h_dict)
        action_mean,action_logstd=action_dict['action_mean'],action_dict['action_logstd']
        # print("action_mean",action_mean.shape)
        dist = Normal(action_mean,action_logstd.exp())
        action = dist.sample([1])[0]
        # print("action_sample",action.shape)
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)

        return value, action, action_log_probs

    def predict(self, input_dict,train,h_dict):
        """ Predict action from parameterized policy, action with maximum probability is selected as greedy action
        """
        action_dict= self.model.policy(input_dict,train,h_dict)
        action_mean=action_dict['action_mean']
        return action_mean

    def value(self, input_dict,train,h_dict):
        """ Predict value from parameterized value function
        """
        value_dict = self.model.value(input_dict,train,h_dict)
        value=value_dict['value']
        return value
class GRUPPOAgent(parl.Agent):
    """ Agent of Mujoco env

    Args:
        algorithm (`parl.Algorithm`): algorithm to be used in this agent.
    """

    def __init__(self, algorithm,input_output_info):
        super(GRUPPOAgent, self).__init__(algorithm)
        self.train_batchsize=input_output_info['train_batchsize']

    def predict(self,input_dict,train,h_dict):
        """ Predict action from current policy given observation

        Args:
            obs (np.array): observation
        """

        action = self.alg.predict(input_dict,train,h_dict)

        return action.detach().numpy()

    def sample(self,input_dict,train,h_dict):
        """ Sample action from current policy given observation

        Args:
            obs (np.array): observation
        """
        value, action, action_log_probs = self.alg.sample(input_dict,train,h_dict)
        return value.detach().numpy(), action.detach().numpy(), action_log_probs.detach().numpy()
    def reset(self):
        self.alg.model.actor_model.reset()
        self.alg.model.critic_model.reset()
    
    def learn(self, next_value, gamma, gae_lambda, ppo_epoch, num_mini_batch,
              rollouts):
        """ Learn current batch of rollout for ppo_epoch epochs.

        Args:
            next_value (np.array): next predicted value for calculating advantage
            gamma (float): the discounting factor
            gae_lambda (float): lambda for calculating n step return
            ppo_epoch (int): number of epochs K
            num_mini_batch (int): number of mini-batches
            rollouts (RolloutStorage): the rollout storage that contains the current rollout
        """
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        t=0
        for e in range(ppo_epoch):
            data_generator = rollouts.sample_batch(next_value, gamma,gae_lambda, num_mini_batch)

            sample_dict_list=[]
            for sample_dict in data_generator:
                # ori_train_dict={key:paddle.to_tensor(sample_dict[key]) for key in sample_dict.keys()}
                # print([[key,sample_dict[key].shape] for key in sample_dict.keys()])
                # for key in sample_dict.keys():
                #     print([key,sample_dict[key].shape])
                # input()
                sample_dict_list.append(sample_dict)
                if len(sample_dict_list)>=self.train_batchsize:
                    sample_dict_list_for_train=sample_dict_list
                    sample_dict_list=[]
                else:
                    continue

                train_dict={
                    "train_control":to_time_paddle([sd['train_control'] for sd in sample_dict_list_for_train],'vec',self.train_batchsize,-1),

                    "actor_init_h1":to_time_paddle([sd['actor_init_h1'] for sd in sample_dict_list_for_train],'vec',1,self.train_batchsize),
                    "actor_init_h2":to_time_paddle([sd['actor_init_h2'] for sd in sample_dict_list_for_train],'vec',1,self.train_batchsize),
                    "critic_init_h":to_time_paddle([sd['critic_init_h'] for sd in sample_dict_list_for_train],'vec',1,self.train_batchsize),

                    "env_img":to_time_paddle([sd['env_img'] for sd in sample_dict_list_for_train],'img',self.train_batchsize,-1),
                    "env_vec":to_time_paddle([sd['env_vec'] for sd in sample_dict_list_for_train],'vec',self.train_batchsize,-1),
                    "hpc_img":to_time_paddle([sd['hpc_img'] for sd in sample_dict_list_for_train],'img',self.train_batchsize,-1),
                    "hpc_vec":to_time_paddle([sd['hpc_vec'] for sd in sample_dict_list_for_train],'vec',self.train_batchsize,-1),

                    "actions_batch":to_time_paddle([sd['actions_batch'] for sd in sample_dict_list_for_train],'vec',self.train_batchsize,-1),
                    "old_action_log_probs_batch":to_time_paddle([sd['old_action_log_probs_batch'] for sd in sample_dict_list_for_train],'vec',self.train_batchsize,-1),

                    "value_preds_batch":to_time_paddle([sd['value_preds_batch'] for sd in sample_dict_list_for_train],'vec',self.train_batchsize,-1),
                    "returns_batch":to_time_paddle([sd['returns_batch'] for sd in sample_dict_list_for_train],'vec',self.train_batchsize,-1),
                    "adv_targ":to_time_paddle([sd['adv_targ'] for sd in sample_dict_list_for_train],'vec',self.train_batchsize,-1),
                }

                t+=1
                # while 1:
                value_loss, action_loss, dist_entropy = self.alg.learn(train_dict)
                    # print("still train loss=",value_loss, action_loss, dist_entropy)

                value_loss_epoch += value_loss
                action_loss_epoch += action_loss
                dist_entropy_epoch += dist_entropy

        num_updates = ppo_epoch * num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def value(self,input_dict,train,h_dict):
        """ Predict value from current value function given observation

        Args:
            obs (np.array): observation
        """

        val = self.alg.value(input_dict,train,h_dict)

        return val.detach().numpy()
    def save_model(self,save_dir,iter_num):
        print(f"start_save_model {save_dir}/{iter_num}_model.pdparams")
        paddle.save(self.alg.model.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")
        print(f"save model {save_dir}/{iter_num}_model.pdparams success")
        
    def load_model(self,save_dir,iter_num):
        print(f"start load_model {save_dir}/{iter_num}_model.pdparams")
        params_state = paddle.load(path=f"{save_dir}/{iter_num}_model.pdparams")
        self.alg.model.set_state_dict(params_state)
        print(f"load model {save_dir}/{iter_num}_model.pdparams success")
class RolloutStorage(object):
    def __init__(self, num_steps, input_output_info):
        self.num_steps = num_steps
        self.train_batchsize=input_output_info['train_batchsize']

        self.env_img=np.zeros((num_steps + 1, *input_output_info['env_img_shape']), dtype='float32')
        self.env_vec = np.zeros((num_steps + 1, input_output_info['env_vec_dim']), dtype='float32')
        self.hpc_img=np.zeros((num_steps + 1, *input_output_info['hpc_img_shape']), dtype='float32')
        self.hpc_vec = np.zeros((num_steps + 1, input_output_info['hpc_vec_dim']), dtype='float32')

        self.actions = np.zeros((num_steps, input_output_info['action_dim']), dtype='float32')
        self.action_log_probs = np.zeros((num_steps, ), dtype='float32')

        self.value_preds = np.zeros((num_steps + 1, ), dtype='float32')
        self.returns = np.zeros((num_steps + 1, ), dtype='float32')
        
        self.rewards = np.zeros((num_steps, ), dtype='float32')

        self.actor_gru1_h=np.zeros((num_steps,input_output_info['actor_gru1_hid_dim']),dtype='float32')
        self.actor_gru2_h=np.zeros((num_steps,input_output_info['actor_gru2_hid_dim']),dtype='float32')
        self.critic_gru_h=np.zeros((num_steps,input_output_info['critic_gru_hid_dim']),dtype='float32')

        self.masks = np.ones((num_steps + 1, ), dtype='bool')
        self.bad_masks = np.ones((num_steps + 1, ), dtype='bool')

        self.step = 0

    def append(self, collect_dict):
        self.env_img[self.step + 1] = collect_dict['env_img']
        self.env_vec[self.step + 1] = collect_dict['env_vec']
        self.hpc_img[self.step + 1] = collect_dict['hpc_img']
        self.hpc_vec[self.step + 1] = collect_dict['hpc_vec']

        self.actions[self.step] = collect_dict['action']
        self.action_log_probs[self.step] = collect_dict['action_log_prob']

        self.value_preds[self.step] = collect_dict['value_pred']

        self.rewards[self.step] = collect_dict['reward']

        self.actor_gru1_h[self.step]=collect_dict['actor_gru1_h']
        self.actor_gru2_h[self.step]=collect_dict['actor_gru2_h']
        self.critic_gru_h[self.step]=collect_dict['critic_gru_h']

        self.masks[self.step + 1] = collect_dict['mask']
        self.bad_masks[self.step + 1] = collect_dict['bad_mask']
        
        self.step = (self.step + 1) % self.num_steps
        # print(f"roll out storage step={self.step}")

    def _pad_vec(self,x,max_len):
        x_shape=np.shape(x)
        if len(x)==max_len:
            return x
        elif len(x_shape)==1:
            out=np.zeros([max_len],dtype=x.dtype)
            out[:len(x)]=x
            return out
        else:
            # print(max_len,x_shape)
            out=np.zeros([max_len,*x_shape[1:]],dtype=x.dtype)
            out[:len(x)]=x
            return out
    def gru_indices(self):
        self.train_history_len=hp.PPO_TRAIN_HISTORY_LEN
        self.train_control=[0.0 if i<self.train_history_len else 1.0 for i in range(self.train_history_len*2)]
        train_start_idx=np.random.randint(-self.train_history_len+2,self.num_steps-1)
        read_start_idx=train_start_idx-self.train_history_len
        end_idx=train_start_idx+self.train_history_len
        history_slice=slice(max(read_start_idx,0),min(end_idx,self.num_steps))
        train_control=np.reshape(np.array(self.train_control[max(0,-read_start_idx):2*self.train_history_len+min(self.num_steps-end_idx,0)]),[-1,1])
        return history_slice,train_control
    def sample_batch(self,
                     next_value,
                     gamma,
                     gae_lambda,
                     num_mini_batch,
                     mini_batch_size=None):
        # calculate return and advantage first
        self.compute_returns(next_value, gamma, gae_lambda)
        advantages = self.returns[:-1] - self.value_preds[:-1]

        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)
        
        # generate sample batch
        # mini_batch_size = self.num_steps // num_mini_batch
        # sampler = BatchSampler(
        #     sampler=RandomSampler(range(self.num_steps)),
        #     batch_size=mini_batch_size,
        #     drop_last=True)
        # print(self.num_steps,num_mini_batch)
        for i in range(8*self.train_batchsize):
            indices,train_control=self.gru_indices()
            train_control=self._pad_vec(train_control,self.train_history_len*2)
            actor_init_h1=np.zeros_like(self.actor_gru1_h[0]) if indices.start==0 else self.actor_gru1_h[indices.start-1]
            actor_init_h1=np.reshape(actor_init_h1,[1,1,-1])
            actor_init_h2=np.zeros_like(self.actor_gru2_h[0]) if indices.start==0 else self.actor_gru2_h[indices.start-1]
            actor_init_h2=np.reshape(actor_init_h2,[1,1,-1])
            critic_init_h=np.zeros_like(self.critic_gru_h[0]) if indices.start==0 else self.critic_gru_h[indices.start-1]
            critic_init_h=np.reshape(critic_init_h,[1,1,-1])
            # indices=slice(0,self.num_steps)
            # train_control=np.ones([self.num_steps])
            env_img_batch=self._pad_vec(self.env_img[:-1][indices],self.train_history_len*2)
            env_vec_batch=self._pad_vec(self.env_vec[:-1][indices],self.train_history_len*2)
            hpc_img_batch=self._pad_vec(self.hpc_img[:-1][indices],self.train_history_len*2)
            hpc_vec_batch=self._pad_vec(self.hpc_vec[:-1][indices],self.train_history_len*2)

            actions_batch=self._pad_vec(self.actions[indices],self.train_history_len*2)
            old_action_log_probs_batch = self._pad_vec(self.action_log_probs[indices],self.train_history_len*2)

            value_preds_batch = self._pad_vec(self.value_preds[:-1][indices],self.train_history_len*2)
            returns_batch = self._pad_vec(self.returns[:-1][indices],self.train_history_len*2)
            adv_targ = self._pad_vec(advantages[indices],self.train_history_len*2)

            "reshape"
            value_preds_batch = value_preds_batch.reshape(-1, 1)
            returns_batch = returns_batch.reshape(-1, 1)
            old_action_log_probs_batch = old_action_log_probs_batch.reshape(-1, 1)
            adv_targ = adv_targ.reshape(-1, 1)

            sample_dict={
                "train_control":train_control,
                "actor_init_h1":actor_init_h1,
                "actor_init_h2":actor_init_h2,
            
                "critic_init_h":critic_init_h,

                "env_img":env_img_batch,
                "env_vec":env_vec_batch,
                "hpc_img":hpc_img_batch,
                "hpc_vec":hpc_vec_batch,

                "actions_batch":actions_batch,
                "old_action_log_probs_batch":old_action_log_probs_batch,

                "value_preds_batch":value_preds_batch,
                "returns_batch":returns_batch,
                "adv_targ":adv_targ,
            }
            yield sample_dict

    def after_update(self):
        self.env_img[0]=np.copy(self.env_img[-1])
        self.env_vec[0]=np.copy(self.env_vec[-1])
        self.hpc_img[0]=np.copy(self.hpc_img[-1])
        self.hpc_vec[0]=np.copy(self.hpc_vec[-1])

        self.masks[0] = np.copy(self.masks[-1])
        self.bad_masks[0] = np.copy(self.bad_masks[-1])

    def compute_returns(self, next_value, gamma, gae_lambda):

        # input()
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size)):
            delta = self.rewards[step] + gamma * self.value_preds[
                step + 1] * self.masks[step + 1] - self.value_preds[step]

            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae

            gae = gae * self.bad_masks[step + 1]
            
            self.returns[step] = gae + self.value_preds[step]
class PPO_GRU_Module():
    def __init__(self,input_output_info):
        # super(SACModule,self).__init__()
        # self.obs_dim=obs_dim
        # self.action_dim=action_dim
        # self.model = MujocoModel(obs_dim, action_dim)
        self.model=GRU_PPO_Model(input_output_info)
        self.algorithm = GRUPPO(
            model=self.model,
            clip_param=hp.PPO_CLIP_PARAM,
            value_loss_coef=hp.PPO_VALUE_LOSS_COEF,
            entropy_coef=hp.PPO_ENTROPY_COEF,
            initial_lr=hp.PPO_LR,
            eps=hp.PPO_EPS,
            max_grad_norm=hp.PPO_MAX_GRAD_NROM,
            use_clipped_value_loss=False)
        self.agent = GRUPPOAgent(self.algorithm,input_output_info)
        self.rpm = RolloutStorage(hp.PPO_NUM_STEPS, input_output_info)

        self._reset()
        self.interact_steps=0
        self.learn_steps=0
        self.warm_up_random=True
    def _reset(self):
        self.agent.reset()
    def _input(self,input_dict,train,h_dict):
        """
        s，a，r的历史记录
        HPC离散编码激活程度值
        內嗅皮层的未来预测值
        """
        value, action, action_log_prob=self.agent.sample(input_dict,train,h_dict)
        actor_gru_h1,actor_gru_h2,critic_gru_h=self.agent.alg.model.actor_model.h1.numpy(),self.agent.alg.model.actor_model.h2.numpy(),self.agent.alg.model.critic_model.h.numpy()
        # print(action.shape)
        output_dict={
            "value":np.squeeze(value),
            "action":np.squeeze(action),
            "action_log_prob":np.squeeze(action_log_prob),
            "actor_gru_h1":actor_gru_h1,
            "actor_gru_h2":actor_gru_h2,
            "critic_gru_h":critic_gru_h
        }
        return output_dict
    def _rpm_collect(self,collect_dict):
        # obs=self.obs_to_std(obs)
        # next_obs=self.obs_to_std(next_obs)
        self.rpm.append(collect_dict)
        # self.rpm.append(obs, action, reward, next_obs, terminal)
    def _learn(self):
        next_value = self.agent.value({
            'env_img':to_time_paddle(self.rpm.env_img[-1],'img',1,1),
            'env_vec':to_time_paddle(self.rpm.env_vec[-1],'vec',1,1),
            'hpc_img':to_time_paddle(self.rpm.hpc_img[-1],'img',1,1),
            'hpc_vec':to_time_paddle(self.rpm.hpc_vec[-1],'vec',1,1),
            },True,
            {
                'critic_init_h':to_time_paddle(self.rpm.critic_gru_h[-1],'vec',1,1),
            }
            )
        value_loss, action_loss, dist_entropy = self.agent.learn(
            next_value, hp.PPO_GAMMA, hp.PPO_GAE_LAMBDA, hp.PPO_EPOCH, hp.PPO_BATCH_SIZE, self.rpm)
        print(f"v_loss{value_loss},a_loss{action_loss},d_entropy{dist_entropy}")
        self.rpm.after_update()
    def save_model(self,save_dir,iter_num):
        print(f"start_save_model {save_dir}/{iter_num}_model.pdparams")
        paddle.save(self.model.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")
        print(f"save model {save_dir}/{iter_num}_model.pdparams success")
        
    def load_model(self,save_dir,iter_num,warm_up_random):
        print(f"start load_model {save_dir}/{iter_num}_model.pdparams")
        params_state = paddle.load(path=f"{save_dir}/{iter_num}_model.pdparams")
        self.model.set_state_dict(params_state)
        print(f"load model {save_dir}/{iter_num}_model.pdparams success")
        self.warm_up_random=warm_up_random
    def update(self,state_dict):
        self.model.set_state_dict(state_dict)
        print("update param success")