#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import imp
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
from pygame import init
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
saclock=threading.RLock()
def nl_func(x):
    return F.leaky_relu(x,negative_slope=0.1)
def trans_static(model):
    model=model.eval()
    model=to_static(model)
    return model

class GRU_PPO_Model(parl.Model):
    def __init__(self, img_shape,obs_dim, action_dim):
        super(GRU_PPO_Model, self).__init__()
        self.actor_model = GRU_PPO_Actor(img_shape,obs_dim, action_dim)
        # self.actor_model.eval()
        
        self.critic_model =GRU_PPO_Critic(img_shape,obs_dim)
        # self.critic_model.eval()
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

    def policy(self, img,obs,train,init_h):
        # print(type(img))
        # print(type(obs))
        tt=timer_tool("model policy")
        img=self.reshape_img(img)
        obs=self.reshape_obs(obs)
        out=self.actor_model(img,obs,train,init_h)
        tt.end("all")
        return out

    def value(self, img,obs,train,init_h):
        # print(type(img))
        tt=timer_tool("model value")
        img=self.reshape_img(img)
        obs=self.reshape_obs(obs)
        out=self.critic_model(img,obs,train,init_h)
        tt.end("all")
        return out


    def get_actor_params(self):

        return self.actor_model.parameters()

    def get_critic_params(self):

        return self.critic_model.parameters()
class Conv_net(nn.Layer):
    def __init__(self,img_shape):
        super(Conv_net,self).__init__()
        self.conv1=nn.Conv2D(
            in_channels=img_shape[0],out_channels=4,
            kernel_size=[3,3],stride=[2,2],padding='SAME')
        self.conv2=nn.Conv2D(
            in_channels=4,out_channels=4,
            kernel_size=[3,3],stride=[2,2],padding='SAME')
        self.conv3=nn.Conv2D(
            in_channels=4,out_channels=1,
            kernel_size=[3,3],stride=[2,2],padding='SAME')
    def forward(self,img):
        batch,time,channel,h,w=img.shape

        "batch,time,channel,h,w"
        x=paddle.reshape(img,[batch*time,channel,h,w])
        x=nl_func(self.conv1(x))
        x=nl_func(self.conv2(x))
        x=nl_func(self.conv3(x))
        x=paddle.reshape(x,[batch,time,-1])
        return x
        
class GRU_PPO_Actor(parl.Model):
    def __init__(self, img_shape,obs_dim,action_dim, lstm_hid_dim=hp.PPO_GRU_HID_DIM):
        super(GRU_PPO_Actor, self).__init__()
        self.convnet=Conv_net(img_shape)
        self.gru=nn.GRU(input_size=int(img_shape[2]*img_shape[2]/8**2)+obs_dim,hidden_size=lstm_hid_dim,num_layers=1)

        self.l1 = nn.Linear(lstm_hid_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64,64)
        self.mean_linear = nn.Linear(64, action_dim)
        self.std_linear = nn.Linear(64, action_dim)
        self.lstm_hid_dim=lstm_hid_dim
        self.reset()
    def reset(self):
        self.h=paddle.zeros([1,1,self.lstm_hid_dim])
    # @to_static
    def forward(self, img,obs,train,init_h):
        """
        LSTM输入
        x=paddle.randn([batch_size,time_steps,input_size])
        pre_h=paddle.randn([num_layers*direction_num,batch_size,hid_dim])
        pre_c=paddle.randn([num_layers*direction_num,batch_size,hid_dim])
        输出
        y,(h,c)=LSTM(x,(pre_h,pre_c))
        y.shape=[batch_size,time_steps,input_size]
        """

        # img_shape: batch,time,c,h,w
        # img_shape=img.shape
        # tt=timer_tool("actor")
        img_conv_result=self.convnet(img)
        # tt.end_and_start("conv")
        # print("actor input shape=",img_conv_result.shape,obs.shape)
        if train:
            if init_h is None:
                raise("actor train mode must input init_hc")
            else:
                batch_size,time_len,obs_dim=obs.shape
                x=paddle.concat([img_conv_result,obs],axis=-1)
                # saclock.acquire()
                x,h=self.gru(x,init_h)
                # saclock.release()
                # tt.end_and_start("gru")

                x=paddle.reshape(x,[batch_size*time_len,-1])
                x = nl_func(self.l1(x))
                x = nl_func(self.l2(x))
                x = nl_func(self.l3(x))

                # x=paddle.reshape(x,[batch_size,time_len,-1])

                # x,(h,c)=self.lstm(x,(init_h,init_c))
                # x=paddle.reshape(x,[batch_size*time_len,-1])

                act_mean = self.mean_linear(x)
                act_std = self.std_linear(x)
                # tt.end_and_start("mlp")
                act_log_std = paddle.clip(act_std, min=hp.SAC_LOG_SIG_MIN, max=hp.SAC_LOG_SIG_MAX)
                return act_mean,act_log_std
                # return paddle.reshape(act_mean,[batch_size,time_len,-1]),paddle.reshape(act_log_std,[batch_size,time_len,-1])
        else:

            self.pre_h=self.h


            batch_size,time_len,obs_dim=obs.shape

            # x=paddle.reshape(obs,[batch_size*time_len,obs_dim])
            x=paddle.concat([img_conv_result,obs],axis=-1)
            # saclock.acquire()

            x,self.h=self.gru(x,self.pre_h)

            # saclock.release()
            # tt.end_and_start("gru")
            x=paddle.reshape(x,[batch_size*time_len,-1])

            x = nl_func(self.l1(x))
            x = nl_func(self.l2(x))
            x = nl_func(self.l3(x))



            # x=paddle.reshape(x,[batch_size,time_len,-1])
            # x,(self.h,self.c)=self.lstm(x,(self.pre_h,self.pre_c))

            # x=paddle.reshape(x,[batch_size*time_len,-1])

            act_mean = self.mean_linear(x)
            act_std = self.std_linear(x)
            # tt.end_and_start("mlp")
            act_log_std = paddle.clip(act_std, min=hp.SAC_LOG_SIG_MIN, max=hp.SAC_LOG_SIG_MAX)
            return act_mean,act_log_std



class GRU_NET(nn.Layer):
    def __init__(self,input_size,hidden_size,num_layers):
        super(GRU_NET,self).__init__()
        self.gru=nn.GRU(input_size,hidden_size,num_layers)
        # print(input_size)
    # @to_static
    def forward(self,obs,h):
        batch_size,time_len,_=obs.shape

        # q1=paddle.reshape(q1,[batch_size,1,-1])
        # saclock.acquire()
        # print("grunet input shape=",obs.shape,h.shape)
        v,h=self.gru(obs,h)
        # saclock.release()
        return v,h
class V_NET(nn.Layer):
    def __init__(self,lstm_hid_dim):
        super(V_NET,self).__init__()
        self.l1 = nn.Linear(lstm_hid_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 64)
        self.l4 = nn.Linear(64, 1)
    # @to_static
    def forward(self,v):
        batch_size,time_len,_=v.shape
        v=paddle.reshape(v,[batch_size*time_len,-1])
        v= nl_func(self.l1(v))
        v= nl_func(self.l2(v))
        v= nl_func(self.l3(v))
        v = self.l4(v)
        return v

class GRU_PPO_Critic(parl.Model):
    def __init__(self, img_shape,obs_dim,lstm_hid_dim=hp.PPO_GRU_HID_DIM):
        # print("gru_ppo init_param=",img_shape,obs_dim,int(img_shape[2]*img_shape[1]/8**2)+obs_dim)
        super(GRU_PPO_Critic, self).__init__()
        self.lstm_hid_dim=lstm_hid_dim
        self.convnet=Conv_net(img_shape)
        self.gru_net=GRU_NET(input_size=int(img_shape[2]*img_shape[1]/8**2)+obs_dim,hidden_size=lstm_hid_dim,num_layers=1)
        # print("grunet_input_Dim=",img_shape[2]*img_shape[1]/8**2,obs_dim)
        self.v_net=V_NET(lstm_hid_dim)
        self.reset()
    def reset(self):
        self.h=paddle.zeros([1,1,self.lstm_hid_dim])
    # @to_static
    def forward(self, img,obs,train,init_h):
        """
        LSTM输入
        x=paddle.randn([batch_size,time_steps,input_size])
        pre_h=paddle.randn([num_layers*direction_num,batch_size,hid_dim])
        pre_c=paddle.randn([num_layers*direction_num,batch_size,hid_dim])
        输出
        y,(h,c)=LSTM(x,(pre_h,pre_c))
        """
        img_conv_result=self.convnet(img)
        # print(img_conv_result.shape,obs.shape)
        # print("critic input shape",img_conv_result.shape,obs.shape)
        # print("***********")
        x=paddle.concat([img_conv_result,obs],axis=-1)
        batch_size,time_len,_=obs.shape

        # print(img_conv_result.shape,obs.shape,self.gru_net)
        if train:
            # init_h=paddle.zeros([1,batch_size,self.lstm_hid_dim])
            v=self.v_net(self.gru_net(x,init_h)[0])
            return v
        else:
            self.pre_h=self.h
            v,self.h=self.gru_net(x,self.pre_h)
            v=self.v_net(v)
            return v
class GRUPPOAgent(parl.Agent):
    """ Agent of Mujoco env

    Args:
        algorithm (`parl.Algorithm`): algorithm to be used in this agent.
    """

    def __init__(self, algorithm):
        super(GRUPPOAgent, self).__init__(algorithm)

    def predict(self, img,obs,train,init_h_a):
        """ Predict action from current policy given observation

        Args:
            obs (np.array): observation
        """

        img=paddle.to_tensor(img, dtype='float32')
        obs = paddle.to_tensor(obs, dtype='float32')
        action = self.alg.predict(img,obs,train,init_h_a)

        return action.detach().numpy()

    def sample(self,img, obs,train,init_h_v,init_h_a):
        """ Sample action from current policy given observation

        Args:
            obs (np.array): observation
        """

        img=paddle.to_tensor(img, dtype='float32')
        obs = paddle.to_tensor(obs,dtype='float32')
        value, action, action_log_probs = self.alg.sample(img,obs,train,init_h_v,init_h_a)

        return value.detach().numpy(), action.detach().numpy(), \
            action_log_probs.detach().numpy()
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
            data_generator = rollouts.sample_batch(next_value, gamma,
                                                   gae_lambda, num_mini_batch)

            for sample in data_generator:
                img_batch,obs_batch, actions_batch, \
                    value_preds_batch, return_batch, old_action_log_probs_batch, \
                            adv_targ,train_control,actor_init_h,critic_init_h = sample
                img_batch= paddle.to_tensor(img_batch)
                obs_batch = paddle.to_tensor(obs_batch)
                actions_batch = paddle.to_tensor(actions_batch)
                value_preds_batch = paddle.to_tensor(value_preds_batch)
                return_batch = paddle.to_tensor(return_batch)
                old_action_log_probs_batch = paddle.to_tensor(
                    old_action_log_probs_batch)

                adv_targ = paddle.to_tensor(adv_targ)
                train_control=paddle.to_tensor(train_control)
                actor_init_h=paddle.to_tensor(actor_init_h)
                critic_init_h=paddle.to_tensor(critic_init_h)


                t+=1
                value_loss, action_loss, dist_entropy = self.alg.learn(
                    img_batch,obs_batch, actions_batch, value_preds_batch, return_batch,
                    old_action_log_probs_batch, adv_targ,train_control,actor_init_h,critic_init_h)

                value_loss_epoch += value_loss
                action_loss_epoch += action_loss
                dist_entropy_epoch += dist_entropy

        num_updates = ppo_epoch * num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def value(self,img, obs,train,init_h_v):
        """ Predict value from current value function given observation

        Args:
            obs (np.array): observation
        """
        img=paddle.to_tensor(img)
        obs = paddle.to_tensor(obs)
        val = self.alg.value(img,obs,train,init_h_v)

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

class GRUPPO(parl.Algorithm):
    def __init__(self,
                 model,
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

    def learn(self, img_batch,obs_batch, actions_batch, value_preds_batch, return_batch,
              old_action_log_probs_batch, adv_targ,train_control,actor_init_h,critic_init_h):
        """ update the value network and policy network parameters.
        """
        values = self.model.value(img_batch,obs_batch,train=True,init_h=critic_init_h)

        # log std so the std is always positive after e^{log_std}
        mean, log_std = self.model.policy(img_batch,obs_batch,train=True,init_h=actor_init_h)
        dist = Normal(mean, log_std.exp())

        
        # Continuous actions are usually considered to be independent,
        # so we can sum components of the ``log_prob`` or the entropy.
        action_log_probs = dist.log_prob(actions_batch).sum(
            axis=-1, keepdim=True)
        dist_entropy = (dist.entropy().sum(axis=-1)*train_control).mean()
        

        ratio = paddle.exp(action_log_probs - old_action_log_probs_batch)
        ratio_in_clip=((ratio-(1-self.clip_param))*(ratio-(1+self.clip_param)))<0

 

        surr1 = ratio * adv_targ
        surr2 = paddle.clip(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ
        action_loss = -(paddle.minimum(surr1, surr2)*train_control).mean()

        # calculate value loss using semi gradient TD
        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clip(-self.clip_param, self.clip_param)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
            value_loss = (0.5 * paddle.maximum(value_losses,value_losses_clipped)*train_control).mean()
        else:
            value_loss = (0.5 * (return_batch - values).pow(2)*train_control).mean()

        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()
        self.optimizer.step()
        self.optimizer.clear_grad()

        return value_loss.numpy(), action_loss.numpy(), dist_entropy.numpy()

    def sample(self, img,obs,train,init_h_v,init_h_a):
        """ Sample action from parameterized policy
        """
        # obs=np.reshape(obs,[])
        # tt=timer_tool("agent sample")
        value = self.model.value(img,obs,train,init_h_v)
        mean, log_std = self.model.policy(img,obs,train,init_h_a)
        dist = Normal(mean, log_std.exp())
        action = dist.sample([1])
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)

        return value, action, action_log_probs

    def predict(self, img,obs,train,init_h_a):
        """ Predict action from parameterized policy, action with maximum probability is selected as greedy action
        """
        mean, _ = self.model.policy(img,obs,train,init_h_a)
        return mean

    def value(self, img,obs,train,init_h_v):
        """ Predict value from parameterized value function
        """
        return self.model.value(img,obs,train,init_h_v)
class RolloutStorage(object):
    def __init__(self, num_steps, img_shape,obs_dim, act_dim,gru_hid_dim):
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.img=np.zeros((num_steps + 1, *img_shape), dtype='float32')
        self.obs = np.zeros((num_steps + 1, obs_dim), dtype='float32')
        self.actions = np.zeros((num_steps, act_dim), dtype='float32')
        self.gru_hid_dim=np.zeros((num_steps,gru_hid_dim),dtype='float32')
        self.value_preds = np.zeros((num_steps + 1, ), dtype='float32')
        self.returns = np.zeros((num_steps + 1, ), dtype='float32')
        self.action_log_probs = np.zeros((num_steps, ), dtype='float32')
        self.rewards = np.zeros((num_steps, ), dtype='float32')
        self.actor_gru_h=np.zeros((num_steps,gru_hid_dim),dtype='float32')
        self.critic_gru_h=np.zeros((num_steps,gru_hid_dim),dtype='float32')

        self.masks = np.ones((num_steps + 1, ), dtype='bool')
        self.bad_masks = np.ones((num_steps + 1, ), dtype='bool')

        self.step = 0

    def append(self, img,obs, actions, action_log_probs, value_preds, rewards,
               masks, bad_masks,actor_gru_h,critic_gru_h):
        self.img[self.step + 1] = img
        self.obs[self.step + 1] = obs
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.action_log_probs[self.step] = action_log_probs
        self.value_preds[self.step] = value_preds
        self.masks[self.step + 1] = masks
        self.bad_masks[self.step + 1] = bad_masks
        self.actor_gru_h[self.step]=actor_gru_h
        self.critic_gru_h[self.step]=critic_gru_h
        self.step = (self.step + 1) % self.num_steps
    def _pad_vec(self,x,max_len):
        x_shape=np.shape(x)

        if len(x)==max_len:
            return x
        elif len(x_shape)==1:
            out=np.zeros([max_len],dtype=x.dtype)
            out[:len(x)]=x
            return out
        else:
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
        train_control=np.array(self.train_control[max(0,-read_start_idx):2*self.train_history_len+min(self.num_steps-end_idx,0)])
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
        for i in range(32):
            indices,train_control=self.gru_indices()
            actor_init_h=np.zeros_like(self.actor_gru_h[0]) if indices.start==0 else self.actor_gru_h[indices.start-1]
            actor_init_h=np.reshape(actor_init_h,[1,1,-1])
            critic_init_h=np.zeros_like(self.critic_gru_h[0]) if indices.start==0 else self.critic_gru_h[indices.start-1]
            critic_init_h=np.reshape(critic_init_h,[1,1,-1])
            # indices=slice(0,self.num_steps)
            # train_control=np.ones([self.num_steps])
            img_batch=self.img[:-1][indices]
            obs_batch = self.obs[:-1][indices]
            actions_batch = self.actions[indices]
            value_preds_batch = self.value_preds[:-1][indices]
            returns_batch = self.returns[:-1][indices]
            old_action_log_probs_batch = self.action_log_probs[indices]

            value_preds_batch = value_preds_batch.reshape(-1, 1)
            returns_batch = returns_batch.reshape(-1, 1)
            old_action_log_probs_batch = old_action_log_probs_batch.reshape(
                -1, 1)

            adv_targ = advantages[indices]
            adv_targ = adv_targ.reshape(-1, 1)

            yield img_batch,obs_batch, actions_batch, value_preds_batch, returns_batch, old_action_log_probs_batch, adv_targ,train_control,actor_init_h,critic_init_h

    def after_update(self):
        self.img[0]=np.copy(self.img[-1])
        self.obs[0] = np.copy(self.obs[-1])
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
    def __init__(self,img_shape,obs_dim,action_dim):
        # super(SACModule,self).__init__()
        # self.obs_dim=obs_dim
        self.action_dim=action_dim
        # self.model = MujocoModel(obs_dim, action_dim)
        self.model=GRU_PPO_Model(img_shape,obs_dim,action_dim)
        self.algorithm = GRUPPO(
            model=self.model,
            clip_param=hp.PPO_CLIP_PARAM,
            value_loss_coef=hp.PPO_VALUE_LOSS_COEF,
            entropy_coef=hp.PPO_ENTROPY_COEF,
            initial_lr=hp.PPO_LR,
            eps=hp.PPO_EPS,
            max_grad_norm=hp.PPO_MAX_GRAD_NROM,
            use_clipped_value_loss=True)
        self.agent = GRUPPOAgent(self.algorithm)
        self.rpm = RolloutStorage(hp.PPO_NUM_STEPS, img_shape,obs_dim,action_dim,gru_hid_dim=hp.PPO_GRU_HID_DIM)
        # num_steps, img_shape,obs_dim, act_dim,gru_hid_dim
        self._reset()
        self.interact_steps=0
        self.learn_steps=0
        self.warm_up_random=True
    def _reset(self):
        self.agent.reset()
    def _input(self,img,obs):
        """
        s，a，r的历史记录
        HPC离散编码激活程度值
        內嗅皮层的未来预测值
        """
        value, action, action_log_prob=self.agent.sample(img,obs,False,None,None)
        actor_gru_h,critic_gru_h=self.agent.alg.model.actor_model.h.numpy(),self.agent.alg.model.critic_model.h.numpy()
            
        return np.squeeze(value),np.squeeze(action),np.squeeze(action_log_prob),actor_gru_h,critic_gru_h
    def _rpm_collect(self,img,obs, action, action_log_prob, value, reward, masks,bad_masks,actor_gru_h,critic_gru_h):
        # obs=self.obs_to_std(obs)
        # next_obs=self.obs_to_std(next_obs)

        self.rpm.append(img,obs, action, action_log_prob, value, reward,masks, bad_masks,actor_gru_h,critic_gru_h)
        # self.rpm.append(obs, action, reward, next_obs, terminal)
    def _learn(self):
        next_value = self.agent.value(self.rpm.img[-1],self.rpm.obs[-1],False,None)
        value_loss, action_loss, dist_entropy = self.agent.learn(
            next_value, hp.PPO_GAMMA, hp.PPO_GAE_LAMBDA, hp.PPO_EPOCH, hp.PPO_BATCH_SIZE, self.rpm)
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
    

