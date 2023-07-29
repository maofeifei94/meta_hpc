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


def nl_func(x):
    return F.leaky_relu(x,negative_slope=0.1)

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

class pfc_actor(parl.Model):
    def __init__(self, env_obs_dim,hpc_obs_dim,action_dim,gru1_to_gru2_vec_dim, gru1_hid_dim=hp.PPO_GRU_HID_DIM,gru2_hid_dim=hp.PPO_GRU_HID_DIM):
        super(pfc_actor, self).__init__()
        self.gru1_hid_dim=gru1_hid_dim
        self.gru2_hid_dim=gru2_hid_dim



        "从obs到gru1"
        self.obs_to_gru1=nn.Sequential(
            nn.Linear(env_obs_dim+hpc_obs_dim,128),nl_func(),
            nn.Linear(128,128),nl_func()
            )

        self.gru1=nn.GRU(input_size=128,hidden_size=gru1_hid_dim)

        "从gru1到gru2"
        self.gru1_to_gru2=nn.Sequential(
            nn.Linear(gru1_hid_dim,gru1_to_gru2_vec_dim),nl_func()
            )
        self.gru2=nn.GRU(input_size=env_obs_dim+gru1_to_gru2_vec_dim,hidden_size=gru2_hid_dim)

        "从gru2到action"
        self.gru2_to_action_mean=nn.Sequential(
            nn.Linear(gru2_hid_dim,action_dim)
            )
        self.gru2_to_action_std=nn.Sequential(
            nn.Linear(gru2_hid_dim,action_dim)
            )

        "从gru1到hpc"
        self.gru1_to_hpc=nn.Sequential(
            nn.Linear(gru1_hid_dim,hpc_obs_dim-1),nn.Sigmoid()
            )

        self.reset()

    def reset(self):
        self.h1=paddle.zeros([1,1,self.gru1_hid_dim])
        self.h2=paddle.zeros([1,1,self.gru2_hid_dim])
    def no_time_layer(self,x,layer):
        "batch和time合并，通过全连接层，再将batch和time分开"
        batch_size,time_len=x.shape[:2]
        x_no_time=paddle.reshape(x,[batch_size*time_len,-1])
        return paddle.reshape(layer(x_no_time),[batch_size,time_len,-1])

    def forward(self, env_input,hpc_input,train,init_h1,init_h2):
        """
        LSTM输入
        x=paddle.randn([batch_size,time_steps,input_size])
        pre_h=paddle.randn([num_layers*direction_num,batch_size,hid_dim])
        pre_c=paddle.randn([num_layers*direction_num,batch_size,hid_dim])
        输出
        y,(h,c)=LSTM(x,(pre_h,pre_c))
        y.shape=[batch_size,time_steps,input_size]
        """
        env_img_hid_vec,env_place_vec,env_reward=env_input
        hpc_img_hid_vec,hpc_place_vec,hpd_reward=hpc_input


        obs_all=paddle.concat([env_img_hid_vec,env_place_vec,env_reward,hpc_img_hid_vec,hpc_place_vec,hpd_reward],axis=-1)
        hid_to_gru1=self.no_time_layer(obs_all,self.obs_to_gru1)

        if train:
            if init_h1 is None or init_h2 is None:
                raise("actor train mode must input init_h")
            pass
        else:
            init_h1=self.h1
            init_h2=self.h2
        
        gru1_out,h1=self.gru1(hid_to_gru1,init_h1)

        hid_to_gru2=self.no_time_layer(gru1_out,self.gru1_to_gru2)
        gru2_out,h2=self.gru2(hid_to_gru2,init_h2)
        action_mean_env=self.no_time_layer(gru2_out,self.gru2_to_action_mean)
        action_std_env=self.no_time_layer(gru2_out,self.gru2_to_action_std)

        action_hpc=self.no_time_layer(gru1_out,self.gru1_to_hpc)

        if train:
            pass
        else:
            self.h1=h1
            self.h2=h2
        
        return action_mean_env,action_std_env,action_hpc
class pfc_critic(parl.Model):
    def __init__(self, env_obs_dim,gru_hid_dim=hp.PPO_GRU_HID_DIM):
        # print("gru_ppo init_param=",img_shape,obs_dim,int(img_shape[2]*img_shape[1]/8**2)+obs_dim)
        super(pfc_critic, self).__init__()
        self.gru_hid_dim=gru_hid_dim
        self.obs_to_gru=nn.Sequential(
            nn.Linear(env_obs_dim,128),nl_func(),
            nn.Linear(128,128),nl_func()
            )

        self.gru=nn.GRU(input_size=128,hidden_size=gru_hid_dim)

        self.gru_to_value=nn.Sequential(
            nn.Linear(gru_hid_dim,128),nl_func(),
            nn.Linear(128,128),nl_func(),
            nn.Linear(128,1)
            )

        self.reset()

    def reset(self):
        self.h=paddle.zeros([1,1,self.gru_hid_dim])

    def no_time_layer(self,x,layer):
        "batch和time合并，通过全连接层，再将batch和time分开"
        batch_size,time_len=x.shape[:2]
        x_no_time=paddle.reshape(x,[batch_size*time_len,-1])
        return paddle.reshape(layer(x_no_time),[batch_size,time_len,-1])
    def forward(self, env_input,train,init_h):
        """
        LSTM输入
        x=paddle.randn([batch_size,time_steps,input_size])
        pre_h=paddle.randn([num_layers*direction_num,batch_size,hid_dim])
        pre_c=paddle.randn([num_layers*direction_num,batch_size,hid_dim])
        输出
        y,(h,c)=LSTM(x,(pre_h,pre_c))

        """
        env_img_hid_vec,env_place_vec,env_reward=env_input
        obs_all=paddle.concat([env_img_hid_vec,env_place_vec,env_reward],axis=-1)
        hid_to_gru=self.no_time_layer(obs_all,self.obs_to_gru)

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

        value_pred=self.no_time_layer(gru_out,self.gru_to_value)

        return value_pred


class GRU_PPO_Model(parl.Model):
    def __init__(self, env_obs_dim,hpc_obs_dim,action_dim,gru1_to_gru2_vec_dim):
        super(GRU_PPO_Model, self).__init__()
        self.actor_model = pfc_actor(env_obs_dim,hpc_obs_dim,action_dim,gru1_to_gru2_vec_dim)
        self.critic_model =pfc_critic(env_obs_dim)

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
    def policy(self, env_input,hpc_input,train,init_h1,init_h2):
        tt=timer_tool("model policy")
        action_mean_env,action_std_env,action_hpc=self.actor_model(env_input,hpc_input,train,init_h1,init_h2)
        tt.end("all")
        return action_mean_env,action_std_env,action_hpc
    def value(self, env_input,train,init_h):
        tt=timer_tool("model value")
        value=self.critic_model(env_input,train,init_h)
        tt.end("all")
        return value
    def get_actor_params(self):
        return self.actor_model.parameters()
    def get_critic_params(self):
        return self.critic_model.parameters()

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

    def sample(self, env_input,hpc_input,train,init_h1,init_h2,init_h_value):
        """ Sample action from parameterized policy
        """
        # obs=np.reshape(obs,[])
        # tt=timer_tool("agent sample")
        value = self.model.value(env_input,train,init_h_value)
        action_mean_env,action_std_env,action_hpc= self.model.policy(env_input,hpc_input,train,init_h1,init_h2)
        dist = Normal(action_mean_env, action_std_env.exp())
        action = dist.sample([1])
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)

        return value, action, action_log_probs

    def predict(self, env_input,hpc_input,train,init_h1,init_h2):
        """ Predict action from parameterized policy, action with maximum probability is selected as greedy action
        """
        mean, _ ,_= self.model.policy(env_input,hpc_input,train,init_h1,init_h2)
        return mean

    def value(self, env_input,train,init_h_value):
        """ Predict value from parameterized value function
        """
        return self.model.value(env_input,train,init_h_value)
class GRUPPOAgent(parl.Agent):
    """ Agent of Mujoco env

    Args:
        algorithm (`parl.Algorithm`): algorithm to be used in this agent.
    """

    def __init__(self, algorithm):
        super(GRUPPOAgent, self).__init__(algorithm)

    def predict(self,env_input,hpc_input,train,init_h1,init_h2 ):
        """ Predict action from current policy given observation

        Args:
            obs (np.array): observation
        """


        action = self.alg.predict(env_input,hpc_input,train,init_h1,init_h2)

        return action.detach().numpy()

    def sample(self,env_input,hpc_input,train,init_h1,init_h2,init_h_value):
        """ Sample action from current policy given observation

        Args:
            obs (np.array): observation
        """
        value, action, action_log_probs = self.alg.sample(env_input,hpc_input,train,init_h1,init_h2,init_h_value)
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

    def value(self,env_input,train,init_h_value):
        """ Predict value from current value function given observation

        Args:
            obs (np.array): observation
        """

        val = self.alg.value(env_input,train,init_h_value)

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

class PPO_GRU_Module():
    def __init__(self,env_obs_dim,hpc_obs_dim,action_dim,gru1_to_gru2_vec_dim):
        # super(SACModule,self).__init__()
        # self.obs_dim=obs_dim
        self.action_dim=action_dim
        # self.model = MujocoModel(obs_dim, action_dim)
        self.model=GRU_PPO_Model(env_obs_dim,hpc_obs_dim,action_dim,gru1_to_gru2_vec_dim)
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
        # self.rpm = RolloutStorage(hp.PPO_NUM_STEPS, img_shape,obs_dim,action_dim,gru_hid_dim=hp.PPO_GRU_HID_DIM)
        # num_steps, img_shape,obs_dim, act_dim,gru_hid_dim
        self._reset()
        self.interact_steps=0
        self.learn_steps=0
        self.warm_up_random=True
    def _reset(self):
        self.agent.reset()
    def _input(self,env_input,hpc_input,train,init_h1,init_h2,init_h_value):
        """
        s，a，r的历史记录
        HPC离散编码激活程度值
        內嗅皮层的未来预测值
        """
        value, action, action_log_prob=self.agent.sample(env_input,hpc_input,False,None,None)
        actor_gru_h1,actor_gru_h2,critic_gru_h=self.agent.alg.model.actor_model.h1.numpy(),self.agent.alg.model.actor_model.h2.numpy(),self.agent.alg.model.critic_model.h.numpy()

        return np.squeeze(value),np.squeeze(action),np.squeeze(action_log_prob),actor_gru_h1,actor_gru_h2,critic_gru_h
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