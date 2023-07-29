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
from mlutils.multiprocess import state_dict_to_np,set_state_dict_from_np
from mlutils.ml import ModelSaver
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

"ppo_gru_hybrid"
class Agent(parl.Agent,ModelSaver):
    """ Agent of Mujoco env

    Args:
        algorithm (`parl.Algorithm`): algorithm to be used in this agent.
    """

    def __init__(self, algorithm,input_output_info):
        super().__init__(algorithm)
        self.hp=input_output_info['hyperparam']
        # print(self.hp)


    # def predict(self,input_dict,train,h_dict):
    #     """ Predict action from current policy given observation

    #     Args:
    #         obs (np.array): observation
    #     """

    #     action = self.alg.predict(input_dict,train,h_dict)

    #     return action.detach().numpy()
    def interact(self,input_dict,train,h_dict):
        """
        若train=True,代表训练模式,需要指定h,h_dict!=None
        若train=False,代表与环境交互模式,h_dict=None,actor/critic使用self.h
        """
        sample_dict=self.sample(input_dict,train,h_dict)
        actor_gru_h,critic_gru_h=self.alg.model.actor_model.h.numpy(),self.alg.model.critic_model.h.numpy()

        output_dict={
            "value":np.squeeze(sample_dict['value']),

            "action_discrete":np.squeeze(sample_dict['action_discrete']),
            "action_discrete_log_probs":np.squeeze(sample_dict['action_discrete_log_probs']),
            "action_continuous":np.squeeze(sample_dict['action_continuous']),
            "action_continuous_log_probs":np.squeeze(sample_dict['action_continuous_log_probs']),
            "actor_gru_h":actor_gru_h,
            "critic_gru_h":critic_gru_h
        }
        return output_dict
    def test(self,input_dict,train,h_dict):
        sample_dict=self.predict(input_dict,train,h_dict)
        actor_gru_h,critic_gru_h=self.alg.model.actor_model.h.numpy(),self.alg.model.critic_model.h.numpy()

        output_dict={
            "value":np.squeeze(sample_dict['value']),

            "action_discrete":np.squeeze(sample_dict['action_discrete']),
            "action_discrete_log_probs":np.squeeze(sample_dict['action_discrete_log_probs']),
            "action_continuous":np.squeeze(sample_dict['action_continuous']),
            "action_continuous_log_probs":np.squeeze(sample_dict['action_continuous_log_probs']),
            "actor_gru_h":actor_gru_h,
            "critic_gru_h":critic_gru_h
        }
        return output_dict
    def predict(self,input_dict,train,h_dict):
        """ Sample action from current policy given observation

        Args:
            obs (np.array): observation
        """
        sample_dict= self.alg.predict(input_dict,train,h_dict)
        # for key in sample_dict.keys():
        #     print(key)
        #     print(sample_dict[key].numpy())
        return {key:sample_dict[key].numpy() for key in sample_dict.keys()}
    def sample(self,input_dict,train,h_dict):
        """ Sample action from current policy given observation

        Args:
            obs (np.array): observation
        """
        sample_dict= self.alg.sample(input_dict,train,h_dict)
        return {key:sample_dict[key].numpy() for key in sample_dict.keys()}

    def reset(self):
        self.alg.model.actor_model.reset()
        self.alg.model.critic_model.reset()


    def learn(self, data_buffer_list):

        """ Learn current batch of rollout for ppo_epoch epochs.

        Args:
            next_value (np.array): next predicted value for calculating advantage
            gamma (float): the discounting factor
            gae_lambda (float): lambda for calculating n step return
            ppo_epoch (int): number of epochs K
            rollouts (RolloutStorage): the rollout storage that contains the current rollout
        """

        value_loss_epoch = 0
        action_discrete_loss_epoch = 0
        action_discrete_dist_entropy_epoch = 0
        action_continuous_loss_epoch = 0
        action_continuous_dist_entropy_epoch = 0
        value_clip_mean_ratio_epoch=0

       

        t=0


        "cumpute gae return"
        for data_buffer in data_buffer_list:
            next_value=self.value(
                {'env_vec':to_time_paddle(data_buffer.env_vec[-1],'vec',1,1)},
                True,
                {'critic_init_h':to_time_paddle(data_buffer.critic_gru_h[-1],'vec',1,1)}
                )
            data_buffer.compute_returns(next_value,self.hp.PPO_GAMMA,self.hp.PPO_GAE_LAMBDA,)

            
        
        for e in range(self.hp.PPO_EPOCH):
            for b in range(self.hp.PPO_BATCH_NUM):
                sample_dict_list=[]
                for s in range(self.hp.PPO_BATCH_SIZE):
                    buffer_index=s%len(data_buffer_list)
                    sample_dict_list.append(data_buffer_list[buffer_index].sample_batch())
            

                sample_dict_list_for_train=sample_dict_list





                train_dict={
                    "indices":[sd['indices'] for sd in sample_dict_list_for_train],
                    "train_control":to_time_paddle([sd['train_control'] for sd in sample_dict_list_for_train],'vec',self.hp.PPO_BATCH_SIZE,-1),
                    "actor_init_h":to_time_paddle([sd['actor_init_h'] for sd in sample_dict_list_for_train],'vec',1,self.hp.PPO_BATCH_SIZE),
                    "critic_init_h":to_time_paddle([sd['critic_init_h'] for sd in sample_dict_list_for_train],'vec',1,self.hp.PPO_BATCH_SIZE),
                    "env_vec":to_time_paddle([sd['env_vec'] for sd in sample_dict_list_for_train],'vec',self.hp.PPO_BATCH_SIZE,-1),
                    ###  hybrid  ###
                    "actions_discrete_batch":to_time_paddle([sd['actions_discrete_batch'] for sd in sample_dict_list_for_train],'vec',self.hp.PPO_BATCH_SIZE,-1),
                    "actions_continuous_batch":to_time_paddle([sd['actions_continuous_batch'] for sd in sample_dict_list_for_train],'vec',self.hp.PPO_BATCH_SIZE,-1),
                    "old_action_discrete_log_probs_batch":to_time_paddle([sd['old_action_discrete_log_probs_batch'] for sd in sample_dict_list_for_train],'vec',self.hp.PPO_BATCH_SIZE,-1),
                    "old_action_continuous_log_probs_batch":to_time_paddle([sd['old_action_continuous_log_probs_batch'] for sd in sample_dict_list_for_train],'vec',self.hp.PPO_BATCH_SIZE,-1),
                    "value_preds_batch":to_time_paddle([sd['value_preds_batch'] for sd in sample_dict_list_for_train],'vec',self.hp.PPO_BATCH_SIZE,-1),
                    "returns_batch":to_time_paddle([sd['returns_batch'] for sd in sample_dict_list_for_train],'vec',self.hp.PPO_BATCH_SIZE,-1),
                    "adv_targ":to_time_paddle([sd['adv_targ'] for sd in sample_dict_list_for_train],'vec',self.hp.PPO_BATCH_SIZE,-1),
                }

                if t%self.hp.PPO_ACTOR_TRAIN_INTERVAL==0:
                    action_continuous_loss, action_continuous_dist_entropy,action_discrete_loss,action_discrete_dist_entropy=self.alg.learn_actor(train_dict)
                    # while 1:
                    #     action_continuous_loss, action_continuous_dist_entropy,action_discrete_loss,action_discrete_dist_entropy=self.alg.learn_actor(train_dict)
                    #     print(action_continuous_loss)
                    action_continuous_loss_epoch +=action_continuous_loss
                    action_continuous_dist_entropy_epoch +=action_continuous_dist_entropy
                    action_discrete_loss_epoch +=action_discrete_loss
                    action_discrete_dist_entropy_epoch +=action_discrete_dist_entropy
                if t%self.hp.PPO_CRITIC_TRAIN_INTERVAL==0:

                    value_loss,value_clip_mean_ratio=self.alg.learn_critic(train_dict)
                    # print("train loss=",value_loss,value_clip_mean_ratio)
                    # while 1:
                    #     value_loss,value_clip_mean_ratio=self.alg.learn_critic(train_dict)
                    #     print("still train loss=",value_loss,value_clip_mean_ratio)
                    value_loss_epoch += value_loss
                    value_clip_mean_ratio_epoch+=value_clip_mean_ratio

                t+=1
                # value_loss, action_continuous_loss, action_continuous_dist_entropy ,action_discrete_loss,action_discrete_dist_entropy,value_clip_mean_ratio= self.alg.learn(train_dict)

                # while 1:
                #     value_loss, action_continuous_loss, action_continuous_dist_entropy ,action_discrete_loss,action_discrete_dist_entropy,value_clip_mean_ratio= self.alg.learn(train_dict)
                #     print("still train loss=",value_loss, action_continuous_loss, action_continuous_dist_entropy ,action_discrete_loss,action_discrete_dist_entropy,value_clip_mean_ratio)


        num_updates = self.hp.PPO_EPOCH * self.hp.PPO_BATCH_NUM

        action_continuous_loss_epoch /= (num_updates/self.hp.PPO_ACTOR_TRAIN_INTERVAL)
        action_continuous_dist_entropy_epoch /= (num_updates/self.hp.PPO_ACTOR_TRAIN_INTERVAL)
        action_discrete_loss_epoch /= (num_updates/self.hp.PPO_ACTOR_TRAIN_INTERVAL)
        action_discrete_dist_entropy_epoch /= (num_updates/self.hp.PPO_ACTOR_TRAIN_INTERVAL)

        value_loss_epoch /= (num_updates/self.hp.PPO_CRITIC_TRAIN_INTERVAL)
        value_clip_mean_ratio_epoch/=(num_updates/self.hp.PPO_CRITIC_TRAIN_INTERVAL)

        return value_loss_epoch,action_continuous_loss_epoch,action_continuous_dist_entropy_epoch,action_discrete_loss_epoch, action_discrete_dist_entropy_epoch,value_clip_mean_ratio_epoch

    def value(self,input_dict,train,h_dict):
        """ Predict value from current value function given observation

        Args:
            obs (np.array): observation
        """

        val = self.alg.value(input_dict,train,h_dict)

        return val.detach().numpy()
    def save_model(self,save_dir,iter_num):
        "调用ModelSaver类的方法"
        super().save_model(model=self.alg.model,save_dir=save_dir,iter_num=iter_num)
    def load_model(self,save_dir,iter_num):
        "调用ModelSaver类的方法"
        super().load_model(model=self.alg.model,save_dir=save_dir,iter_num=iter_num)
    def update_model(self,state_dict):
        "调用ModelSaver类的方法"
        super().update_model(model=self.alg.model,state_dict=state_dict)
    def update_model_from_np(self,state_dict_np):
        "调用ModelSaver类的方法"
        super().update_model_from_np(model=self.alg.model,state_dict_np=state_dict_np)
    def send_model_to_np(self):
        "调用ModelSaver类的方法"
        return super().send_model_to_np(model=self.alg.model)
