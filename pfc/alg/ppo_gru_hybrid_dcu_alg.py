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
from hyperparam import Hybrid_PPO_HyperParam as hp
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

"paddle的原始categorical类entropy方法有问题"
class m_Categorical():
    def __init__(self,probs):
        # probs_sum=paddle.sum(probs,axis=-1,keepdim=True)
        # if not paddle.all(probs_sum==1.0):
        #     raise ValueError("Categorical distribution probs sum != 1")
        self.probs=probs
    def sample(self,sample_num):
        assert(isinstance(sample_num,list))
        "probs_thresh"
        category_num=self.probs.shape[-1]
        thresh_add_mat=paddle.to_tensor(np.triu(np.ones([category_num,category_num],np.float32)),'float32')
        probs_flat=paddle.reshape(self.probs,[-1,category_num])

        probs_sum_upper_thresh=paddle.matmul(probs_flat,thresh_add_mat)
        probs_sum_lower_thresh=paddle.concat([probs_sum_upper_thresh[:,-1:]-1,probs_sum_upper_thresh[:,:-1]],axis=-1)

        # print("upper",probs_sum_upper_thresh,"lower",probs_sum_lower_thresh)


        sample_shape=sample_num+probs_flat.shape
        sample_shape[-1]=1
        sample_rand=paddle.rand(sample_shape,'float32')
        result=paddle.cast(sample_rand>=probs_sum_lower_thresh,'float32')*paddle.cast(sample_rand<probs_sum_upper_thresh,'float32')
        sample_index=paddle.argmax(result,axis=-1,keepdim=True)

        sample_index_reshape=paddle.reshape(sample_index,sample_num+self.probs.shape[:-1])
        
        # print(sample_rand.shape)
        # print(sample_rand>=probs_sum_lower_thresh.shape)
        # print(sample_rand<probs_sum_upper_thresh.shape)
        # print(result.shape)
        # print(sample_index.shape)
        return sample_index_reshape
    def entropy(self):
        dist_sum = paddle.sum(self.probs,axis=-1, keepdim=True)
        prob = self.probs / dist_sum
        # prob=paddle.clip(prob,1e-4,1)
        neg_entropy = paddle.sum(
            prob * paddle.log(prob),axis=-1, keepdim=True)
        entropy = -neg_entropy
        return entropy

class PPO_GRU_HYBRID(parl.Algorithm):
    def __init__(self,model,input_output_info):
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
        self.hp=input_output_info['hyperparam']
        # self.clip_param = clip_param

        # self.value_loss_coef = value_loss_coef
        # self.entropy_coef = entropy_coef
        # self.use_clipped_value_loss = use_clipped_value_loss
        clip = nn.ClipGradByNorm(self.hp.PPO_MAX_GRAD_NROM)

        self.optimizer_a = optim.Adam(
            parameters=model.actor_model.parameters(),
            learning_rate=self.hp.PPO_ACTOR_LR,
            epsilon=self.hp.PPO_EPS,
            grad_clip=clip)
        # clip_c=nn.ClipGradByGlobalNorm(1e-8)
        self.optimizer_c = optim.Adam(
            learning_rate=self.hp.PPO_CRITIC_LR,
            # beta1=0.5,
            parameters=model.critic_model.parameters(),
            epsilon=self.hp.PPO_EPS,
            grad_clip=clip)
        # self.optimizer_c=optim.Momentum(
        #     learning_rate=self.hp.PPO_LR_C,
        #     parameters=model.critic_model.parameters()
        # )
        # self.optimizer_c = optim.RMSProp(
        #     learning_rate=self.hp.PPO_LR_C,
        #     # beta1=0.9,
        #     parameters=model.critic_model.parameters(),
        #     epsilon=self.hp.PPO_EPS,
        #     grad_clip=clip)

    def learn_actor(self,train_dict):
        def sprint(x,name):
            pass
            # print(name,x.shape)
        value_dict = self.model.value(train_dict,train=True,h_dict=train_dict)
        values=value_dict['value']
        action_dict = self.model.policy(train_dict,train=True,h_dict=train_dict)
        train_control=train_dict['train_control']
        adv_targ=train_dict['adv_targ']

        "discrete loss"
        action_discrete_probs=action_dict["action_discrete_probs"]
        old_action_discrete=train_dict["actions_discrete_batch"]
        old_action_discrete_log_probs_batch=train_dict['old_action_discrete_log_probs_batch']

        action_discrete_dist=m_Categorical(probs=action_discrete_probs)
        action_discrete_dim=action_discrete_probs.shape[-1]
        action_discrete_onehot=paddle.squeeze(F.one_hot(old_action_discrete,action_discrete_dim),axis=-2)
        # print(old_action_discrete.shape,F.one_hot(old_action_discrete,action_discrete_dim).shape,action_discrete_onehot.shape)
        action_discrete_log_probs=paddle.log(paddle.sum(action_discrete_probs*action_discrete_onehot,axis=-1,keepdim=True))
        # print(action_discrete_probs[0,:10],action_discrete_onehot[0,:10])
        # print(action_discrete_log_probs[0,:10])

        action_discrete_dist_entropy=paddle.mean(action_discrete_dist.entropy().sum(axis=-1,keepdim=True)*train_control)
        # print(action_discrete_dist_entropy)
        action_discrete_ratio=paddle.exp(action_discrete_log_probs-old_action_discrete_log_probs_batch)
        # print(action_discrete_log_probs[0,:10],old_action_discrete_log_probs_batch[0,:10])
        # print(paddle.max(action_discrete_ratio*train_control),paddle.min(action_discrete_ratio*train_control))

        action_discrete_surr1=action_discrete_ratio*adv_targ
        action_discrete_surr2=paddle.clip(action_discrete_ratio,1.0-self.hp.PPO_ACTOR_DISCRETE_CLIP_PARAM,1.0+self.hp.PPO_ACTOR_DISCRETE_CLIP_PARAM)*adv_targ

        action_discrete_loss=-(paddle.minimum(action_discrete_surr1, action_discrete_surr2)*train_control).mean()

        "continuous loss"
        action_continuous_mean=action_dict["action_continuous_mean"]
        action_continuous_logstd=action_dict["action_continuous_logstd"]
        old_action_continuous=train_dict["actions_continuous_batch"]
        old_action_continuous_log_probs_batch=train_dict['old_action_continuous_log_probs_batch']

        action_continuous_dist=Normal(action_continuous_mean,action_continuous_logstd.exp())
        action_continuous_log_probs=action_continuous_dist.log_prob(old_action_continuous).sum(axis=-1,keepdim=True)
        # print(action_continuous_log_probs.shape,old_action_continuous_log_probs_batch.shape)
        # print(paddle.mean(action_continuous_dist.scale))
        # print((action_continuous_dist.entropy()-0.5*np.log(2*np.pi*np.exp(1))).sum(axis=-1,keepdim=True).mean())
        # print(paddle.sum((action_continuous_dist.entropy()-0.5*np.log(2*np.pi*np.exp(1))).mean(axis=-1,keepdim=True)*train_control)/paddle.sum(train_control))
        action_continuous_dist_entropy=paddle.mean((action_continuous_dist.entropy()-0.5*np.log(2*np.pi*np.exp(1))).sum(axis=-1,keepdim=True)*train_control)
        action_continuous_entropy_for_show=paddle.sum((action_continuous_dist.entropy()-0.5*np.log(2*np.pi*np.exp(1))).mean(axis=-1,keepdim=True)*train_control)/paddle.sum(train_control)
        action_continuous_ratio=paddle.exp(action_continuous_log_probs-old_action_continuous_log_probs_batch)
        action_continuous_surr1=action_continuous_ratio*adv_targ
        action_continuous_surr2=paddle.clip(action_continuous_ratio,1.0-self.hp.PPO_ACTOR_CONTINUOUS_CLIP_PARAM,1.0+self.hp.PPO_ACTOR_CONTINUOUS_CLIP_PARAM)*adv_targ
        action_continuous_loss=-(paddle.minimum(action_continuous_surr1,action_continuous_surr2)*train_control).mean()

        actor_loss=(action_continuous_loss-action_continuous_dist_entropy*self.hp.PPO_ACTOR_CONTINUOUS_ENTROPY_COEF)*self.hp.PPO_ACTOR_CONTINUOUS_LOSS_RATIO
        actor_loss+=(action_discrete_loss-action_discrete_dist_entropy*self.hp.PPO_ACTOR_DISCRETE_ENTROPY_COEF)*self.hp.PPO_ACTOR_DISCRETE_LOSS_RATIO


        self.optimizer_a.clear_grad()
        actor_loss.backward()
        self.optimizer_a.step()

        return action_continuous_loss.numpy(), action_continuous_entropy_for_show.numpy(),action_discrete_loss.numpy(),action_discrete_dist_entropy.numpy()

    def learn_critic(self,train_dict):
        value_dict = self.model.value(train_dict,train=True,h_dict=train_dict)
        values=value_dict['value']
        action_dict = self.model.policy(train_dict,train=True,h_dict=train_dict)
        train_control=train_dict['train_control']
        adv_targ=train_dict['adv_targ']
        if self.hp.PPO_CRITIC_USE_CLIPPED_VALUE_LOSS:
            value_delta=values-train_dict['value_preds_batch']
            value_delta_target=train_dict['returns_batch'] - train_dict['value_preds_batch']
            value_delta_target_non_zero=value_delta_target+(1-paddle.abs(paddle.sign(value_delta_target)))*1e-12
            value_update_ratio=value_delta/value_delta_target_non_zero

            value_delta_target_mean_size=paddle.sum(paddle.abs(value_delta_target)*train_control,axis=[0,1,2],keepdim=True)/paddle.sum(train_control).detach()
            # value_loss=(paddle.abs((value_delta-value_delta_target*self.hp.PPO_CLIPPED_VALUE_RATIO).pow(2)/value_delta_target_non_zero)*train_control).mean()
            # value_loss=((value_update_ratio-self.hp.PPO_CLIPPED_VALUE_RATIO)**2*train_control).mean()
            value_loss=(0.5 * (value_delta - value_delta_target*self.hp.PPO_CRITIC_CLIPPED_VALUE_RATIO).pow(2)*train_dict['train_control']).mean()
            # print('train_dict["advantages_min_size"]',train_dict["advantages_min_size"])

            # value_loss=((value_delta-value_delta_target*self.hp.PPO_CLIPPED_VALUE_RATIO).pow(2)/paddle.clip(paddle.abs(value_delta_target_non_zero),0.001,1e12)*train_control).mean()
            # value_loss*=float(train_dict["advantages_min_size"])

            # value_loss=(paddle.abs(value_delta-value_delta_target*self.hp.PPO_CLIPPED_VALUE_RATIO).pow(2)/paddle.clip(paddle.abs(value_delta_target_non_zero),0.01,1e12).pow(0.5)*train_control).mean()
            value_clip_mean_ratio=paddle.sum(paddle.abs(value_update_ratio-self.hp.PPO_CRITIC_CLIPPED_VALUE_RATIO)*train_control)/paddle.sum(train_control)
            
            # adv_log_mean=train_dict["adv_log_mean"]
            # value_delta_target_log_mean=paddle.sum(paddle.log(paddle.where(train_control>0.5,paddle.abs(value_delta_target_non_zero),paddle.ones_like(values))))/paddle.sum(train_control)
            # value_loss=(
            #     paddle.abs(value_delta-value_delta_target*self.hp.PPO_CLIPPED_VALUE_RATIO).pow(2)\
            #     # /paddle.clip(paddle.abs(value_delta_target_non_zero),value_delta_target_log_mean.exp()*0.1,1e12)\
            #     /paddle.clip(paddle.abs(value_delta_target_non_zero),float(np.exp(adv_log_mean))*0.01,1e12)\
            #     *float(np.exp(adv_log_mean))*train_control
            #     ).mean()
            # value_loss=(
            #     paddle.abs(value_delta-value_delta_target*self.hp.PPO_CLIPPED_VALUE_RATIO).pow(3)\
            #     # /value_delta_target_non_zero
            #     # /paddle.clip(paddle.abs(value_delta_target_non_zero),value_delta_target_log_mean.exp()*0.1,1e12)\
            #     /paddle.clip(paddle.abs(value_delta_target_non_zero),float(np.exp(adv_log_mean))*0.01,1e12)\
            #     *float(np.exp(adv_log_mean))\
            #     *train_control
            #     ).mean()
            # print("train_loss=",value_loss.numpy().reshape([-1]),value_clip_mean_ratio.numpy().reshape([-1]),value_delta_target_mean_size.numpy().reshape([-1]))
        else:
            value_loss = (0.5 * (train_dict['returns_batch'] - values).pow(2)*train_dict['train_control']).mean()
            value_clip_mean_ratio=paddle.ones([1])
        critic_loss=value_loss

        self.optimizer_c.clear_grad()
        critic_loss.backward()
        self.optimizer_c.step()

        return value_loss.numpy(),value_clip_mean_ratio.numpy()
    def sample(self, input_dict,train,h_dict):
        """ Sample action from parameterized policy
        """
        # obs=np.reshape(obs,[])
        # tt=timer_tool("agent sample")
        value_dict = self.model.value(input_dict,train,h_dict)
        value=value_dict['value']

        action_dict= self.model.policy(input_dict,train,h_dict)

        "discrete action"
        action_discrete_probs=action_dict["action_discrete_probs"]
        # probs = action_dict['action_probs']
        action_discrete_dist = m_Categorical(probs=action_discrete_probs)
        action_discrete = action_discrete_dist.sample([1])
        act_discrete_dim = action_discrete_probs.shape[-1]
        action_discrete_onehot = paddle.squeeze(F.one_hot(action_discrete, act_discrete_dim),axis=-2)
        action_discrete_log_probs=paddle.log(paddle.sum(action_discrete_probs*action_discrete_onehot,axis=-1,keepdim=True))

        "continuous action"
        action_continuous_mean,action_continuous_logstd=action_dict['action_continuous_mean'],action_dict['action_continuous_logstd']
        action_continuous_dist=Normal(action_continuous_mean,action_continuous_logstd.exp())
        action_continuous=action_continuous_dist.sample([1])[0]
        action_continuous_log_probs=action_continuous_dist.log_prob(action_continuous).sum(axis=-1,keepdim=True)

        sample_dict={
            "value":value,
            "action_discrete":action_discrete,
            "action_discrete_log_probs":action_discrete_log_probs,
            "action_continuous":action_continuous,
            "action_continuous_log_probs":action_continuous_log_probs,
        }
        return sample_dict
    def predict(self, input_dict,train,h_dict):
        """ Sample action from parameterized policy
        """
        # obs=np.reshape(obs,[])
        # tt=timer_tool("agent sample")
        value_dict = self.model.value(input_dict,train,h_dict)
        value=value_dict['value']

        action_dict= self.model.policy(input_dict,train,h_dict)

        "discrete action"
        action_discrete_probs=action_dict["action_discrete_probs"]
        # probs = action_dict['action_probs']
        action_discrete_dist = m_Categorical(probs=action_discrete_probs)
        action_discrete = paddle.argmax(action_discrete_probs,axis=-1,keepdim=False)
        act_discrete_dim = action_discrete_probs.shape[-1]
        action_discrete_onehot = paddle.squeeze(F.one_hot(action_discrete, act_discrete_dim),axis=-2)
        action_discrete_log_probs=paddle.log(paddle.sum(action_discrete_probs*action_discrete_onehot,axis=-1,keepdim=True))

        "continuous action"
        action_continuous_mean,action_continuous_logstd=action_dict['action_continuous_mean'],action_dict['action_continuous_logstd']
        action_continuous_dist=Normal(action_continuous_mean,action_continuous_logstd.exp())
        action_continuous=action_continuous_mean

        action_continuous_log_probs=action_continuous_dist.log_prob(action_continuous).sum(axis=-1,keepdim=True)

        sample_dict={
            "value":value,
            "action_discrete":action_discrete,
            "action_discrete_log_probs":action_discrete_log_probs,
            "action_continuous":action_continuous,
            "action_continuous_log_probs":action_continuous_log_probs,
        }
        return sample_dict

    # def predict(self, input_dict,train,h_dict):
    #     """ Predict action from parameterized policy, action with maximum probability is selected as greedy action
    #     """
    #     action_dict= self.model.policy(input_dict,train,h_dict)
    #     probs = action_dict['action_probs']
    #     action = paddle.argmax(probs, 1)

    #     return action

    def value(self, input_dict,train,h_dict):
        """ Predict value from parameterized value function
        """
        value_dict = self.model.value(input_dict,train,h_dict)
        value=value_dict['value']
        return value