
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


def wlog(x):
    pass
    logging.info(x)
def nl_func():
    # return nn.LeakyReLU(negative_slope=0.1)
    return nn.Tanh()
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

"""
input_output_info={
    'actor_gru_hid_dim'
    'critic_gru_hid_dim'
    'obs_dim'
    'action_env_dim'
    'to_static'
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
    'env_vec'
    'action_env'
    'action_hpc'
}
h_dict={
    'actor_init_h1'
    'actor_init_h2'
    'critic_init_h'
}
"""
class m_Categorical(Categorical):
    def entropy(self):
        dist_sum = paddle.sum(self.logits,axis=-1, keepdim=True)
        prob = self.logits / dist_sum
        # prob=paddle.clip(prob,1e-4,1)
        neg_entropy = paddle.sum(
            prob * paddle.log(prob),axis=-1, keepdim=True)
        entropy = -neg_entropy
        return entropy
        
class pfc_actor(parl.Model):
    def __init__(self, input_output_info):
        super(pfc_actor, self).__init__()
        self.gru_hid_dim=input_output_info['actor_gru_hid_dim']
        self.obs_dim=input_output_info['obs_dim']
        # self.action_env_dim=input_output_info['action_env_discrete_dim']+input_output_info['action_env_continuous_dim']
        self.action_env_discrete_dim=input_output_info['action_env_discrete_dim']
        self.action_env_continuous_dim=input_output_info['action_env_continuous_dim']
        "从obs到gru"
        self.obs_to_gru=nn.Sequential(
            nn.Linear(self.obs_dim,128),nl_func(),
            nn.Linear(128,128),nl_func(),
            nn.Linear(128,128),nl_func(),
            nn.Linear(128,128),nl_func(),
            nn.Linear(128,128),nl_func(),
            )

        self.gru=nn.GRU(input_size=128,hidden_size=self.gru_hid_dim)

        
        "从gru到env_action"
        self.gru_to_env_discrete_action=nn.Sequential(
            nn.Linear(self.gru_hid_dim,128),nl_func(),
            nn.Linear(128,128),nl_func(),
            nn.Linear(128,128),nl_func(),
            nn.Linear(128,self.action_env_discrete_dim)
            )
        self.gru_to_env_continuous_action=nn.Sequential(
            nn.Linear(self.gru_hid_dim,128),nl_func(),
            nn.Linear(128,128),nl_func(),
            nn.Linear(128,128),nl_func(),
            nn.Linear(128,self.action_env_continuous_dim*2)
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

        action_continuous_std=F.sigmoid(action_continuous_std*hp.PPO_ACTOR_CONTINUOUS_SCALE_LOGITS)*hp.PPO_ACTOR_CONTINUOUS_SCALE_STD+hp.PPO_ACTOR_CONTINUOUS_MIN_STD
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
            self.h=h

        return action_dict



class pfc_critic(parl.Model):
    def __init__(self, input_output_info):
        # print("gru_ppo init_param=",img_shape,obs_dim,int(img_shape[2]*img_shape[1]/8**2)+obs_dim)
        super(pfc_critic, self).__init__()
        self.gru_hid_dim=input_output_info['critic_gru_hid_dim']
        self.obs_dim=input_output_info['obs_dim']


        "从obs到gru"
        self.obs_to_gru=nn.Sequential(
            nn.Linear(self.obs_dim,128),nl_func(),
            nn.Linear(128,128),nl_func(),
            nn.Linear(128,128),nl_func(),
            nn.Linear(128,128),nl_func(),
            nn.Linear(128,128),nl_func(),
            )

        self.gru=nn.GRU(input_size=128,hidden_size=self.gru_hid_dim)

        
        "从gru到reward"
        self.gru_to_value=nn.Sequential(
            nn.Linear(self.gru_hid_dim,128),nl_func(),
            nn.Linear(128,128),nl_func(),
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
            self.h=h


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
    def __init__(self,model:GRU_PPO_Model):
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
        # self.clip_param = clip_param

        # self.value_loss_coef = value_loss_coef
        # self.entropy_coef = entropy_coef
        # self.use_clipped_value_loss = use_clipped_value_loss
        clip = nn.ClipGradByNorm(hp.PPO_MAX_GRAD_NROM)

        self.optimizer_a = optim.Adam(
            parameters=model.actor_model.parameters(),
            learning_rate=hp.PPO_ACTOR_LR,
            epsilon=hp.PPO_EPS,
            grad_clip=clip)
        # clip_c=nn.ClipGradByGlobalNorm(1e-8)
        self.optimizer_c = optim.Adam(
            learning_rate=hp.PPO_CRITIC_LR,
            # beta1=0.5,
            parameters=model.critic_model.parameters(),
            epsilon=hp.PPO_EPS,
            grad_clip=clip)
        # self.optimizer_c=optim.Momentum(
        #     learning_rate=hp.PPO_LR_C,
        #     parameters=model.critic_model.parameters()
        # )
        # self.optimizer_c = optim.RMSProp(
        #     learning_rate=hp.PPO_LR_C,
        #     # beta1=0.9,
        #     parameters=model.critic_model.parameters(),
        #     epsilon=hp.PPO_EPS,
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

        action_discrete_dist=m_Categorical(logits=action_discrete_probs)
        action_discrete_dim=action_discrete_probs.shape[-1]
        action_discrete_onehot=paddle.squeeze(F.one_hot(old_action_discrete,action_discrete_dim),axis=-2)
        # print(old_action_discrete.shape,F.one_hot(old_action_discrete,action_discrete_dim).shape,action_discrete_onehot.shape)
        action_discrete_log_probs=paddle.sum(action_discrete_probs*action_discrete_onehot,axis=-1,keepdim=True)
        # print(action_discrete_probs[0,:10],action_discrete_onehot[0,:10])
        # print(action_discrete_log_probs[0,:10])

        action_discrete_dist_entropy=paddle.mean(action_discrete_dist.entropy().sum(axis=-1,keepdim=True)*train_control)
        # print(action_discrete_dist_entropy)
        action_discrete_ratio=paddle.exp(action_discrete_log_probs-old_action_discrete_log_probs_batch)
        # print(action_discrete_log_probs[0,:10],old_action_discrete_log_probs_batch[0,:10])
        # print(paddle.max(action_discrete_ratio*train_control),paddle.min(action_discrete_ratio*train_control))

        action_discrete_surr1=action_discrete_ratio*adv_targ
        action_discrete_surr2=paddle.clip(action_discrete_ratio,1.0-hp.PPO_ACTOR_DISCRETE_CLIP_PARAM,1.0+hp.PPO_ACTOR_DISCRETE_CLIP_PARAM)*adv_targ

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
        action_continuous_surr2=paddle.clip(action_continuous_ratio,1.0-hp.PPO_ACTOR_CONTINUOUS_CLIP_PARAM,1.0+hp.PPO_ACTOR_CONTINUOUS_CLIP_PARAM)*adv_targ
        action_continuous_loss=-(paddle.minimum(action_continuous_surr1,action_continuous_surr2)*train_control).mean()

        actor_loss=(action_continuous_loss-action_continuous_dist_entropy*hp.PPO_ACTOR_CONTINUOUS_ENTROPY_COEF)*hp.PPO_ACTOR_CONTINUOUS_LOSS_RATIO
        actor_loss+=(action_discrete_loss-action_discrete_dist_entropy*hp.PPO_ACTOR_DISCRETE_ENTROPY_COEF)*hp.PPO_ACTOR_DISCRETE_LOSS_RATIO


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
        if hp.PPO_CRITIC_USE_CLIPPED_VALUE_LOSS:
            value_delta=values-train_dict['value_preds_batch']
            value_delta_target=train_dict['returns_batch'] - train_dict['value_preds_batch']
            value_delta_target_non_zero=value_delta_target+(1-paddle.abs(paddle.sign(value_delta_target)))*1e-12
            value_update_ratio=value_delta/value_delta_target_non_zero

            value_delta_target_mean_size=paddle.sum(paddle.abs(value_delta_target)*train_control,axis=[0,1,2],keepdim=True)/paddle.sum(train_control).detach()
            # value_loss=(paddle.abs((value_delta-value_delta_target*hp.PPO_CLIPPED_VALUE_RATIO).pow(2)/value_delta_target_non_zero)*train_control).mean()
            # value_loss=((value_update_ratio-hp.PPO_CLIPPED_VALUE_RATIO)**2*train_control).mean()
            value_loss=(0.5 * (value_delta - value_delta_target*hp.PPO_CRITIC_CLIPPED_VALUE_RATIO).pow(2)*train_dict['train_control']).mean()
            # print('train_dict["advantages_min_size"]',train_dict["advantages_min_size"])

            # value_loss=((value_delta-value_delta_target*hp.PPO_CLIPPED_VALUE_RATIO).pow(2)/paddle.clip(paddle.abs(value_delta_target_non_zero),0.001,1e12)*train_control).mean()
            # value_loss*=float(train_dict["advantages_min_size"])

            # value_loss=(paddle.abs(value_delta-value_delta_target*hp.PPO_CLIPPED_VALUE_RATIO).pow(2)/paddle.clip(paddle.abs(value_delta_target_non_zero),0.01,1e12).pow(0.5)*train_control).mean()
            value_clip_mean_ratio=paddle.sum(paddle.abs(value_update_ratio-hp.PPO_CRITIC_CLIPPED_VALUE_RATIO)*train_control)/paddle.sum(train_control)
            
            adv_log_mean=train_dict["adv_log_mean"]
            # value_delta_target_log_mean=paddle.sum(paddle.log(paddle.where(train_control>0.5,paddle.abs(value_delta_target_non_zero),paddle.ones_like(values))))/paddle.sum(train_control)
            # value_loss=(
            #     paddle.abs(value_delta-value_delta_target*hp.PPO_CLIPPED_VALUE_RATIO).pow(2)\
            #     # /paddle.clip(paddle.abs(value_delta_target_non_zero),value_delta_target_log_mean.exp()*0.1,1e12)\
            #     /paddle.clip(paddle.abs(value_delta_target_non_zero),float(np.exp(adv_log_mean))*0.01,1e12)\
            #     *float(np.exp(adv_log_mean))*train_control
            #     ).mean()
            # value_loss=(
            #     paddle.abs(value_delta-value_delta_target*hp.PPO_CLIPPED_VALUE_RATIO).pow(3)\
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
        action_discrete_dist = m_Categorical(logits=action_discrete_probs)
        action_discrete = action_discrete_dist.sample([1])
        act_discrete_dim = action_discrete_probs.shape[-1]
        action_discrete_onehot = paddle.squeeze(F.one_hot(action_discrete, act_discrete_dim),axis=-2)
        action_discrete_log_probs=paddle.sum(action_discrete_probs*action_discrete_onehot,axis=-1,keepdim=True)

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
class GRUPPOAgent(parl.Agent):
    """ Agent of Mujoco env

    Args:
        algorithm (`parl.Algorithm`): algorithm to be used in this agent.
    """

    def __init__(self, algorithm:GRUPPO,input_output_info):
        super(GRUPPOAgent, self).__init__(algorithm)


    # def predict(self,input_dict,train,h_dict):
    #     """ Predict action from current policy given observation

    #     Args:
    #         obs (np.array): observation
    #     """

    #     action = self.alg.predict(input_dict,train,h_dict)

    #     return action.detach().numpy()

    def sample(self,input_dict,train,h_dict):
        """ Sample action from current policy given observation

        Args:
            obs (np.array): observation
        """
        sample_dict= self.alg.sample(input_dict,train,h_dict)
        # for key in sample_dict.keys():
        #     print(key)
        #     print(sample_dict[key].numpy())
        return {key:sample_dict[key].numpy() for key in sample_dict.keys()}
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
        action_discrete_loss_epoch = 0
        action_discrete_dist_entropy_epoch = 0
        action_continuous_loss_epoch = 0
        action_continuous_dist_entropy_epoch = 0
        value_clip_mean_ratio_epoch=0

        # clip = nn.ClipGradByNorm(hp.PPO_MAX_GRAD_NROM)
        # self.alg.optimizer_c = optim.rmsprop(
        #     learning_rate=hp.PPO_LR_C,
        #     # beta1=0.9,
        #     parameters=self.alg.model.critic_model.parameters(),
        #     epsilon=hp.PPO_EPS,
        #     grad_clip=clip)

        [_d for _d in rollouts.sample_batch(next_value, gamma,gae_lambda, hp.PPO_BATCH_SIZE,1)]


        t=0
        for e in range(ppo_epoch):
            # print(f"ppo_epoch {e}/{ppo_epoch}")

            
            data_generator = rollouts.sample_batch(next_value, gamma,gae_lambda, hp.PPO_BATCH_SIZE,hp.PPO_BATCH_NUM)

            advantages=np.abs(rollouts.returns[:-1]-rollouts.value_preds[:-1])
            adv_log_mean=np.mean(np.log(advantages))
            # print(np.min(advantages),np.min(np.abs(rollouts.returns[:-1]-rollouts.value_preds[:-1])),np.mean(np.abs(advantages)),np.max(np.abs(advantages)))
            sample_dict_list=[]

            _g=0
            for sample_dict in data_generator:
                advantages=np.maximum(0.001,np.abs(rollouts.returns[:-1]-rollouts.value_preds[:-1]))
                # print(np.min(advantages),np.min(np.abs(rollouts.returns[:-1]-rollouts.value_preds[:-1])),np.mean(np.abs(advantages)),np.max(np.abs(advantages)))
                # input()
                # ori_train_dict={key:paddle.to_tensor(sample_dict[key]) for key in sample_dict.keys()}
                # print([[key,sample_dict[key].shape] for key in sample_dict.keys()])
                # for key in sample_dict.keys():
                #     print([key,sample_dict[key].shape])
                # input()
                
                sample_dict_list.append(sample_dict)
                if len(sample_dict_list)>=hp.PPO_BATCH_SIZE:
                    sample_dict_list_for_train=sample_dict_list
                    sample_dict_list=[]
                else:
                    continue

                train_dict={
                    "indices":[sd['indices'] for sd in sample_dict_list_for_train],
                    "train_control":to_time_paddle([sd['train_control'] for sd in sample_dict_list_for_train],'vec',hp.PPO_BATCH_SIZE,-1),

                    "actor_init_h":to_time_paddle([sd['actor_init_h'] for sd in sample_dict_list_for_train],'vec',1,hp.PPO_BATCH_SIZE),

                    "critic_init_h":to_time_paddle([sd['critic_init_h'] for sd in sample_dict_list_for_train],'vec',1,hp.PPO_BATCH_SIZE),


                    "env_vec":to_time_paddle([sd['env_vec'] for sd in sample_dict_list_for_train],'vec',hp.PPO_BATCH_SIZE,-1),

                    # "model_probs_batch":to_time_paddle([sd['model_probs_batch'] for sd in sample_dict_list_for_train],'vec',hp.PPO_BATCH_SIZE,-1),

                    ###  hybrid  ###
                    "actions_discrete_batch":to_time_paddle([sd['actions_discrete_batch'] for sd in sample_dict_list_for_train],'vec',hp.PPO_BATCH_SIZE,-1),
                    "actions_continuous_batch":to_time_paddle([sd['actions_continuous_batch'] for sd in sample_dict_list_for_train],'vec',hp.PPO_BATCH_SIZE,-1),
                    "old_action_discrete_log_probs_batch":to_time_paddle([sd['old_action_discrete_log_probs_batch'] for sd in sample_dict_list_for_train],'vec',hp.PPO_BATCH_SIZE,-1),
                    "old_action_continuous_log_probs_batch":to_time_paddle([sd['old_action_continuous_log_probs_batch'] for sd in sample_dict_list_for_train],'vec',hp.PPO_BATCH_SIZE,-1),

                    "value_preds_batch":to_time_paddle([sd['value_preds_batch'] for sd in sample_dict_list_for_train],'vec',hp.PPO_BATCH_SIZE,-1),
                    "returns_batch":to_time_paddle([sd['returns_batch'] for sd in sample_dict_list_for_train],'vec',hp.PPO_BATCH_SIZE,-1),
                    "adv_targ":to_time_paddle([sd['adv_targ'] for sd in sample_dict_list_for_train],'vec',hp.PPO_BATCH_SIZE,-1),
                    # "advantages_min_size":np.min(np.abs(advantages)),
                    # "advantages_mean_size":np.mean(np.abs(advantages)),
                    "adv_log_mean":adv_log_mean
                }

                # print([train_dict[td].shape for td in train_dict.keys()])
                # wlog(f"|epoch={e}|batchindex={t}")
                # wlog(f"|indices{train_dict['indices']}")

                if t%hp.PPO_ACTOR_TRAIN_RATE==0:
                    action_continuous_loss, action_continuous_dist_entropy,action_discrete_loss,action_discrete_dist_entropy=self.alg.learn_actor(train_dict)
                    # while 1:
                    #     action_continuous_loss, action_continuous_dist_entropy,action_discrete_loss,action_discrete_dist_entropy=self.alg.learn_actor(train_dict)
                    #     print(action_continuous_loss)
                    action_continuous_loss_epoch +=action_continuous_loss
                    action_continuous_dist_entropy_epoch +=action_continuous_dist_entropy
                    action_discrete_loss_epoch +=action_discrete_loss
                    action_discrete_dist_entropy_epoch +=action_discrete_dist_entropy
                if t%hp.PPO_CRITIC_TRAIN_RATE==0:

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


        num_updates = ppo_epoch * hp.PPO_BATCH_NUM

        action_continuous_loss_epoch /= (num_updates/hp.PPO_ACTOR_TRAIN_RATE)
        action_continuous_dist_entropy_epoch /= (num_updates/hp.PPO_ACTOR_TRAIN_RATE)
        action_discrete_loss_epoch /= (num_updates/hp.PPO_ACTOR_TRAIN_RATE)
        action_discrete_dist_entropy_epoch /= (num_updates/hp.PPO_ACTOR_TRAIN_RATE)

        value_loss_epoch /= (num_updates/hp.PPO_CRITIC_TRAIN_RATE)
        value_clip_mean_ratio_epoch/=(num_updates/hp.PPO_CRITIC_TRAIN_RATE)

        return value_loss_epoch,action_continuous_loss_epoch,action_continuous_dist_entropy_epoch,action_discrete_loss_epoch, action_discrete_dist_entropy_epoch,value_clip_mean_ratio_epoch

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
        # self.action_dim=input_output_info['action_dim']


        # self.env_img=np.zeros((num_steps + 1, *input_output_info['env_img_shape']), dtype='float32')
        self.env_vec = np.zeros((num_steps + 1, input_output_info['env_vec_dim']), dtype='float32')
        # self.hpc_img=np.zeros((num_steps + 1, *input_output_info['hpc_img_shape']), dtype='float32')
        # self.hpc_vec = np.zeros((num_steps + 1, input_output_info['hpc_vec_dim']), dtype='float32')

        # self.model_probs=np.zeros((num_steps, input_output_info['action_dim']), dtype='float32')

        ###  hybrid  ###
        self.actions_discrete = np.zeros((num_steps, 1), dtype='int64')
        self.action_discrete_log_probs = np.zeros((num_steps, ), dtype='float32')
        self.actions_continuous = np.zeros((num_steps, input_output_info['action_env_continuous_dim']), dtype='float32')
        self.action_continuous_log_probs = np.zeros((num_steps, ), dtype='float32')

        self.value_preds = np.zeros((num_steps+1, ), dtype='float32')
        self.returns = np.zeros((num_steps + 1, ), dtype='float32')
        
        self.rewards = np.zeros((num_steps, ), dtype='float32')

        self.actor_gru_h=np.zeros((num_steps+1,input_output_info['actor_gru_hid_dim']),dtype='float32')
        # self.actor_gru2_h=np.zeros((num_steps,input_output_info['actor_gru2_hid_dim']),dtype='float32')
        self.critic_gru_h=np.zeros((num_steps+1,input_output_info['critic_gru_hid_dim']),dtype='float32')

        self.masks = np.ones((num_steps + 1, ), dtype='bool')
        self.bad_masks = np.ones((num_steps + 1, ), dtype='bool')

        self.step = 0

    def append(self, collect_dict):
        # self.env_img[self.step + 1] = collect_dict['env_img']
        self.env_vec[self.step + 1] = collect_dict['env_vec']
        # self.hpc_img[self.step + 1] = collect_dict['hpc_img']
        # self.hpc_vec[self.step + 1] = collect_dict['hpc_vec']
        
        # self.model_probs[self.step] = collect_dict['model_probs']

        ###  hybrid  ###
        self.actions_discrete[self.step] = collect_dict['action_discrete']
        self.action_discrete_log_probs[self.step] = collect_dict['action_discrete_log_prob']
        self.actions_continuous[self.step] = collect_dict['action_continuous']
        self.action_continuous_log_probs[self.step] = collect_dict['action_continuous_log_prob']


        self.value_preds[self.step] = collect_dict['value_pred']

        self.rewards[self.step] = collect_dict['reward']

        # self.actor_gru1_h[self.step]=collect_dict['actor_gru1_h']
        self.actor_gru_h[self.step+1]=collect_dict['actor_gru_h']
        self.critic_gru_h[self.step+1]=collect_dict['critic_gru_h']

        self.masks[self.step + 1] = collect_dict['mask']
        self.bad_masks[self.step + 1] = collect_dict['bad_mask']
        
        self.step = (self.step + 1) % self.num_steps
        # print(f"roll out storage step={self.step}")
        # print("step:",self.step)
    def after_update(self):
        # self.env_img[0]=np.copy(self.env_img[-1])
        self.env_vec[0]=np.copy(self.env_vec[-1])
        # self.hpc_img[0]=np.copy(self.hpc_img[-1])
        # self.hpc_vec[0]=np.copy(self.hpc_vec[-1])

        self.masks[0] = np.copy(self.masks[-1])
        self.bad_masks[0] = np.copy(self.bad_masks[-1])

        self.actor_gru_h[0]=np.copy(self.actor_gru_h[-1])
        self.critic_gru_h[0]=np.copy(self.critic_gru_h[-1])

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
        # train_start_idx=128
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
                     mini_batch_size):
        # calculate return and advantage first
        self.compute_returns(next_value, gamma, gae_lambda)
        # print("compute_returns")
        advantages = self.returns[:-1] - self.value_preds[:-1]

        # print("advantage",advantages.shape,advantages.mean().shape,advantages.std().shape)
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-4)
        
        # generate sample batch
        # mini_batch_size = self.num_steps // num_mini_batch
        # sampler = BatchSampler(
        #     sampler=RandomSampler(range(self.num_steps)),
        #     batch_size=mini_batch_size,
        #     drop_last=True)
        # print(self.num_steps,num_mini_batch)
        # wlog(f"|mean_self.action_log_probs={np.mean(self.action_log_probs)}")
        for i in range(num_mini_batch*mini_batch_size):
            
            indices,train_control=self.gru_indices()
            
            train_control=self._pad_vec(train_control,self.train_history_len*2)
            actor_init_h=self.actor_gru_h[indices.start]
            actor_init_h=np.reshape(actor_init_h,[1,1,-1])
            # actor_init_h2=np.zeros_like(self.actor_gru2_h[0]) if indices.start==0 else self.actor_gru2_h[indices.start-1]
            # actor_init_h2=np.reshape(actor_init_h2,[1,1,-1])
            critic_init_h=self.critic_gru_h[indices.start]
            critic_init_h=np.reshape(critic_init_h,[1,1,-1])
            # indices=slice(0,self.num_steps)
            # train_control=np.ones([self.num_steps])
            # env_img_batch=self._pad_vec(self.env_img[:-1][indices],self.train_history_len*2)
            env_vec_batch=self._pad_vec(self.env_vec[:-1][indices],self.train_history_len*2)
            # hpc_img_batch=self._pad_vec(self.hpc_img[:-1][indices],self.train_history_len*2)
            # hpc_vec_batch=self._pad_vec(self.hpc_vec[:-1][indices],self.train_history_len*2)

            ###  hybrid  ###
            actions_discrete_batch=self._pad_vec(self.actions_discrete[indices],self.train_history_len*2)
            actions_continuous_batch=self._pad_vec(self.actions_continuous[indices],self.train_history_len*2)

            old_action_discrete_log_probs_batch = self._pad_vec(self.action_discrete_log_probs[indices],self.train_history_len*2)
            old_action_continuous_log_probs_batch = self._pad_vec(self.action_continuous_log_probs[indices],self.train_history_len*2)

            # model_probs_batch = self._pad_vec(self.model_probs[indices],self.train_history_len*2)
            
            value_preds_batch = self._pad_vec(self.value_preds[:-1][indices],self.train_history_len*2)
            returns_batch = self._pad_vec(self.returns[:-1][indices],self.train_history_len*2)
            adv_targ = self._pad_vec(advantages[indices],self.train_history_len*2)

            "reshape"
            value_preds_batch = value_preds_batch.reshape(-1, 1)
            returns_batch = returns_batch.reshape(-1, 1)
            # model_probs_batch=model_probs_batch.reshape(-1,self.action_dim)
            old_action_discrete_log_probs_batch=old_action_discrete_log_probs_batch.reshape(-1, 1)
            old_action_continuous_log_probs_batch=old_action_continuous_log_probs_batch.reshape(-1, 1)

            adv_targ = adv_targ.reshape(-1, 1)

            sample_dict={
                "indices":[indices.start,indices.stop],
                "train_control":train_control,
                "actor_init_h":actor_init_h,
                # "actor_init_h2":actor_init_h2,
            
                "critic_init_h":critic_init_h,

                # "env_img":env_img_batch,
                "env_vec":env_vec_batch,
                # "hpc_img":hpc_img_batch,
                # "hpc_vec":hpc_vec_batch,
                # "model_probs_batch":model_probs_batch,

                "actions_discrete_batch":actions_discrete_batch,
                "old_action_discrete_log_probs_batch":old_action_discrete_log_probs_batch,

                "actions_continuous_batch":actions_continuous_batch,
                "old_action_continuous_log_probs_batch":old_action_continuous_log_probs_batch,

                "value_preds_batch":value_preds_batch,
                "returns_batch":returns_batch,
                "adv_targ":adv_targ,
            }
            yield sample_dict



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
        self.algorithm = GRUPPO(model=self.model)
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
        sample_dict=self.agent.sample(input_dict,train,h_dict)
        actor_gru_h,critic_gru_h=self.agent.alg.model.actor_model.h.numpy(),self.agent.alg.model.critic_model.h.numpy()
        # print(action.shape)
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
    def _rpm_collect(self,collect_dict):
        # obs=self.obs_to_std(obs)
        # next_obs=self.obs_to_std(next_obs)
        self.rpm.append(collect_dict)
        # self.rpm.append(obs, action, reward, next_obs, terminal)
    
    def _learn(self):
        next_value = self.agent.value({
            # 'env_img':to_time_paddle(self.rpm.env_img[-1],'img',1,1),
            'env_vec':to_time_paddle(self.rpm.env_vec[-1],'vec',1,1),
            # 'hpc_img':to_time_paddle(self.rpm.hpc_img[-1],'img',1,1),
            # 'hpc_vec':to_time_paddle(self.rpm.hpc_vec[-1],'vec',1,1),
            },True,
            {
                'critic_init_h':to_time_paddle(self.rpm.critic_gru_h[-1],'vec',1,1),
            }
            )
        value_loss_epoch, action_continuous_loss_epoch,action_continuous_dist_entropy_epoch ,action_discrete_loss_epoch, action_discrete_dist_entropy_epoch,value_clip_mean_ratio_epoch= self.agent.learn(
            next_value, hp.PPO_GAMMA, hp.PPO_GAE_LAMBDA, hp.PPO_EPOCH, hp.PPO_BATCH_SIZE, self.rpm)
        print(f"v_loss{value_loss_epoch},a_discrete_loss{action_discrete_loss_epoch},discrete_d_entropy{action_discrete_dist_entropy_epoch},a_continuous_loss{action_continuous_loss_epoch},continuous_d_entropy{action_continuous_dist_entropy_epoch},value_clip_mean_ratio{value_clip_mean_ratio_epoch}")
        self.rpm.after_update()
        return value_loss_epoch, action_continuous_loss_epoch,action_continuous_dist_entropy_epoch,action_discrete_loss_epoch, action_discrete_dist_entropy_epoch,value_clip_mean_ratio_epoch
    def save_model(self,save_dir,iter_num):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"start_save_model {save_dir}/{iter_num}_model.pdparams")
        paddle.save(self.model.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")
        print(f"save model {save_dir}/{iter_num}_model.pdparams success")
        
    def load_model(self,save_dir,iter_num):
        print(f"start load_model {save_dir}/{iter_num}_model.pdparams")
        params_state = paddle.load(path=f"{save_dir}/{iter_num}_model.pdparams")
        self.model.set_state_dict(params_state)
        print(f"load model {save_dir}/{iter_num}_model.pdparams success")

    def update(self,state_dict):
        self.model.set_state_dict(state_dict)
        print("update param success")