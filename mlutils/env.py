import numpy as np
import cv2
import copy
from mlutils.common import *
red= np.array([255, 0, 0])[::-1]
green= np.array([0, 255, 0])[::-1]
blue=np.array([0, 0, 255])[::-1]
purple=np.array([112, 39, 195])[::-1]
yellow=np.array([255, 255, 0])[::-1]
grey=np.array([100, 100, 100])[::-1]

class EnvNeverDone():
    def __init__(self,env,obs_scale):
        self.obs_scale=obs_scale
        self.env=env
        self.action_space=env.action_space
        self.observation_space=env.observation_space

    def reset(self):
        print("start reset")
        obs=self.env.reset()
        return obs*self.obs_scale

    def step(self,action):
        obs,reward,done,info= self.env.step(action)
        if done:
            obs=self.env.reset()
        return obs*self.obs_scale,reward,False,info
    def render(self,mode):
        return self.env.render(mode=mode)

class Obj():
    def __init__(self,name,location:Location,color) -> None:
        """
        env的物体类
        
        """
        self.name=name
        self.color=color
        self.location=location
        self.alive=True
    def dead(self):
        self.alive=False
    def move(self,move_y,move_x):
        self.location.move(move_y,move_x)
    def in_obj(self,other_obj):
        return self.location.in_location(other_obj.location)
    def in_obj_list(self,obj_list):
        for obj in obj_list:
            if self.in_obj(obj):
                return True
        return False
    def color_equal(self,other_color):
        for c1,c2 in zip(self.color,other_color):
            if c1!=c2:
                return False
        return True
class State():
    def __init__(
        self,
        name,
        init_value=100,max_value=100,
        step_cost=-1,
        punish_thresh=60,punish_value=-1,#低于此值且高于warning，每一步受到惩罚；高于此值，不惩罚
        warning_thresh=15,warning_value=-3,#低于此值，每一步受到更大的惩罚；高于此值，不惩罚
        reward_thresh=90,reward_ratio=1,#低于此值，value每增加1受到奖励；高于此值代表饱和，不奖励
        ) -> None:
        """
        例如：punish_thresh=60,reward_thresh_90
        当value<=60，value每次更新将受到惩罚punishi_value
        当value>60，value每减少1将不会受到惩罚
        当value<=90，value每增加1将会得到奖励reward_value
        当value>90，属于吃饱状态，value增加不会得到奖励
        """
        self.name=name

        self.value=float(init_value)
        self.pre_value=self.value
        self.max_value=float(max_value)
        self.step_cost=step_cost

        self.punish_thresh=punish_thresh
        self.punish_value=punish_value

        self.warning_thresh=warning_thresh
        self.warning_value=warning_value

        self.reward_thresh=reward_thresh
        self.reward_ratio=reward_ratio

    @property
    def ratio(self):
        return self.value/self.max_value
    def __repr__(self):
        return f"{self.name} value={self.value}"
    def change_value(self,delta_v):
        self.pre_value=copy.deepcopy(self.value)
        self.value=np.clip(self.value+delta_v,0,self.max_value)
    def update(self,env_value_add,env_value_sub):
        "先加再减然后step,加后计算reward，减后计算punish"
        self.change_value(env_value_add)
        reward=self.reward()

        self.change_value(env_value_sub)
        self.change_value(self.step_cost)
        
        punish=self.punish()
        warninig=self.warning()
        return reward+punish+warninig

    def punish(self):
        if self.value<=self.punish_thresh and self.value>self.warning_thresh:
            return self.punish_value
        else:
            return 0
    def warning(self):
        if self.value<=self.warning_thresh:
            return self.warning_value
        else:
            return 0
    def reward(self):
        if self.pre_value<=self.reward_thresh:
            "如果增加后大于reward_thresh，则只奖励thresh之前的部分"
            return self.reward_ratio*min(self.value-self.pre_value,self.reward_thresh-self.pre_value)
        else:
            return 0






