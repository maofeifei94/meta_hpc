import numpy as np
import cv2
import copy
import random
# from mlutils.img import crop_img,rect
# from mlutils.ml import array_in

class FindBall():
    def __init__(self) -> None:
        self.box_num=10
        self.ball_num=1
        self.obs_dim=self.box_num+2
        self.action_num=self.box_num
        self.action_dim=int(np.log2(self.action_num-1))+1

    def reset(self):  
        self.inner_reset()
        self.action=0
        # obs=self.inner_reset()
        obs,_,_,_=self.step(self.action)
        return obs
    def inner_reset(self):
        "ball index"
        self.ball_index_list=list(range(self.box_num))
        random.shuffle(self.ball_index_list)
        self.ball_index_list=self.ball_index_list[:self.ball_num]
        self.open_state=np.zeros([self.box_num],np.int)
        

    def render(self):
        pass

    def get_obs(self):
        "[action,contain_ball?,finish?]"
        action_vec=np.eye(self.box_num,self.box_num)[self.action]
        contain_ball=float(self.action in self.ball_index_list)
        finish=contain_ball
        return [*action_vec,contain_ball,finish]
    def reward_func(self):
        return float(self.action in self.ball_index_list)
    def inner_reward_func(self,action):
        if self.open_state[action]==1:
            inner_r=0
        else:
            inner_r=1/(self.box_num-np.sum(self.open_state))
        inner_r=0.1 if inner_r>0 else 0
        return inner_r
    
    def done_func(self):
        return self.action in self.ball_index_list
    def step(self,action):
        self.action=action

        obs=self.get_obs()
        reward=self.reward_func()
        inner_reward=self.inner_reward_func(action)
        inner_done=self.done_func()

        self.open_state[action]=1

        if inner_done:
            self.inner_reset()

        

        info={
            "out_reward":reward,
            "inner_reward":inner_reward,
        }
        
        return obs,reward+inner_reward,False,info

if __name__=="__main__":
    e=FindBall()
    e.reset()
    while 1:
        action=np.random.randint(0,3)
        print(e.ball_index_list,action)
        obs,reward,done=e.step(action)
        print(obs,reward,done)


