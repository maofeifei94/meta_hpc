import numpy as np
import cv2
import copy

class FindBallEnv():
    def __init__(self) -> None:
        self.box_num=10
        self.max_steps=20
    def reset(self):
        self.ball_index=np.random.randint(0,self.box_num)
        self.check_index=0
        self._steps=0
        obs=self.get_obs(self.check_index)
        return obs
    def step(self,action):
        self._steps+=1
        self.check_index=action
        obs=self.get_obs(self.check_index)
        reward=self.get_reward(self.check_index)
        done=self.get_done()
        return obs,reward,done
        
    def get_obs(self,check_index):
        obs=np.zeros([self.box_num])
        obs[check_index]=1.0
        reward=self.get_reward(check_index)
        return [obs,np.eye(2,2)[int(reward)]]
    def get_reward(self,check_index):
        return float(self.ball_index==check_index)
    def get_done(self):
        return self._steps>=self.max_steps

class PredGoalEnv():
    def __init__(self) -> None:
        self.img_size=[300,300]
        self.frog_map_size=[300,300]
        self.step_size=0.1
        
        self.agent_sight_pixel=64
        self.agent_sight=float(self.agent_sight_pixel)/self.frog_map_size[0]
        
        
    def reset(self):
        self.agent_site=np.random.uniform(0,1,[2])
        self.goal_site=np.random.uniform(0,1,[2])

        "frog"
        self.frog_map=np.zeros([*self.frog_map_size,1])

        self.update_frog_map()
        obs=self.get_obs
        return obs
    def update_frog_map(self):
        agent_site_map=(self.agent_site*self.frog_map_size).astype(np.int)
        self.frog_map[
            max(agent_site_map[0]-self.agent_sight_pixel,0):min(agent_site_map[0]+self.agent_sight_pixel,self.frog_map_size[0]),
            max(agent_site_map[1]-self.agent_sight_pixel,0):min(agent_site_map[1]+self.agent_sight_pixel,self.frog_map_size[1])]=1.0
       
    def get_obs(self):
        see_goal=1.0 if np.all(np.abs(self.agent_site-self.goal_site)<self.agent_sight) else 0.0
        obs=[*self.agent_site,see_goal]
        return obs
    def step(self,action):
        self.agent_site=np.clip(self.agent_site+self.step_size*action,0,1)
        self.update_frog_map()
        obs=self.get_obs()
        return obs,0,False
        
    def render(self):
        frog_map_show=np.where(self.frog_map>0.5,0.0,0.2)
        agent_site_map=(self.agent_site*self.frog_map_size).astype(np.int)
        goal_site_map=(self.goal_site*self.frog_map_size).astype(np.int)
        cv2.rectangle(frog_map_show,(agent_site_map[1]-self.agent_sight_pixel,agent_site_map[0]-self.agent_sight_pixel),(agent_site_map[1]+self.agent_sight_pixel,agent_site_map[0]+self.agent_sight_pixel),(1.0),2)
        cv2.circle(frog_map_show,(agent_site_map[1],agent_site_map[0]),4,(0.7),-1)

        goal_size=6
        cv2.rectangle(frog_map_show,(goal_site_map[1]-goal_size,goal_site_map[0]-goal_size),(goal_site_map[1]+goal_size,goal_site_map[0]+goal_size),(0.5),-1)
        cv2.imshow("frog_map",frog_map_show)
        cv2.waitKey()


if __name__=="__main__":
    env=PredGoalEnv()
    env.reset()

    for i in range(100):
        obs,_,_=env.step(np.random.uniform(-1,1,[2]))
        print(obs)
        env.render()
        