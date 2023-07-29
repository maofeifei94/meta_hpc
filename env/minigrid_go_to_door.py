from urllib.parse import non_hierarchical
import numpy as np
import cv2
import copy
from mlutils.img import crop_img,Location,Point,Rect,location_in
from mlutils.ml import array_in,one_hot
from mlutils.env import Obj,State,red,green,yellow,grey



class Wall(Obj):
    def __init__(self, name, location: Location, color=grey) -> None:
        super().__init__(name, location, color)
class Door(Obj):
    def __init__(self, name, location: Location, color=grey) -> None:
        super().__init__(name, location, color)
class Signal(Obj):
    def __init__(self, name, location: Location, color=grey) -> None:
        super().__init__(name, location, color)



class Agent(Obj):
    def __init__(self, name,location,grid_size,color=red) -> None:
        if not isinstance(location,Point):
            raise ValueError(f"Agent location should be Point but receive {type(location)}")
        super().__init__(name, location, color)
        self.action_moving_mapping={
            0:[1,0],
            1:[0,1],
            2:[-1,0],
            3:[0,-1],
            4:[0,0]
        }
        self.grid_size=np.array(grid_size)

    def update_location(self,env_update_info):
        """
        action
        0:direction-1
        1:direction+1
        """
        action=env_update_info["action"]
        wall_list=env_update_info["wall_list"]

        moving_vec=self.action_moving_mapping[action]
        pre_location=copy.deepcopy(self.location)
        next_location=self.location.move(*moving_vec)

        if self.in_obj_list(wall_list):
            self.location=pre_location
        else:
            self.location=next_location

class MiniGrid_base():
    def __init__(self) -> None:
        pass
    def get_sight_rect(self):
        loc=self.agent.location
        return Rect(loc.y-3,loc.x-3,loc.y+4,loc.x+4)
    def render_layer(self,img,obj_list):
        for obj in obj_list:
            loc=obj.location
            # print(loc,loc.slice_miny,loc.slice_maxy,loc.slice_minx,loc.slice_maxx)
            img[loc.slice_miny:loc.slice_maxy,loc.slice_minx:loc.slice_maxx]=obj.color
        return img
    def render_bg(self):
        init_img=np.full([*self.grid_size,3],30,np.uint8)
        self.bg_layer=self.render_layer(init_img,self.wall_list)
        return self.bg_layer
    # def render_food(self,img):
    #     return self.render_layer(img,self.food_list)
    # def render_bed(self,img):
    #     return self.render_layer(img,self.bed_list)
    def render_signal(self,img):
        return self.render_layer(img,self.signal_list)        
    def render_door(self,img):
        return self.render_layer(img,self.door_list)
    def render_agent(self,img):
        return self.render_layer(img,[self.agent])
class MiniGridGoToDoor(MiniGrid_base):
    def __init__(self ):
        
        self.action_dim_discrete=5
        
        # self.action_dim=int(np.log2(self.action_num-1))+1

        self.grid_size=[16,16] #y,x
        self.obs_range=[8,8]

        self.obs_dim=int(np.prod(self.obs_range)*3)
        self.max_ep_steps=200

    def reset(self):
        obs=self.inner_reset()
        return obs
    def inner_reset(self):
        "build obj"
        self.build_wall()
        self.build_door()
        self.build_signal()
        self.build_agent()

        "bg layer"
        self.bg_layer=self.render_bg()
        self.step_num=0
        self.action=self.action_dim_discrete-1
        obs=self.get_obs()
        return obs

    def step(self,action):
        action=int(action)
        self.action=action

        "move"
        env_update_info={
            "action":action,
            "wall_list":self.wall_list
        }
        self.agent.update_location(env_update_info)
        "contact obj"
        self.obj_interact_info=self.get_obj_interact_info()


        self.step_num+=1
        # print(self.agent)
        # print("agent_state_reward",agent_state_reward)

        "obs"
        obs=self.get_obs()

        "reward"
        reward=self.reward_func()

        inner_done=self.inner_done_func()
        if inner_done:
            obs=self.inner_reset()
        info={
            "inner_reward":0,
            "out_reward":reward,
        }


        # print(reward,self.obj_interact_info)
        return np.reshape(obs,[-1]).astype(np.float32)/255,reward,False,info
    def get_obj_interact_info(self):
        in_door0=self.agent.in_obj(self.door_list[0])
        in_door1=self.agent.in_obj(self.door_list[1])
        interact_info={
            "in_door0":in_door0,
            "in_door1":in_door1,
        }
        return interact_info

    def get_obs(self):
        "render agent"
        self.full_map=self.render_agent(self.render_signal(self.render_door(copy.deepcopy(self.bg_layer))))
        "crop agent sight img"
        self.sight_rect=self.get_sight_rect()
        self.sight_img=crop_img(self.full_map,self.sight_rect)

        return self.sight_img
    def inner_done_func(self):
        if self.step_num>=self.max_ep_steps:
            inner_done=True
        elif self.obj_interact_info['in_door0'] or self.obj_interact_info['in_door1']:
            inner_done=True
        else:
            inner_done=False
        return inner_done
    def reward_func(self):
        if self.obj_interact_info['in_door0']:
            if self.door_list[0].color_equal(self.signal_list[0].color):
                reward=1.0
            else:
                reward=-1.0
        elif self.obj_interact_info['in_door1']:
            if self.door_list[1].color_equal(self.signal_list[0].color):
                reward=1.0
            else:
                reward=-1.0
        elif self.step_num>=self.max_ep_steps:
            reward=-1.0
        else:
            reward=0
        return reward
    # def inner_reward_func(self,obs,action):
    #     vec_for_kl_pred=obs[:8*8*3]
    #     action_for_kl_pred=one_hot(action,3)
    #     real_kl,pred_kl=self.kl_pred_net.pred(vec_for_kl_pred,action_for_kl_pred)
    #     reward=(real_kl+pred_kl)*0.02
    #     self.ep_inner_pred_reward+=pred_kl*0.02
    #     self.ep_inner_real_reward+=real_kl*0.02
    #     return reward
    def render(self,show=True):
        scale=15
        # def draw_agent(agent,img):
        #     direction_vec=np.array(agent.direction_mapping[agent.direction])
        #     left_hand_vec=np.array(agent.direction_mapping[(agent.direction+1)%4])
        #     center=agent.site+0.5
        #     head_point=((center+direction_vec*0.5)*scale).astype(np.int)
        #     left_point=((center-direction_vec*0.5+left_hand_vec*0.5)*scale).astype(np.int)
        #     right_point=((center-direction_vec*0.5-left_hand_vec*0.5)*scale).astype(np.int)
        #     # print([tuple(head_point[::-1]),tuple(left_point[::-1])])
        #     pts=np.array([head_point[::-1],left_point[::-1],right_point[::-1]]).reshape([1,3,2])
        #     # print(tuple(agent.color))
        #     # print(pts,np.shape(img))
        #     tuple([int(c) for c in agent.color])
        #     cv2.fillPoly(img,pts,(0,0,255))
        #     return img
        def draw_sight(sight_rect,img,scale):
            cv2.rectangle(img,
            (int(sight_rect.minx*scale),int(sight_rect.miny*scale)),
            (int((sight_rect.maxx+1)*scale-1),int((sight_rect.maxy+1)*scale-1)),
            (0,0,255),
            2
            )
            return img
        render_img=copy.deepcopy(self.full_map)
        # render_img=self.render_layer(self.grid_size,[self.agent.site],self.agent.color,render_img)
        render_img=cv2.resize(render_img,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        render_img=draw_sight(self.sight_rect,render_img,scale)

        render_obs=cv2.resize(self.sight_img,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        if show:
            cv2.imshow("MiniGridLife",render_img)
            cv2.imshow("obs",render_obs)
            # cv2.imshow
            # cv2.imshow("obs",cv2.resize(self.obs,None,fx=5,fy=5,interpolation=cv2.INTER_AREA))
            cv2.waitKey()
        return render_img

    def build_wall(self):
        self.wall_list=[
            Wall("wall0",Rect(0,0,self.grid_size[0]-2,0)),
            Wall("wall1",Rect(self.grid_size[0]-1,0,self.grid_size[0]-1,self.grid_size[1]-2)),
            Wall("wall2",Rect(1,self.grid_size[1]-1,self.grid_size[0]-1,self.grid_size[1]-1)),
            Wall("wall3",Rect(0,1,0,self.grid_size[1]-1)),
            ]
    def build_signal(self):
        rand_x=np.random.randint(1,self.grid_size[1]-2)
        rand_y=np.random.randint(2,self.grid_size[0]-2)
        rand_color=[green,yellow][np.random.randint(0,2)]
        self.signal_list=[
            Signal("signal0",Point(rand_y,rand_x),rand_color)
        ]
    def build_door(self):
        self.door_list=[
            Door("door0",Rect(1,1,1,self.grid_size[1]//2-1),green),
            Door("door1",Rect(1,self.grid_size[1]//2,1,self.grid_size[1]-2),yellow)
        ]


    def build_agent(self):
        self.agent=Agent("agent0",Point(8,8),self.grid_size)
        # print(self.agent.location)
        # input()


if __name__=="__main__":
    env=MiniGridGoToDoor()
    env.reset()
    while 1:
        env.step(np.random.randint(0,4))
        env.render()

