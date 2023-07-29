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
class Food(Obj):
    def __init__(self, name, location: Location, color=green) -> None:
        super().__init__(name, location, color)
class Bed(Obj):
    def __init__(self, name, location: Location, color=yellow) -> None:
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
        self.reset_state()
    def __repr__(self):
        return f"{str(self.energy)}\n{str(self.bloodsugar)}"
    @property
    def state_vec(self):
        return [s.ratio for s in self.state_list]
    def reset_state(self):
        self.energy=State("energy")
        self.bloodsugar=State("bloodsugar")
        self.state_list=[self.energy,self.bloodsugar]
    def update_state(self,env_update_info):
        in_food=env_update_info['in_food']
        in_bed=env_update_info['in_bed']
        env_bloodsugar_add=10 if in_food else 0
        env_energy_add=10 if in_bed else 0
        
        reward=0
        reward+=self.bloodsugar.update(env_bloodsugar_add,0)
        reward+=self.energy.update(env_energy_add,0)
        return reward
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
    def render_food(self,img):
        return self.render_layer(img,self.food_list)
    def render_bed(self,img):
        return self.render_layer(img,self.bed_list)
    def render_agent(self,img):
        return self.render_layer(img,[self.agent])
class MiniGridLife(MiniGrid_base):
    def __init__(self ):
        
        self.action_dim_discrete=5
        
        # self.action_dim=int(np.log2(self.action_num-1))+1

        self.grid_size=[32,32]
        self.obs_range=[8,8]

        self.obs_dim=2+self.action_dim_discrete+np.sum(self.grid_size)
        self.max_ep_steps=400

    def reset(self):
        obs=self.inner_reset()
        return obs
    def inner_reset(self):
        self.build_wall()
        self.build_bed()
        self.build_food()
        self.build_agent()
        self.bg_layer=self.render_bg()
        self.step_num=0
        self.action=self.action_dim_discrete-1
        obs=self.get_obs()
        return obs

    def step(self,action):
        action=int(action)
        self.action=action

        "update"
        env_update_info={
            "action":action,
            "wall_list":self.wall_list
        }
        self.agent.update_location(env_update_info)
        obj_interact_info=self.get_obj_interact_info()
        env_update_info.update(obj_interact_info)
        self.agent_state_reward=self.agent.update_state(env_update_info)

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
        return obs,reward,False,info
    def get_obj_interact_info(self):
        in_food=self.agent.in_obj_list(self.food_list)
        in_bed=self.agent.in_obj_list(self.bed_list)
        interact_info={
            "in_food":in_food,
            "in_bed":in_bed,
        }
        return interact_info

    # def get_obs(self):
    #     self.full_map=self.render_agent(self.render_food(self.render_bed(copy.deepcopy(self.bg_layer))))
    #     self.sight_rect=self.get_sight_rect()
    #     self.sight_img=crop_img(self.full_map,self.sight_rect)
    #     sight_vec=np.array(self.sight_img,np.float32).reshape([-1])/255
    #     state_vec=self.agent.state_vec
    #     # print(state_vec)
    #     self.action_vec=np.eye(self.action_dim_discrete,self.action_dim_discrete)[self.action]
    #     obs_vec=np.concatenate([sight_vec,self.action_vec,state_vec])
    #     # print(obs_vec)
    #     return obs_vec
    def get_obs(self):
        self.full_map=self.render_agent(self.render_food(self.render_bed(copy.deepcopy(self.bg_layer))))
        self.sight_rect=self.get_sight_rect()
        self.sight_img=crop_img(self.full_map,self.sight_rect)
        sight_vec=np.array(self.sight_img,np.float32).reshape([-1])/255
        state_vec=self.agent.state_vec
        # print(state_vec)
        self.action_vec=np.eye(self.action_dim_discrete,self.action_dim_discrete)[self.action]

        location_vec_y=np.eye(self.grid_size[0],self.grid_size[0])[int(self.agent.location.y)]
        location_vec_x=np.eye(self.grid_size[1],self.grid_size[1])[int(self.agent.location.x)]
        location_vec=np.reshape([location_vec_y,location_vec_x],[-1])
        # print(location_vec,self.agent.location)

        obs_vec=np.concatenate([self.action_vec,location_vec,state_vec])
        # print(obs_vec)
        return obs_vec
    def inner_done_func(self):
        inner_done=self.step_num>=self.max_ep_steps
        return inner_done
    def reward_func(self):
        reward=self.agent_state_reward*0.1
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
            Wall("wall0",Rect(0,0,30,0)),
            Wall("wall1",Rect(31,0,31,30)),
            Wall("wall2",Rect(1,31,31,31)),
            Wall("wall3",Rect(0,1,0,31)),
            ]
    def build_bed(self):
        self.bed_list=[
            Bed("bed0",Rect(14,14,17,17))
        ]
    def build_food(self):
        food_rand_location_list=[
            Rect(1,1,3,3),
            Rect(1,28,3,30),
            Rect(28,1,30,3),
            Rect(28,28,30,30)
        ]
        # food_location_index=np.random.randint(0,len(food_rand_location_list))
        food_location_index=0
        food_location=food_rand_location_list[food_location_index]
        self.food_list=[
            Food("food0",food_location)
        ]

    def build_agent(self):
        self.agent=Agent("agent0",Point(15,15),self.grid_size)
        # print(self.agent.location)
        # input()


if __name__=="__main__":
    env=MiniGridLife()
    env.reset()
    while 1:
        env.step(np.random.randint(0,4))
        env.render()

