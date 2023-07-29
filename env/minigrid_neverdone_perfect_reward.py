import numpy as np
import cv2
import copy
from mlutils.img import crop_img,Rect
from mlutils.ml import array_in,one_hot

red= np.array([255, 0, 0])[::-1]
green= np.array([0, 255, 0])[::-1]
blue=np.array([0, 0, 255])[::-1]
purple=np.array([112, 39, 195])[::-1]
yellow=np.array([255, 255, 0])[::-1]
grey=np.array([100, 100, 100])[::-1]

class Obj():
    def __init__(self,name,color,site) -> None:
        self.name=name
        self.color=color
        self.site=np.array(site)
        self.alive=True
    def dead(self):
        self.alive=False
class Agent(Obj):
    def __init__(self, name, color, site,direction,grid_size) -> None:
        super().__init__(name, color, site)
        """
        direction
        方向向量
        0:h 1:w 2:-h 3:-w
        """
        """
            2
            |
        3——   ——1
            |
            0
        """
        self.direction_mapping={
            0:[1,0],
            1:[0,1],
            2:[-1,0],
            3:[0,-1]
        }
        """
        agent左手方向向量
        """

        self.direction=direction
        self.grid_size=np.array(grid_size)
    def update(self,action,wall_site_list):
        """
        action
        0:direction-1
        1:direction+1
        """
        if action in [0,1]:
            self.direction=self.direction+1 if action==1 else self.direction-1
            self.direction=self.direction%4
        else:
            next_site=self.site+self.direction_mapping[self.direction]
            if array_in(next_site ,wall_site_list):
                self.site=self.site
            else:
                self.site=np.clip(next_site,[0,0],self.grid_size-1)

class MiniGrid_base():
    def __init__(self) -> None:
        pass
    def get_sight_rect(self,agent:Agent):
        direction_vec=np.array(agent.direction_mapping[agent.direction])
        left_hand_vec=np.array(agent.direction_mapping[(agent.direction+1)%4])
        corner1=agent.site+left_hand_vec*4
        corner2=agent.site+direction_vec*7-left_hand_vec*3
        minyx=np.minimum(corner1,corner2)
        maxyx=np.maximum(corner1,corner2)
        return Rect(*minyx,*maxyx)
    def render_layer(self,img_size,site_list,color,img=None):
        img=np.zeros([*img_size,3],np.uint8) if img is None else img
        for site in site_list:
            img[site[0],site[1]]=color
        return img
class MiniGrid_fourroom_obs8x8(MiniGrid_base):
    def __init__(self ):
        self.obs_dim=8*8*3+1+256+256
        self.action_num=3
        self.action_dim_continue=int(np.log2(self.action_num-1))+1
        self.action_dim_discrete=self.action_num

        self.grid_size=[19,19]
        self.obs_range=[8,8]
        self.max_steps=100
        self.build_wall()

        from pfc.gru_pred_conv_8x8 import GruPredHid
        self.kl_pred_net=GruPredHid()
        self.kl_pred_net.load_model("all_models/minigrid_model",582000)
        self.kl_pred_net.eval()

    def reset(self):
        obs=self.inner_reset()
        return obs
    def inner_reset(self):
        self.door_site_list,self.wall_site_list,self.goal_site_list,self.agent=self.build_grid()
        self.bg_layer=self.render_background()
        self.step_num=0
        self.kl_pred_net.pred_reset()

        self.vis_map=np.zeros(self.grid_size,np.int64)
        self.pre_vis_map_ratio=0

        self.contain_goal=False
        self.ep_first_contain_goal=None
        self.ep_contain_goal=False

        self.obs=self.get_obs()

        self.ep_inner_real_reward=0
        self.ep_inner_pred_reward=0
        self.ep_inner_reward=0



        return self.add_pred_h_to_obs(self.obs)
    def add_pred_h_to_obs(self,obs):
        def to_std(x):
            return (x-np.mean(x))/(np.std(x)+1e-8)
        kl_pred_h=self.kl_pred_net.kl_pred_gru_h.numpy().reshape([-1])
        kl_encoder_h=self.kl_pred_net.encoder_gru_h.numpy().reshape([-1])

        return np.concatenate([obs,to_std(kl_pred_h),to_std(kl_encoder_h)])
        
    def step(self,action):
        "update"
        self.agent.update(action,self.wall_site_list)
        "obs"
        obs=self.get_obs()
        self.step_num+=1
        "reward,done"
        reward=self.reward_func()
        inner_reward=self.inner_reward_func()
        inner_done=self.done_func()
        self.ep_inner_reward+=inner_reward
        # print("ep_inner_reward=",self.ep_inner_reward)
        

        info={
            "out_reward":reward,
            "inner_reward":inner_reward,
            "inner_done":inner_done,
            "ep_inner_pred_reward":self.ep_inner_pred_reward,
            "ep_inner_real_reward":self.ep_inner_real_reward,
            "ep_inner_reward":self.ep_inner_reward,
            "contain_goal":self.contain_goal
        }
        if inner_done:
            obs=self.inner_reset()

        self.after_update()
        return obs if inner_done else self.add_pred_h_to_obs(obs),reward+inner_reward,False,info
    def after_update(self):
        self.pre_vis_map_ratio=self.get_vis_map_ratio()
    def get_vis_map_ratio(self):
        # print(np.sum(self.vis_map),np.sum(self.vis_map)/np.prod(self.grid_size))
        return np.sum(self.vis_map)/np.prod(self.grid_size)

    def get_obs(self):
        # self.full_img=self.render_layer(self.grid_size,[self.agent.site],red,copy.deepcopy(self.bg_layer))
        self.sight_rect=self.get_sight_rect(self.agent)
        # print(self.sight_rect,self.agent.site,self.agent.direction)
        self.sight_img=crop_img(self.bg_layer,self.sight_rect)

        miny=max(0,self.sight_rect.miny)
        maxy=min(self.grid_size[0],self.sight_rect.maxy+1)
        minx=max(0,self.sight_rect.minx)
        maxx=min(self.grid_size[1],self.sight_rect.maxx+1)
        self.vis_map[miny:maxy,minx:maxx]=1
        # print("contain yellow",np.any(np.all(self.sight_img==yellow,axis=-1)))
        if self.ep_first_contain_goal==True:
            self.ep_first_contain_goal=False
        if np.any(np.all(self.sight_img==yellow,axis=-1)):
            self.ep_contain_goal=True
            self.contain_goal=True
            if self.ep_first_contain_goal is None:
                self.ep_first_contain_goal=True
        else:
            self.contain_goal=False
        # self.sight_img=np.tile(np.reshape(self.sight_img,[8,1,8,1,3]),[1,8,1,8,1]).reshape([64,64,3])
        done=self.done_func()

        env_vec=np.transpose(self.sight_img,[2,0,1])
        env_vec=np.reshape(env_vec,[-1]).astype(np.float32)
        env_vec=env_vec/255

        return np.array([*env_vec,float(done)],np.float32)
    def done_func(self):
        if array_in(self.agent.site,self.goal_site_list):
            done=True
        else:
            done=True if self.step_num>=self.max_steps else False
        return done
    def reward_func(self):
        if array_in(self.agent.site,self.goal_site_list):
            reward=10
        else:
            reward=0
        return reward
    def inner_reward_func(self):
        # vec_for_kl_pred=obs[:8*8*3]
        # action_for_kl_pred=one_hot(action,3)
        # real_kl,pred_kl=self.kl_pred_net.pred(vec_for_kl_pred,action_for_kl_pred)
        # reward=(real_kl+pred_kl)*0.02

        # self.ep_inner_pred_reward+=pred_kl*0.02
        # self.ep_inner_real_reward+=real_kl*0.02
        # print(self.ep_first_contain_goal)
        if self.ep_first_contain_goal==True:
            contain_goal_r=1.0
            # print("first contain_goal")
        else:
            contain_goal_r=0.0
        return (self.get_vis_map_ratio()-self.pre_vis_map_ratio)*5+float(contain_goal_r)*1.0


    def render(self,show=True):
        scale=30
        """
            2
            |
        3——   ——1
            |
            0
        """
        def draw_agent(agent,img):
            direction_vec=np.array(agent.direction_mapping[agent.direction])
            left_hand_vec=np.array(agent.direction_mapping[(agent.direction+1)%4])
            center=agent.site+0.5
            head_point=((center+direction_vec*0.5)*scale).astype(np.int)
            left_point=((center-direction_vec*0.5+left_hand_vec*0.5)*scale).astype(np.int)
            right_point=((center-direction_vec*0.5-left_hand_vec*0.5)*scale).astype(np.int)
            # print([tuple(head_point[::-1]),tuple(left_point[::-1])])
            pts=np.array([head_point[::-1],left_point[::-1],right_point[::-1]]).reshape([1,3,2])
            # print(tuple(agent.color))
            # print(pts,np.shape(img))
            tuple([int(c) for c in agent.color])
            cv2.fillPoly(img,pts,(0,0,255))
            return img
        def draw_sight(sight_rect,img,scale):
            masks=np.zeros_like(img)
            cv2.rectangle(masks,
            (int(sight_rect.minx*scale),int(sight_rect.miny*scale)),
            (int((sight_rect.maxx+1)*scale-1),int((sight_rect.maxy+1)*scale-1)),
            (255,255,255),
            -1
            )
            img=np.where(masks>0,(np.clip(1.5*img,0,255)).astype(np.uint8),(0.6*img).astype(np.uint8))
            cv2.rectangle(img,
            (int(sight_rect.minx*scale),int(sight_rect.miny*scale)),
            (int((sight_rect.maxx+1)*scale-1),int((sight_rect.maxy+1)*scale-1)),
            (150,0,0),
            1
            )
            return img


        render_img=copy.deepcopy(self.bg_layer)
        # render_img=self.render_layer(self.grid_size,[self.agent.site],self.agent.color,render_img)
        render_img=cv2.resize(render_img,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        render_img=draw_sight(self.sight_rect,render_img,scale)
        render_img=draw_agent(self.agent,render_img)
        if show:
            cv2.imshow("MiniGrid_fourroom",render_img)
            # cv2.imshow("obs",cv2.resize(self.obs,None,fx=5,fy=5,interpolation=cv2.INTER_AREA))
            cv2.imshow("vis_map",cv2.resize(self.vis_map.astype(np.float32),None,fx=15,fy=15,interpolation=cv2.INTER_AREA))
            cv2.waitKey(100)
        return render_img
    def render_background(self):
        init_img=np.full([*self.grid_size,3],30,np.uint8)
        self.wall_layer=self.render_layer(self.grid_size,self.wall_site_list,grey,init_img)
        self.bg_layer=self.render_layer(self.grid_size,self.goal_site_list,yellow,self.wall_layer)
        return self.bg_layer
    def build_wall(self):
        "door and wall"
        self.door_site_list=[
            np.array([6,9]),
            np.array([15,9]),
            np.array([9,5]),
            np.array([9,14])]
        self.wall_site_list=[]
        
        for h in [0,9,18]:
            for w in range(0,19):
                site=np.array([h,w],)
                # print(([site]+[site])[0],site,self.wall_site_list+self. door_site_list)
                if not array_in(site, self.wall_site_list+self. door_site_list):
                    self.wall_site_list.append(site)
        for w in [0,9,18]:
            for h in range(0,19):
                site=np.array([h,w])
                # if not (site in self.wall_site_list+self. door_site_list):
                if not array_in(site, self.wall_site_list+self. door_site_list):
                    self.wall_site_list.append(site)

    def build_grid(self):
        "goal"
        goal_num=1
        self.goal_site_list=[]
        for _g in range(goal_num):
            while 1:
                goal_site=np.random.randint(0,19,[2])
                if not array_in(goal_site,self.wall_site_list+self.door_site_list+self.goal_site_list):
                    self.goal_site_list.append(goal_site)
                    break
        "agent"
        while 1:
            agent_site=np.random.randint(0,19,[2])
            if not array_in(agent_site,self.wall_site_list+self.door_site_list+self.goal_site_list):
                self.agent_site=agent_site
                break
        self.agent=Agent("agent",red,self.agent_site,np.random.randint(0,4),self.grid_size)
        return self.door_site_list,self.wall_site_list,self.goal_site_list,self.agent
# class MiniGrid_fourroom_obs8x8(MiniGrid_fourroom):
#     def get_obs(self):
#         # self.full_img=self.render_layer(self.grid_size,[self.agent.site],red,copy.deepcopy(self.bg_layer))
#         self.sight_rect=self.get_sight_rect(self.agent)
#         # print(self.sight_rect,self.agent.site,self.agent.direction)
#         self.sight_img=crop_img(self.bg_layer,self.sight_rect)
#         # self.sight_img=np.tile(np.reshape(self.sight_img,[8,1,8,1,3]),[1,8,1,8,1]).reshape([64,64,3])
#         return self.sight_img

if __name__=="__main__":
    env=MiniGrid_fourroom_obs8x8()
    env.reset()
    while 1:
        env.step(np.clip(np.random.randint(0,4),0,2))
        env.render()

