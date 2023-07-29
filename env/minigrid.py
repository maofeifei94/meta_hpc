import numpy as np
import cv2
import copy
from mlutils.img import crop_img,rect
from mlutils.ml import array_in
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
        return rect(*minyx,*maxyx)
    def render_layer(self,img_size,site_list,color,img=None):
        img=np.zeros([*img_size,3],np.uint8) if img is None else img
        for site in site_list:
            img[site[0],site[1]]=color
        return img
class MiniGrid_fourroom(MiniGrid_base):
    def __init__(self ):
        self.grid_size=[19,19]
        self.obs_range=[8,8]
        self.max_steps=100
        self.build_wall()
    def reset(self):
        self.door_site_list,self.wall_site_list,self.goal_site_list,self.agent=self.build_grid()
        self.bg_layer=self.render_background()
        self.obs=self.get_obs()
        self.step_num=0
        return self.obs
        
    def step(self,action):
        self.agent.update(action,self.wall_site_list)
        self.obs=self.get_obs()

        self.step_num+=1

        reward,done=self.reward_done_func()
        # done=True if self.step_num<self.max_steps or re else True
        return self.obs,reward,done


    def get_obs(self):
        # self.full_img=self.render_layer(self.grid_size,[self.agent.site],red,copy.deepcopy(self.bg_layer))
        self.sight_rect=self.get_sight_rect(self.agent)
        # print(self.sight_rect,self.agent.site,self.agent.direction)
        self.sight_img=crop_img(self.bg_layer,self.sight_rect)
        self.sight_img=np.tile(np.reshape(self.sight_img,[8,1,8,1,3]),[1,8,1,8,1]).reshape([64,64,3])
        return self.sight_img
    def reward_done_func(self):
        
        if array_in(self.agent.site,self.goal_site_list):
            reward=1
            done=True
        else:
            reward=0
            done=True if self.step_num>=self.max_steps else False

        return reward,done
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
            cv2.rectangle(img,
            (int(sight_rect.minx*scale),int(sight_rect.miny*scale)),
            (int((sight_rect.maxx+1)*scale-1),int((sight_rect.maxy+1)*scale-1)),
            (0,0,255),
            2
            )
            return img


        render_img=copy.deepcopy(self.bg_layer)
        # render_img=self.render_layer(self.grid_size,[self.agent.site],self.agent.color,render_img)
        render_img=cv2.resize(render_img,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        render_img=draw_sight(self.sight_rect,render_img,scale)
        render_img=draw_agent(self.agent,render_img)
        if show:
            cv2.imshow("MiniGrid_fourroom",render_img)
            cv2.imshow("obs",cv2.resize(self.obs,None,fx=5,fy=5,interpolation=cv2.INTER_AREA))
            cv2.waitKey()
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
class MiniGrid_fourroom_obs8x8(MiniGrid_fourroom):
    def get_obs(self):
        # self.full_img=self.render_layer(self.grid_size,[self.agent.site],red,copy.deepcopy(self.bg_layer))
        self.sight_rect=self.get_sight_rect(self.agent)
        # print(self.sight_rect,self.agent.site,self.agent.direction)
        self.sight_img=crop_img(self.bg_layer,self.sight_rect)
        # self.sight_img=np.tile(np.reshape(self.sight_img,[8,1,8,1,3]),[1,8,1,8,1]).reshape([64,64,3])
        return self.sight_img

if __name__=="__main__":
    env=MiniGrid_fourroom()
    env.reset()
    while 1:
        env.step(np.clip(np.random.randint(0,4),0,2))
        env.render()

