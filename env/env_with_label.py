from asyncio import create_subprocess_exec
from turtle import back
import Box2D
from Box2D import (b2World, b2AABB, b2CircleShape, b2Color, b2Vec2)
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape,
                   b2_pi)
from Box2D import b2DrawExtended,b2ContactListener,b2ContactEdge
import pygame
import numpy as np
import cv2
import copy
import random
import time
import sys 
import os
import paddle
sys.path.append(".") 
from myf_ML_util import timer_tool

def cvcolor(color):
    return int(255.0 * color[2]), int(255.0 * color[1]), int(255.0 * color[0])
def cvcoord(pos):
    return tuple(map(int, pos))
class CvDraw(b2DrawExtended):
    """
    This debug draw class accepts callbacks from Box2D (which specifies what to
    draw) and handles all of the rendering.

    If you are writing your own game, you likely will not want to use debug
    drawing.  Debug drawing, as its name implies, is for debugging.
    """
    surface = None
    axisScale = 10.0

    def __init__(self, **kwargs):
        b2DrawExtended.__init__(self, **kwargs)
        self.flipX = False
        self.flipY = False
        self.convertVertices = False
        self.test = test_func()


    def StartDraw(self,center):
        # print(center,type(center))
        
        ws=128
        # center=[300,300]
        # ws=600

        self.zoom = self.test.zoom
        self.center = (center[0]-ws//2,center[1]-ws//2)
        self.offset = self.test.offset
        self.screenSize = [ws,ws]
        
        self.surface=np.zeros([ws,ws,3],np.uint8)

    def sub_center(self,p):
        c=np.array(self.center,dtype=np.int32)
        return p-c
    def EndDraw(self):
        pass

    def DrawPoint(self, p, size, color):
        """
        Draw a single point at point p given a pixel size and color.
        """
        self.DrawCircle(p, size / self.zoom, color, drawwidth=0)

    def DrawAABB(self, aabb, color):
        """
        Draw a wireframe around the AABB with the given color.
        """
        points = [(aabb.lowerBound.x, aabb.lowerBound.y),
                  (aabb.upperBound.x, aabb.lowerBound.y),
                  (aabb.upperBound.x, aabb.upperBound.y),
                  (aabb.lowerBound.x, aabb.upperBound.y)]

        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(self.surface, [pts], True, cvcolor(color))

    def DrawSegment(self, p1, p2, color):
        """
        Draw the line segment from p1-p2 with the specified color.
        """
        cv2.line(self.surface, cvcoord(p1), cvcoord(p2), cvcolor(color), 1)

    def DrawTransform(self, xf):
        """
        Draw the transform xf on the screen
        """
        print("drawtransform")
        p1 = xf.position
        p2 = self.to_screen(p1 + self.axisScale * xf.R.x_axis)
        p3 = self.to_screen(p1 + self.axisScale * xf.R.y_axis)
        p1 = self.to_screen(p1)
        cv2.line(self.surface, cvcoord(p1), cvcoord(p2), (0, 0, 255), 1)
        cv2.line(self.surface, cvcoord(p1), cvcoord(p3), (0, 255, 0), 1)

    def DrawCircle(self, center, radius, color, drawwidth=1):
        """
        Draw a wireframe circle given the center, radius, axis of orientation
        and color.
        """
        center=self.sub_center(center)
        radius *= self.zoom
        if radius < 1:
            radius = 1
        else:
            radius = int(radius)

        cv2.circle(self.surface, cvcoord(center),
                   radius, cvcolor(color), drawwidth)

    def DrawSolidCircle(self, center, radius, axis, color):
        """
        Draw a solid circle given the center, radius, axis of orientation and
        color.
        """
        center=self.sub_center(center)

        radius *= self.zoom
        if radius < 1:
            radius = 1
        else:
            radius = int(radius)

        FILL = True
        # print(center)
        if radius<10:
            color=(0.2,0.2,0.8)
        cv2.circle(self.surface, cvcoord(center), radius,
                   cvcolor(color), -1 if FILL else 1)

        # cv2.line(self.surface, cvcoord(center),
        #          cvcoord((center[0] - radius * axis[0],
        #                   center[1] + radius * axis[1])),
        #          (0, 0, 255),
        #          1)

    def DrawPolygon(self, vertices, color):
        """
        Draw a wireframe polygon given the screen vertices with the specified
        color.
        """
        if not vertices:
            return
        if len(vertices) == 2:
            cv2.line(self.surface, cvcoord(vertices[0]), cvcoord(
                vertices[1]), cvcolor(color), 1)
        else:
            pts = np.array(vertices, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(self.surface, [pts], True, cvcolor(color))

    def DrawSolidPolygon(self, vertices, color):
        """
        Draw a filled polygon given the screen vertices with the specified color.
        """
        FILL = True

        if not FILL:
            self.DrawPolygon(vertices, color)
            return

        if not vertices:
            return
        # print(vertices)
        # vertices=np.array(vertices,np.int32)
        # vertices-=self.center
        # print("vert before",vertices)
        vertices=self.sub_center(vertices)
        # print("vert after",vertices)
        if len(vertices) == 2:
            cv2.line(self.surface, cvcoord(vertices[0]), cvcoord(
                vertices[1]), cvcolor(color), 1)
        else:
            pts = np.array(vertices, np.int32)
            pts = pts.reshape((-1, 1, 2))
            # pts-=self.center
            cv2.fillPoly(self.surface, [pts], cvcolor(color))
class test_func():
    def __init__(self):
        self.zoom = 1.0
        self.center = (0,0)
        self.offset = (-0,-0)
        self.screenSize = (600,600)

"""
参数说明
friction:摩擦力
restitution:碰撞弹性
stiffness:拉伸弹性系数
"""
def create_box(world,box_center,box_size,friction=1.0,restitution=1.0,type="static"):
    if type=="static":
        func=world.CreateStaticBody
    elif type=="dynamic":
        func=world.CreateDynamicBody
    box=func(
        position=box_center,
        fixtures=b2FixtureDef(
            shape=b2PolygonShape(box=box_size),
            friction=friction,
            restitution=restitution,
            density=0.5,
            )
        )
    return box
def create_circle(world,circle_center,circle_radiu,friction=1.0,restitution=1.0,type="static"):
    if type=="static":
        func=world.CreateStaticBody
    elif type=="dynamic":
        func=world.CreateDynamicBody
    circle=func(
        position=circle_center,
        fixtures=b2FixtureDef(
            shape=b2CircleShape(radius=circle_radiu),
            friction=friction,
            restitution=restitution,
            density=0.5,
            )
        )
    return circle
def create_goal(world,circle_center,circle_radiu,friction=1.0,restitution=1.0,type="static"):
    if type=="static":
        func=world.CreateStaticBody
    elif type=="dynamic":
        func=world.CreateDynamicBody
    circle=func(
        position=circle_center,
        fixtures=b2FixtureDef(
            shape=b2CircleShape(radius=circle_radiu),
            friction=friction,
            restitution=restitution,
            isSensor=True,
            density=0.5,
            )
        )
    return circle
    
class multi_circle_env():
    def __init__(self,show=False):
        "[width,height]"
        self.grid_size=np.array([20,20],dtype=np.int)
        self.block_size=np.array([30,30],dtype=np.int)

        self.img_size=(self.grid_size*self.block_size).astype(np.float32)

        self.max_steps_num=400
        self.hz=20
        self.show=show

        self.generate_rand_maze()


    def generate_rand_maze(self):
        "draw wall"
        # door_list=[[10,5],[10,15],[5,10],[15,10]]
        door_list=[]
        wall_site_list=[]

        for x in range(20):
            for y in [0,19]:
                if not [x,y] in wall_site_list:
                    wall_site_list.append([x,y])
        for y in range(20):
            for x in [0,19]:
                if not [x,y] in wall_site_list:
                    wall_site_list.append([x,y])

        "set_goal"
        # self.goal_list=[]
        goal_num=20
        self.goal_num=goal_num
        real_goal_num=20
        self.real_goal_num=real_goal_num
        goal_site_list=[]
        self.real_goal_index_list=[]
        

        for i in range(goal_num):
            while 1:
                goal_site=list(np.random.randint(1,19,[2]))
                if goal_site in wall_site_list or goal_site in goal_site_list:
                    continue
                else:
                    break
            goal_site_list.append(goal_site)
        self.real_goal_index_list=list(range(goal_num))
        random.shuffle(self.real_goal_index_list)
        self.real_goal_index_list=self.real_goal_index_list[:real_goal_num]
        "goal_flag 1:奖励 0:惩罚 -1:被吃掉 -2:已移除"
        self.goal_flag_list=np.zeros([goal_num],dtype=np.int32)
        self.goal_flag_list[self.real_goal_index_list]=1
        # self.real_goal=self.goal_list[0]
        self.wall_site_list,self.door_list,self.goal_site_list=wall_site_list,door_list,goal_site_list,
    def rand_real_goal(self):
        self.real_goal_index_list=list(range(self.goal_num))
        random.shuffle(self.real_goal_index_list)
        self.real_goal_index_list=self.real_goal_index_list[:self.real_goal_num]
        "goal_flag 1:奖励 0:惩罚 -1:被吃掉 -2:已移除"
        self.goal_flag_list=np.zeros([self.goal_num],dtype=np.int32)
        self.goal_flag_list[self.real_goal_index_list]=1
    def save_maze_param(self,maze_name,save_dir="maze_param"):
        maze_param=[self.wall_site_list,self.door_list,self.goal_site_list,self.real_goal_index_list,self.goal_flag_list]
        # now_time=time.localtime(time.time())
        npy_path=f"{save_dir}/maze_{maze_name}.npy"
        np.save(npy_path,maze_param)
        return npy_path
    def load_maze_param(self,file_path):
        self.wall_site_list,self.door_list,self.goal_site_list,self.real_goal_index_list,self.goal_flag_list=np.load(file_path,allow_pickle=True)
        # print(self.goal_site_list,self.real_goal_index_list)
        # print("load maze param success")
    def reset(self,):
        self.tt=timer_tool("get obs",False)
        self.generate_rand_maze()
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.renderer=CvDraw()
        self.world.renderer=self.renderer
        

        "draw wall"
        for wall in self.wall_site_list:
            if not wall in self.door_list:
                self._add_wall(grid_y=wall[1],grid_x=wall[0])
        "set ball"
        while 1:
            ball_site=list(np.random.randint(1,19,[2]))
            if ball_site in self.wall_site_list or ball_site in self.goal_site_list:
                continue
            else:
                self.ball=self._add_ball(ball_site[1],ball_site[0])
                # print('ball_site',ball_site,self._get_position(self.ball))
                break
                
        "draw goal"
        self.goal_list=[]
        for goal_site in self.goal_site_list:
            goal=self._add_goal(goal_site[1],goal_site[0])
            self.goal_list.append(goal)
        self.real_goal_list=[self.goal_list[rg] for rg in self.real_goal_index_list]



        # self.contact=b2ContactListener([self.real_goal,self.ball])
        # print(self.contact)
        
        # self.world.contactListener
        

        # "bg_img"
        # self.background_img=self.render_img()   
        # self.background_img=np.ascontiguousarray(self.background_img)
        "prepare texture"
        goal_radiu=int(self.block_size[0]*0.5)
        self.goal_texture=np.zeros([goal_radiu*2+1,goal_radiu*2+1,3],dtype=np.uint8)
        cv2.circle(self.goal_texture,(goal_radiu,goal_radiu),goal_radiu,(200,200,200),-1)

        self.goal_texture_paddle=paddle.to_tensor(self.goal_texture.astype(np.int32))

        self.real_goal_label_texture=np.full([goal_radiu*2+1,goal_radiu*2+1,1],0.5,dtype=np.float32)
        cv2.circle(self.real_goal_label_texture,(goal_radiu,goal_radiu),goal_radiu,(1.0),-1)

        self.fake_goal_label_texture=np.full([goal_radiu*2+1,goal_radiu*2+1,1],0.5,dtype=np.float32)
        cv2.circle(self.fake_goal_label_texture,(goal_radiu,goal_radiu),goal_radiu,(0.0),-1)

        ball_radiu=int(self.block_size[0]*0.25)
        self.ball_texture=np.zeros([ball_radiu*2+1,ball_radiu*2+1,3],dtype=np.uint8)
        cv2.circle(self.ball_texture,(ball_radiu,ball_radiu),ball_radiu,(200,50,50),-1)

        "prepare_data"
        self.goal_img_site_list=self._get_goal_site()
        self.goal_img_site_norm_list=np.array(self.goal_img_site_list,np.float32)/self.img_size
        
        self.real_goal_img_site_list=self.goal_img_site_list[self.real_goal_index_list]
        self.real_goal_img_site_norm_list=np.array(self.real_goal_img_site_list,np.float32)/self.img_size
        # print(self.goal_img_site_list.dtype)
        # input()

    
        "init count"
        self.steps_num=0
        self.target_pos=np.array([0.5,0.5])


        self.ask_place=np.array([0.5,0.5])
        self.get_info()
        obs=self.get_obs()
        return obs
    def step(self,action,ask_place=None,hpc_train_mode=False):
        # if target is not None:
        self.ask_place=ask_place
        tt=timer_tool("env",False)


        "更新goal是否被吃掉"
        if not hpc_train_mode:
            self.process_goal()

        tt.end_and_start("process_goal")

        "physics step"
        # self._apply_action_to_ball(action)
        # if place is not None:
        #     self._apply_place_to_ball(place)
        # else:
        self._apply_action_to_ball(action)
        
        self.physics_run()

        self.steps_num+=1

        tt.end_and_start("physics")
        # print(dir(self.ball))
        "obs reward done"
        obs=self.get_obs()
        self.obs=obs
        # print(self._get_ball_site())
        tt.end_and_start("render")

        reward,done=self.reward_func(hpc_train_mode)
        tt.end_and_start('reward')

        tt.analyze()
        # print(self._get_ball_site())

        return obs,reward,done
    def process_goal(self):
        for i in range(len(self.goal_list)):
            if self.goal_flag_list[i]==-1:
                # print(f"destroy goal {i} and body num= {self.world.bodyCount}")
                # self.world.DestroyFixture(self.goal_list[i].)
                self.world.DestroyBody(self.goal_list[i])
                # self.world.destroy
                # print(f"destroy goal {i}")
                # self.world.Destroy
                self.goal_flag_list[i]=-2

    def reward_func(self,hpc_train_mode):
        # if len(self.ball.contacts)>0:
        #     print(dir(self.ball.contacts[0]))
        #     print(dir(self.ball.contacts[0].contact))
        #     a=self.ball.contacts[0].contact.fixtureA.body
        #     b=self.ball.contacts[0].contact.fixtureB.body
        #     # print(,)
        #     print(type(self.ball),type(a),type(b))
        #     print(self.ball==a,np.sum([g==b for g in self.goal_list]))
        #     for g in self.goal_list:
        #         if g==b:
        #             self.world.DestroyBody(g)
        #             self.goal_list.remove(g)
        #     print(len(self.goal_list),self.world.bodyCount)
        # print(dir(self.ball))
        # reward=0
        # if len(self.ball.contacts)>0:
        #     a=self.ball.contacts[0].contact.fixtureA.body
        #     b=self.ball.contacts[0].contact.fixtureB.body
        #     if b==self.real_goal:
        #         reward=1

        reward=0

        ball_pos=self._get_position(self.ball)
        goal_pos_list=list(map(self._get_position,self.goal_list))
        ball_radiu=self._get_radiu(self.ball)
        goal_radiu_list=list(map(self._get_radiu,self.goal_list))
        # print(ball_pos,real_goal_pos,ball_radiu,real_goal_radiu_list)

        # test_img=np.zeros([600,600],dtype=np.uint8)
        # cv2.circle(test_img,(int(ball_pos[0]),int(ball_pos[1])),int(ball_radiu),(255),-1)
        # for
        # cv2.imshow("test_img",test_img)

        for i in range(len(self.goal_list)):
            goal_flag=self.goal_flag_list[i]
            if goal_flag in [-1,-2]:
                continue
            goal_pos=goal_pos_list[i]
            goal_radiu=goal_radiu_list[i]
            
            d=np.linalg.norm(np.array(goal_pos)-ball_pos)
            # print(i,"distance=",d,ball_radiu+goal_radiu,goal_pos,ball_pos)
            if d<=ball_radiu+goal_radiu:
                if goal_flag==1:
                    reward+=1
                else:
                    reward+=-1
                # print(f"eat goal {i} with flag {goal_flag},distance={d}")

                if not hpc_train_mode:
                    self.goal_flag_list[i]=-1

        # ball_pos=self._get_position(self.ball)
        # print("ball_pos=",ball_pos,dir(self.ball.fixtures[0].shape.radiu),len(self.ball.fixtures))

        done=self.steps_num>=self.max_steps_num or np.sum(self.goal_flag_list==1)<=0

        if np.sum(self.goal_flag_list==1)<=0:
            reward=10
        # print("left_goal",np.sum(self.goal_flag_list==1),done)

        return reward,done

    def get_info(self):
        info={}

        

        self.info=info
        return info

    def get_obs(self):

        tt=self.tt
        self.tt.start()
        all_obs={}

        ball_site=self._get_ball_site()
        all_obs["ball_site"]=ball_site
        all_obs["ball_site_norm"]=np.array(ball_site,np.float32)/self.img_size
        tt.end_and_start("ball_site")


        all_obs["goal_site_list"]=self.goal_img_site_list
        all_obs["goal_site_norm_list"]=self.goal_img_site_norm_list
        tt.end_and_start("goal_site")


        all_obs["real_goal_site"]=self.real_goal_img_site_list
        all_obs["real_goal_site_norm"]=self.real_goal_img_site_norm_list
        tt.end_and_start("real_goal_site")
        

        cv_img=self.render_img_cv(ball_site)
        all_obs["ball_obs"]=np.array(cv2.resize(cv_img,(64,64),interpolation=cv2.INTER_AREA),dtype=np.float32)/255
        tt.end_and_start('world_cv')
        tt.end_and_start("img_cv")

        # tt.start()
        # cv_img=self.render_img_cv(ball_site)
        # all_obs["self_cv"]=np.array(cv2.resize(cv_img,(64,64),interpolation=cv2.INTER_AREA),dtype=np.float32)/255
        # tt.end_and_start('self_cv')


        # cv_img_texture=self.render_img_texture(ball_site)
        # all_obs["ball_obs_texture"]=np.array(cv2.resize(cv_img_texture,(64,64),interpolation=cv2.INTER_AREA),dtype=np.float32)/255
        # tt.end_and_start("texture")

        # paddle_img_texture=self.render_img_texture_paddle(ball_site)
        # tt.end_and_start("texture_paddle")

        label_img=self.render_label(ball_site)
        all_obs["label_img"]=np.expand_dims(cv2.resize(label_img,(64,64),interpolation=cv2.INTER_AREA),axis=-1)
        tt.end_and_start('label1')


        ask_label_img=self.render_label(np.array(self.ask_place,dtype=np.float32)*600)
        all_obs["ask_label_img"]=np.expand_dims(cv2.resize(ask_label_img,(64,64),interpolation=cv2.INTER_AREA),axis=-1)
        tt.end_and_start("lable2")

        all_obs["ask_site_norm"]=np.array(self.ask_place,dtype=np.float32)

        tt.analyze()
        
        return all_obs

    def physics_run(self):
        for i in range(20):
            self.world.Step(1/20,8,3)
            
            # print(self.ball.linearVelocity)
            # print(self._get_ball_site())
    def render_label(self,ball_center):
        img_size=128
        half_img_size=img_size//2
        label_img=np.full([img_size,img_size,1],0.5,np.float32)
        goal_radiu=int(self.block_size[0]*0.5)

        def sub_center(p,c):
            out=np.array(p-c+half_img_size,np.int32)
            return out
        goal_center_site_list=sub_center((self.block_size*(np.array(self.goal_site_list,np.float32)+0.5)[:,::-1]).astype(np.int32),ball_center)
        for goal_center,goal_flag in zip(goal_center_site_list,self.goal_flag_list):

            "goal_flag 1:奖励 0:惩罚 -1:被吃掉 -2:已移除"
            if goal_flag==1:
                color=(1.0)
            elif goal_flag==0:
                color=(0.0)
            else:
                continue


            cv2.circle(label_img,(goal_center[0],goal_center[1]),goal_radiu,color,-1)
            # print(f"cv circle cost {(time.time()-t1)*1e6}")
        return label_img
    def render_label_texture(self,ball_center):
        img_size=128
        half_img_size=img_size//2

        label_img=np.full([img_size,img_size,1],0.5,np.float32)
        goal_radiu=int(self.block_size[0]*0.5)

        def sub_center(p,c):
            out=np.array(p,np.int32)-np.array(c,dtype=np.int32)
            return (out[0]+half_img_size,out[1]+half_img_size)

        for goal_site,goal_flag in zip(self.goal_site_list,self.goal_flag_list):
            "goal_flag 1:奖励 0:惩罚 -1:被吃掉 -2:已移除"
            if goal_flag<-1:
                continue
            goal_site_minxy=self.block_size*np.array(goal_site,np.float32)[::-1]
            goal_site_maxxy=self.block_size*(np.array(goal_site,np.float32)+1)[::-1]+1

            goal_img_site_minxy=sub_center(goal_site_minxy,ball_center)
            goal_img_site_maxxy=sub_center(goal_site_maxxy,ball_center)

            if (np.array(goal_img_site_maxxy)<=0).any() or (np.array(goal_img_site_minxy)>=img_size).any():
                continue

            texture=self.real_goal_label_texture if goal_flag==1 else self.fake_goal_label_texture
            # cv2.copyTo()
            t1=time.time()
            # label_img[
            #     max(goal_img_site_minxy[1],0):min(goal_img_site_maxxy[1],img_size),
            #     max(goal_img_site_minxy[0],0):min(goal_img_site_maxxy[0],img_size)
            #     ]=texture[
            #         max(goal_img_site_minxy[1],0)-goal_img_site_minxy[1]:goal_radiu*2+1-(goal_img_site_maxxy[1]-min(goal_img_site_maxxy[1],img_size)),
            #         max(goal_img_site_minxy[0],0)-goal_img_site_minxy[0]:goal_radiu*2+1-(goal_img_site_maxxy[0]-min(goal_img_site_maxxy[0],img_size))
            #         ]
            cv2.copyTo(texture[
                    max(goal_img_site_minxy[1],0)-goal_img_site_minxy[1]:goal_radiu*2+1-(goal_img_site_maxxy[1]-min(goal_img_site_maxxy[1],img_size)),
                    max(goal_img_site_minxy[0],0)-goal_img_site_minxy[0]:goal_radiu*2+1-(goal_img_site_maxxy[0]-min(goal_img_site_maxxy[0],img_size))
                    ],None,label_img[
                max(goal_img_site_minxy[1],0):min(goal_img_site_maxxy[1],img_size),
                max(goal_img_site_minxy[0],0):min(goal_img_site_maxxy[0],img_size)
                ])
            print(f"texture circle cost {(time.time()-t1)*1e6}")
        return label_img


    def render_img(self,center):

        # tt=timer_tool("env render")
        # tt.start()
        self.renderer.StartDraw(center)
        # tt.end_and_start("fill")
        self.world.renderer.flags = dict(
                drawShapes=True,
                # drawJoints=True,
                # drawAABBs=True,
                # drawPairs=True,
                # drawCOMs=True,
                # convertVertices=True,
                )
        self.world.DrawDebugData()
        cv_img=self.renderer.surface
        return cv_img
    def render_img_cv(self,ball_center):
        tt=timer_tool("**********render_img_cv",False)
        img_size=128
        half_img_size=img_size//2
        cv_img=np.full([img_size,img_size,3],0,np.uint8)
        goal_radiu=int(self.block_size[0]*0.5)
        tt.end_and_start("***init img")

        def sub_center(p,c):
            out=np.array(p-c+half_img_size,np.int32)
            return out

   

        wall_site_list_minxy=self.block_size*np.array(self.wall_site_list,np.float32)[:,::-1]
        wall_site_list_maxxy=self.block_size*(np.array(self.wall_site_list,np.float32)+1)[:,::-1]
        wall_img_site_list_minxy=sub_center(wall_site_list_minxy,ball_center)
        wall_img_site_list_maxxy=sub_center(wall_site_list_maxxy,ball_center)
        wall_need_draw=np.all(wall_img_site_list_maxxy>0,axis=-1)*np.all(wall_img_site_list_minxy<img_size,axis=-1)
        
        for i in range(len(wall_need_draw)):
            if not wall_need_draw[i]:
                continue

            cv2.rectangle(cv_img,
            (wall_img_site_list_minxy[i][0],wall_img_site_list_minxy[i][1]),
            (wall_img_site_list_maxxy[i][0],wall_img_site_list_maxxy[i][1]),
            (50,200,50),-1)
            # total_cv_time+=time.time()-t1

        tt.end_and_start("***wall")

        goal_center_site_list=sub_center((self.block_size*(np.array(self.goal_site_list,np.float32)+0.5)[:,::-1]).astype(np.int32),ball_center)
        for i in range(len(self.goal_site_list)):
            if self.goal_flag_list[i]<-1:
                continue
            # t1=time.time()
            cv2.circle(cv_img,(goal_center_site_list[i][0],goal_center_site_list[i][1]),goal_radiu,(200,200,200),-1)
            # total_cv_time+=time.time()-t1

        ball_radiu=int(goal_radiu/2)
        # t1=time.time()
        cv2.circle(cv_img,(half_img_size,half_img_size),ball_radiu,(200,50,50),-1)
        # total_cv_time+=time.time()-t1
        # print("self cv cost",total_cv_time)
        return cv_img

    def render(self):
        img_size=600
        half_img_size=img_size//2
        cv_img=np.full([img_size,img_size,3],0,np.uint8)
        goal_radiu=int(self.block_size[0]*0.5)

        wall_site_list_minxy=self.block_size*np.array(self.wall_site_list,np.int32)[:,::-1]
        wall_site_list_maxxy=self.block_size*(np.array(self.wall_site_list,np.int32)+1)[:,::-1]
        
        for i in range(len(wall_site_list_minxy)):
            # print(wall_site_list_minxy[i][0])

            cv2.rectangle(cv_img,
            (wall_site_list_minxy[i][0],wall_site_list_minxy[i][1]),
            (wall_site_list_maxxy[i][0],wall_site_list_maxxy[i][1]),
            (50,200,50),-1)


        goal_center_site_list=(self.block_size*(np.array(self.goal_site_list,np.float32)+0.5)[:,::-1]).astype(np.int32)
        for i in range(len(self.goal_site_list)):
            if self.goal_flag_list[i]<-1:
                continue
            # t1=time.time()
            cv2.circle(cv_img,(goal_center_site_list[i][0],goal_center_site_list[i][1]),goal_radiu,(200,200,200),-1)
            # total_cv_time+=time.time()-t1
        ball_site=self._get_ball_site()
        ball_radiu=int(goal_radiu/2)
        # t1=time.time()
        cv2.circle(cv_img,(int(ball_site[0]),int(ball_site[1])),ball_radiu,(200,50,50),-1)
        
        cv2.rectangle(cv_img,(int(ball_site[0])-64,int(ball_site[1])-64),(int(ball_site[0])+64,int(ball_site[1])+64),(200,50,50))

        ask_place=(self.ask_place*600).astype(np.int32)
        cv2.rectangle(cv_img,(int(ask_place[0])-64,int(ask_place[1])-64),(int(ask_place[0])+64,int(ask_place[1])+64),(200,50,150))
        # total_cv_time+=time.time()-t1
        # print("self cv cost",total_cv_time)
        obs=self.obs
        cv2.imshow('ballobs',cv2.resize(obs["ball_obs"],dsize=(256,256)))
        # cv2.imshow('self_cv',cv2.resize(obs["self_cv"],dsize=(256,256)))
        cv2.imshow('labelimg',cv2.resize(obs['label_img'],dsize=(256,256)))
        cv2.imshow('asklabelimg',cv2.resize(obs['ask_label_img'],dsize=(256,256)))

        cv2.imshow("render",cv_img)




        cv2.waitKey()




    def render_img_texture_paddle(self,ball_center):
        img_size=128
        half_img_size=img_size//2
        paddle_img=paddle.zeros([img_size,img_size,3],dtype="int32")
        goal_radiu=int(self.block_size[0]*0.5)

        def sub_center(p,c):
            out=np.array(p,np.int32)-np.array(c,dtype=np.int32)
            return (int(out[0]+half_img_size),int(out[1]+half_img_size))
        
        for wall_site in self.wall_site_list:
            wall_site_minxy=self.block_size*np.array(wall_site,np.float32)[::-1]
            wall_site_maxxy=self.block_size*(np.array(wall_site,np.float32)+1)[::-1]

            wall_img_site_minxy=sub_center(wall_site_minxy,ball_center)
            wall_img_site_maxxy=sub_center(wall_site_maxxy,ball_center)

            if (np.array(wall_img_site_maxxy)<=0).any() or (np.array(wall_img_site_minxy)>=img_size).any():
                continue
            # print("type of max",type(max(wall_img_site_minxy[1],0)))
            print(max(wall_img_site_minxy[1],0),min(wall_img_site_maxxy[1],img_size),
                max(wall_img_site_minxy[0],0),min(wall_img_site_maxxy[0],img_size))
            paddle_img[
                max(wall_img_site_minxy[1],0):min(wall_img_site_maxxy[1],img_size),
                max(wall_img_site_minxy[0],0):min(wall_img_site_maxxy[0],img_size)
                ]=paddle.to_tensor([50,250,50],dtype='int32')
        

        # cv2.imshow("goal_img",cv2.resize(self.goal_img,None,fx=10,fy=10,interpolation=cv2.INTER_AREA))
        # cv2.waitKey()
        for goal_site,goal_flag in zip(self.goal_site_list,self.goal_flag_list):
            "goal_flag 1:奖励 0:惩罚 -1:被吃掉 -2:已移除"
            if goal_flag<-1:
                continue
            goal_site_minxy=self.block_size*np.array(goal_site,np.float32)[::-1]
            goal_site_maxxy=self.block_size*(np.array(goal_site,np.float32)+1)[::-1]+1

            goal_img_site_minxy=sub_center(goal_site_minxy,ball_center)
            goal_img_site_maxxy=sub_center(goal_site_maxxy,ball_center)

            if (np.array(goal_img_site_maxxy)<=0).any() or (np.array(goal_img_site_minxy)>=img_size).any():
                continue
            # print(type())
            paddle_img[
               max(goal_img_site_minxy[1],0):min(goal_img_site_maxxy[1],img_size),
                max(goal_img_site_minxy[0],0):min(goal_img_site_maxxy[0],img_size)
                ]=self.goal_texture_paddle[
                    max(goal_img_site_minxy[1],0)-goal_img_site_minxy[1]:goal_radiu*2+1-(goal_img_site_maxxy[1]-min(goal_img_site_maxxy[1],img_size)),
                    max(goal_img_site_minxy[0],0)-goal_img_site_minxy[0]:goal_radiu*2+1-(goal_img_site_maxxy[0]-min(goal_img_site_maxxy[0],img_size))
                    ]

        ball_radiu=int(goal_radiu/2)
        # cv2.circle(cv_img,(half_img_size,half_img_size),ball_radiu,(200,50,50),-1)

        # cv_img[half_img_size-ball_radiu:half_img_size+ball_radiu+1,half_img_size-ball_radiu:half_img_size+ball_radiu+1]=self.ball_texture
            # # print("render label",goal_center_img_site,goal_flag,ball_center)
            # cv2.circle(label_img,goal_center_label_img_site,goal_radiu,color,-1)
        return paddle_img
    def render_img_texture(self,ball_center):
        img_size=128
        half_img_size=img_size//2
        cv_img=np.full([img_size,img_size,3],0,np.uint8)
        goal_radiu=int(self.block_size[0]*0.5)

        def sub_center(p,c):
            out=np.array(p,np.int32)-np.array(c,dtype=np.int32)
            return (out[0]+half_img_size,out[1]+half_img_size)
        
        for wall_site in self.wall_site_list:
            wall_site_minxy=self.block_size*np.array(wall_site,np.float32)[::-1]
            wall_site_maxxy=self.block_size*(np.array(wall_site,np.float32)+1)[::-1]

            wall_img_site_minxy=sub_center(wall_site_minxy,ball_center)
            wall_img_site_maxxy=sub_center(wall_site_maxxy,ball_center)

            if (np.array(wall_img_site_maxxy)<=0).any() or (np.array(wall_img_site_minxy)>=img_size).any():
                continue

            cv_img[max(wall_img_site_minxy[1],0):min(wall_img_site_maxxy[1],img_size),max(wall_img_site_minxy[0],0):min(wall_img_site_maxxy[0],img_size)]=[50,250,50]
        

        # cv2.imshow("goal_img",cv2.resize(self.goal_img,None,fx=10,fy=10,interpolation=cv2.INTER_AREA))
        # cv2.waitKey()
        for goal_site,goal_flag in zip(self.goal_site_list,self.goal_flag_list):
            "goal_flag 1:奖励 0:惩罚 -1:被吃掉 -2:已移除"
            if goal_flag<-1:
                continue
            goal_site_minxy=self.block_size*np.array(goal_site,np.float32)[::-1]
            goal_site_maxxy=self.block_size*(np.array(goal_site,np.float32)+1)[::-1]+1

            goal_img_site_minxy=sub_center(goal_site_minxy,ball_center)
            goal_img_site_maxxy=sub_center(goal_site_maxxy,ball_center)

            if (np.array(goal_img_site_maxxy)<=0).any() or (np.array(goal_img_site_minxy)>=img_size).any():
                continue

            cv_img[
                max(goal_img_site_minxy[1],0):min(goal_img_site_maxxy[1],img_size),
                max(goal_img_site_minxy[0],0):min(goal_img_site_maxxy[0],img_size)
                ]=self.goal_texture[
                    max(goal_img_site_minxy[1],0)-goal_img_site_minxy[1]:goal_radiu*2+1-(goal_img_site_maxxy[1]-min(goal_img_site_maxxy[1],img_size)),
                    max(goal_img_site_minxy[0],0)-goal_img_site_minxy[0]:goal_radiu*2+1-(goal_img_site_maxxy[0]-min(goal_img_site_maxxy[0],img_size))
                    ]

        ball_radiu=int(goal_radiu/2)
        cv2.circle(cv_img,(half_img_size,half_img_size),ball_radiu,(200,50,50),-1)
        # cv_img[half_img_size-ball_radiu:half_img_size+ball_radiu+1,half_img_size-ball_radiu:half_img_size+ball_radiu+1]=self.ball_texture
            # # print("render label",goal_center_img_site,goal_flag,ball_center)
            # cv2.circle(label_img,goal_center_label_img_site,goal_radiu,color,-1)
        return cv_img



    def _add_wall(self,grid_y,grid_x):
        center,rect_size=self._grid_site_to_img_rect(grid_y,grid_x)
        create_box(self.world,center,rect_size)

    def _add_ball(self,grid_y,grid_x):
        center,rect_size=self._grid_site_to_img_rect(grid_y,grid_x)
        ball=create_circle(self.world,center,rect_size[0]/2,type="dynamic")
        return ball

    def _add_goal(self,grid_y,grid_x):
        center,rect_size=self._grid_site_to_img_rect(grid_y,grid_x)
        goal=create_goal(self.world,center,rect_size[0],type="dynamic")
        return goal

    def _crop_img(self,img,center_site,crop_img_size=[64,64]):
        # out_img_size=[64,64]
        img_size=img.shape
        crop_img=np.zeros([*crop_img_size,3],np.uint8)

        min_x=center_site[0]-int(crop_img_size[1]/2)
        img_min_x=int(max(min_x,0))
        crop_img_min_x=int(img_min_x-min_x)
        
        max_x=center_site[0]+int(crop_img_size[1]/2)
        img_max_x=int(min(max_x,img_size[1]-1))
        crop_img_max_x=crop_img_size[1]-int(max_x-img_max_x)

        min_y=center_site[1]-int(crop_img_size[0]/2)
        img_min_y=int(max(min_y,0))
        crop_img_min_y=int(img_min_y-min_y)
        
        max_y=center_site[1]+int(crop_img_size[0]/2)
        img_max_y=int(min(max_y,img_size[0]-1))
        crop_img_max_y=crop_img_size[0]-int(max_y-img_max_y)
        # print(min_y,max_y,min_x,max_x)
        # print(crop_img_min_y,crop_img_max_y,crop_img_min_x,crop_img_max_x)
        # print(img_min_y,img_max_y,img_min_x,img_max_x)
        crop_img[crop_img_min_y:crop_img_max_y,crop_img_min_x:crop_img_max_x]=img[img_min_y:img_max_y,img_min_x:img_max_x]
        # pass
        return crop_img
    def _grid_site_to_img_rect(self,y,x):
        center=np.array([y*self.block_size[0],x*self.block_size[1]])+self.block_size*0.5
        return center,self.block_size*0.5

    """
    ['ApplyAngularImpulse', 'ApplyForce', 'ApplyForceToCenter', 'ApplyLinearImpulse', 'ApplyTorque', 'ClearUserData', 
    'CreateChainFixture', 'CreateCircleFixture', 'CreateEdgeChain', 'CreateEdgeFixture', 'CreateFixture', 'CreateFixturesFromShapes', 
    'CreateLoopFixture', 'CreatePolygonFixture', 'DestroyFixture', 'Dump', 'GetLinearVelocityFromLocalPoint', 'GetLinearVelocityFromWorldPoint', 
    'GetLocalPoint', 'GetLocalVector', 'GetMassData', 'GetWorldPoint', 'GetWorldVector', 'ResetMassData', 
    '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', 
    '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', 
    '__init_subclass__', '__iter__', '__le__', '__lt__', '__module__', '__ne__', 
    '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', 
    '__str__', '__subclasshook__', '__weakref__','active', 'angle', 'angularDamping', 
    'angularVelocity', 'awake', 'bullet', 'contacts', 'contacts_gen', 'fixedRotation', 
    'fixtures', 'fixtures_gen', 'gravityScale', 'inertia', 'joints', 'joints_gen', 
    'linearDamping', 'linearVelocity', 'localCenter', 'mass', 'massData', 'next', 
    'position', 'sleepingAllowed', 'this', 'thisown', 'transform', 'type', 
    'userData', 'world', 'worldCenter']
    """
    def _apply_action_to_ball(self,action):
        target_v=action*20
        self.ball.linearVelocity=b2Vec2(float(target_v[0]),float(target_v[1]))
        # print(self.ball.linearVelocity)
    def _apply_place_to_ball(self,place):
        place=place*np.array(self.img_size)
        # print("ball_pos,place=",self.ball.position,place)
        self.ball.position=b2Vec2(float(place[0]),float(place[1]))
    def _apply_impulse_to_ball(self,impulse):
        self.ball.ApplyLinearImpulse(b2Vec2(float(impulse[0]),float(impulse[1])),(0,0),False)
    def _get_position(self,x):
        return list(x.position)
    def _get_ball_site(self):
        return self._get_position(self.ball)
    def _get_radiu(self,x):
        # print(type(x.fixtures[0].shape),dir(x.fixtures[0].shape))
        try:
            return x.fixtures[0].shape.radius
        except:
            return 0
    def _get_goal_site(self):
        return np.array([self._get_position(goal) for goal in self.goal_list],np.float32)
    def _update_target_pos(self,target_pos):
        gamma=0.5
        self.target_pos=self.target_pos*gamma+target_pos*(1-gamma)

if __name__=="__main__":
    e=multi_circle_env()
    # e=fourroom_env()
    e.reset()
    for i in range(100000):
        action=np.random.uniform(-1,1,[2])
        obs,reward,done=e.step(action,ask_place=[0.5,0.5])
        if done:
            print("done")
            e.reset()
        print(reward,obs['ball_site_norm'],action)
        cv2.imshow('ballobs',cv2.resize(obs["ball_obs"],dsize=(256,256)))
        # cv2.imshow('self_cv',cv2.resize(obs["self_cv"],dsize=(256,256)))
        cv2.imshow('labelimg',cv2.resize(obs['label_img'],dsize=(256,256)))
        cv2.imshow('asklabelimg',cv2.resize(obs['ask_label_img'],dsize=(256,256)))
        e.render()
        
        # obs,reward,done=e.step(None,place=[(i%101)*0.01]*2)
        # e.render()
        # e.render_img_cv()
        
        # if reward!=0:
        #     pass
