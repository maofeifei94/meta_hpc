from asyncio import create_subprocess_exec
from turtle import back
import Box2D
from Box2D import (b2World, b2AABB, b2CircleShape, b2Color, b2Vec2)
from Box2D import (b2CircleShape, b2EdgeShape, b2FixtureDef, b2PolygonShape,
                   b2_pi)
from Box2D import b2DrawExtended
import pygame
import numpy as np
import cv2
import copy
from myf_ML_util import timer_tool

# class cvDraw():
#     def __init__(self,grid_size,block_size) -> None:
#         self.grid_size=grid_size
#         self.block_size=block_size
#         self.img_size=grid_size*block_size
#         pass
    
#     def reset(self,wall_list):
#         self.bg_img=np.zeros(self.img_size,dtype=np.uint8)
#         for wall in wall_list:
#             wall
#         pass
#     def step(self,ball_site,goal_site):

def cvcolor(color):
    return int(255.0 * color[2]), int(255.0 * color[1]), int(255.0 * color[0])
def cvcoord(pos):
    return tuple(map(int, pos))
class PygameDraw(b2DrawExtended):
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
        self.flipY = True
        self.convertVertices = True
        self.test = test_func()

    def StartDraw(self):
        self.zoom = self.test.zoom
        self.center = self.test.center
        self.offset = self.test.offset
        self.screenSize = self.test.screenSize

        self.surface=np.zeros([600,600,3],np.uint8)

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
        radius *= self.zoom
        if radius < 1:
            radius = 1
        else:
            radius = int(radius)

        FILL = False

        cv2.circle(self.surface, cvcoord(center), radius,
                   cvcolor(color), -1 if FILL else 1)

        cv2.line(self.surface, cvcoord(center),
                 cvcoord((center[0] - radius * axis[0],
                          center[1] + radius * axis[1])),
                 (0, 0, 255),
                 1)

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
        FILL = False

        if not FILL:
            self.DrawPolygon(vertices, color)
            return

        if not vertices:
            return

        if len(vertices) == 2:
            cv2.line(self.surface, cvcoord(vertices[0]), cvcoord(
                vertices[1]), cvcolor(color), 1)
        else:
            pts = np.array(vertices, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(self.surface, [pts], cvcolor(color))

    # the to_screen conversions are done in C with b2DrawExtended, leading to
    # an increase in fps.
    # You can also use the base b2Draw and implement these yourself, as the
    # b2DrawExtended is implemented:
    # def to_screen(self, point):
    #     """
    #     Convert from world to screen coordinates.
    #     In the class instance, we store a zoom factor, an offset indicating where
    #     the view extents start at, and the screen size (in pixels).
    #     """
    #     x=(point.x * self.zoom)-self.offset.x
    #     if self.flipX:
    #         x = self.screenSize.x - x
    #     y=(point.y * self.zoom)-self.offset.y
    #     if self.flipY:
    #         y = self.screenSize.y-y
    #     return (x, y)

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
"""
链接参数
CreateRevoluteJoint:旋转点链接

"""
def create_rovolve_joint(world,bodyA,bodyB,anchor):
    joint=world.CreateRevoluteJoint(
        bodyA=bodyA,
        bodyB=bodyB,
        anchor=anchor,
        # lowerAngle=-8.0 * b2_pi / 180.0*4,
        # upperAngle=8.0 * b2_pi / 180.0*4,
        # enableLimit=True,
    )
    return joint
def create_prismatic_joint(world,bodyA,bodyB,anchor,axis):
    joint=world.CreatePrismaticJoint(bodyA=bodyA,bodyB=bodyB,anchor=anchor,axis=axis)
    return joint
def create_spring(world,bodyA,bodyB,anchorA,anchorB,frequencyHz=2.0,dampingRatio=3.0):
    
    spring=world.CreateDistanceJoint(
        bodyA=bodyA,
        bodyB=bodyB,
        anchorA=anchorA,
        anchorB=anchorB,
        frequencyHz=frequencyHz,
        dampingRatio=dampingRatio,
    )
    return spring

class fourroom_env():
    def __init__(self,show=False):
        "[width,height]"
        self.grid_size=np.array([20,20],dtype=np.int)
        self.block_size=np.array([30,30],dtype=np.int)

        self.img_size=self.grid_size*self.block_size

        self.max_steps_num=100
        self.hz=20
        self.show=show

        self.life_heat_map_img=np.zeros([24,24],dtype=np.float32)

        pass

    def reset(self,):
        self.world = b2World(gravity=(0, 0), doSleep=True)
        
        # if self.show:

        self.screen = pygame.display.set_mode(self.img_size)
        self.renderer=PygameDraw(surface=self.screen)
        self.world.renderer=self.renderer
        self.clock = pygame.time.Clock()

        "draw wall"
        door_list=[[10,5],[10,15],[5,10],[15,10]]
        wall_list=[]

        for x in range(20):
            for y in [0,10,19]:
                if not [x,y] in wall_list:
                    wall_list.append([x,y])
        for y in range(20):
            for x in [0,10,19]:
                if not [x,y] in wall_list:
                    wall_list.append([x,y])
        for wall in wall_list:
            if not wall in door_list:
                self._add_wall(grid_y=wall[1],grid_x=wall[0])
        "set_goal"
        self.goal_list=[]
        for i in range(1):
            while 1:
                goal_site=list(np.random.randint(1,19,[2]))
                if goal_site in wall_list:
                    continue
                else:
                    goal=self._add_goal(goal_site[1],goal_site[0])
                    break
            self.goal_list.append(goal)
        self.real_goal_index=0

        "bg_img"
        self.background_img=self.render_img()   
        self.background_img=np.ascontiguousarray(self.background_img)

        "set ball"
        while 1:
            ball_site=list(np.random.randint(1,19,[2]))
            if ball_site in wall_list:
                continue
            else:
                self.ball=self._add_ball(ball_site[1],ball_site[0])
                break

        
           
        "init count"
        self.steps_num=0

        self.target_pos=np.array([0.5,0.5])

        self.get_info()
        obs=self.get_obs()
        return obs
    def reward_func(self):
        # reward=0
        if np.linalg.norm(self.info['ball_site_norm']-self.info['real_goal_site_norm'])<=0.1:
            reward=1
            done=True
        else:
            reward=0
            done=False or self.steps_num>=self.max_steps_num
        return reward,done



    def step(self,action,target_pos=None):
        # if target is not None:
        tt=timer_tool("env")
        tt.start()
        self._update_target_pos(np.squeeze(target_pos))
        # print(target_pos,self.target_pos)

        "physics step"
        self._apply_action_to_ball(action)
        self.physics_run()
        self.steps_num+=1

        tt.end_and_start("physics")
        # print(dir(self.ball))
        "obs reward done"
        self.info=self.get_info()
        obs=self.get_obs()
        tt.end_and_start("render")
        reward,done=self.reward_func()
        # print(obs['ball_site'],obs['goal_site'])
        # if np.linalg.norm(obs['ball_site']-obs['goal_site'])<=0.1:
        #     reward=1
        #     done=True
        # else:
        #     reward=0
        #     done=False or self.steps_num>=self.max_steps_num
        # done=self.steps_num>=self.max_steps_num

        # print(obs)
        return obs,reward,done
        
    def get_info(self):
        info={}

        ball_site=self._get_ball_site()
        info["ball_site"]=ball_site
        ball_site_norm=np.array(ball_site)/self.img_size
        info["ball_site_norm"]=ball_site_norm

        goal_site_list=self._get_goal_site()
        info["goal_site_list"]=goal_site_list
        goal_site_norm_list=[np.array(goal_site)/self.img_size for goal_site in goal_site_list]
        info["goal_site_norm_list"]=goal_site_norm_list

        real_goal_site=goal_site_list[self.real_goal_index]
        info["real_goal_site"]=real_goal_site
        real_goal_site_norm=goal_site_norm_list[self.real_goal_index]
        info["real_goal_site_norm"]=real_goal_site_norm

        self.info=info
        return info

    def get_obs(self):
        cv_img=self.render_img_cv()
        
        ball_site=self.info["ball_site"]
        ball_site_norm=self.info["ball_site_norm"]
        real_goal_site_norm=self.info["real_goal_site_norm"]
        ball_obs=self._crop_img(cv_img,np.array([self.img_size[1]-ball_site[1],ball_site[0]],dtype=np.int),[256,256])
        self.ball_obs_for_show=ball_obs
        ball_obs=cv2.resize(ball_obs,(16,16),interpolation=cv2.INTER_AREA)
        ball_obs=np.transpose(ball_obs,[2,0,1]).astype(np.float32)/255

        heat_map=np.zeros([24,24],dtype=np.float32)
        ball_block_site=(ball_site_norm*self.grid_size).astype(np.int)
        goal_block_site=(real_goal_site_norm*self.grid_size).astype(np.int)
        heat_map[ball_block_site[0]][self.grid_size[1]-ball_block_site[1]]+=0.5
        heat_map[goal_block_site[0]][self.grid_size[1]-goal_block_site[1]]+=1.0
        
        # goal_site=self._get_goal_site()
        # goal_site_norm=np.array(goal_site)/self.img_size
        # cv2.imshow("img",cv_img)
        # cv2.imshow("crop_img",ball_obs)
        # cv2.waitKey()

        return {"ball_site":ball_site_norm,"goal_site":real_goal_site_norm,"ball_obs":ball_obs,"heat_map":heat_map}
    
    def physics_run(self):
        for i in range(20):
            self.world.Step(1/20,8,3)

    def render_img_cv(self):
        bg_img=copy.deepcopy(self.background_img)
        # print(type(bg_img))
        # cv2.circle()
        cv2.circle(bg_img,(self.img_size[0]-int(self.info["ball_site"][1]),int(self.info["ball_site"][0])),radius=8,color=(100,100,100),thickness=-1)
        # cv2.imshow("render_cv",bg_img)
        return bg_img
        
    # def render_img(self):
    #     bg_img=copy.deepcopy(self.background_img)
    #     # print(type(bg_img))
    #     # cv2.circle()
    #     cv2.circle(bg_img,(self.img_size[0]-int(self.info["ball_site"][1]),int(self.info["ball_site"][0])),radius=8,color=(100,100,100),thickness=-1)
    #     cv2.imshow("render_cv",bg_img)
        

    def render_img(self):
        # print(dir(self.world.bodies[0]))
        # print(dir(self.screen.get_buffer()))
        # print(self.world.bodies)
        tt=timer_tool("env render")
        tt.start()

        self.screen.fill((0, 0, 0))
        self.renderer.StartDraw()
        tt.end_and_start("fill")
        self.world.renderer.flags = dict(drawShapes=True,
                drawJoints=True,
                # drawAABBs=True,
                # drawPairs=True,
                # drawCOMs=True,
                convertVertices=True,
                )
        import time
        a=time.time()
        self.world.DrawDebugData()
        print("debug draw",time.time()-a)
        tt.end_and_start("drawdebug")
        # buffer_raw=self.screen.get_buffer().raw
        # image_buffer=np.frombuffer(buffer_raw,dtype=np.uint8)
        # image_buffer=np.reshape(image_buffer,[600,600,4])[:,:,:3]
        # image_buffer=np.transpose(image_buffer,[1,0,2])
        import time
        
        cv_img=self.renderer.surface

        # cv_img=pygame.surfarray.array3d(self.screen)
        # cv_img=np.transpose(cv_img,[1,0,2])
        tt.end_and_start("get array")
        # cv_img=np.transpose(cv_img,[1,0,2])
        # self.cv_img=cv_img
        return cv_img

    def render(self):
        # self.screen.fill((0, 0, 0))
        # self.renderer.StartDraw()
        # self.world.renderer.flags = dict(drawShapes=True,
        #         drawJoints=True,
        #         # drawAABBs=True,
        #         # drawPairs=True,
        #         # drawCOMs=True,
        #         convertVertices=True,
        #         )
        # self.world.DrawDebugData()
        # print(self.target_pos)
        # self.renderer.DrawPoint([self.target_pos[0]*self.img_size[0],(1-self.target_pos[1])*self.img_size[1]],self.block_size[0]/2,b2Color([0.8,0,0.8]))
        # # self.renderer.DrawPoint([goal_pos[0]*self.img_size[0],(1-goal_pos[1])*self.img_size[1]],self.block_size[0]/2,b2Color([0,0.8,0.8]))
        # # pygame.display.flip()
        cv_img=self.render_img()
        cv2.circle(cv_img,(self.img_size[0]-int(self.target_pos[1]*self.img_size[1]),int(self.target_pos[0]*self.img_size[0])),radius=8,color=(0,180,0),thickness=-1)
        # cv_img=np.transpose(cv_img,[1,0,2])
        real_goal_site=self.info["real_goal_site"]
        cv2.circle(cv_img,(int(self.img_size[0]-real_goal_site[1]),int(real_goal_site[0])),radius=20,color=(0,200,0),thickness=3)

        # ball_site=self.info["ball_site"]
        # ball_site_norm=self.info["ball_site_norm"]
        # real_goal_site_norm=self.info["real_goal_site_norm"]
        # heat_map=np.zeros(self.grid_size,dtype=np.float32)
        # ball_block_site=(ball_site_norm*self.grid_size).astype(np.int)
        # goal_block_site=(real_goal_site_norm*self.grid_size).astype(np.int)
        # heat_map[ball_block_site[0]][self.grid_size[1]-ball_block_site[1]]+=0.5
        # heat_map[goal_block_site[0]][self.grid_size[1]-goal_block_site[1]]+=1.0
        # heat_map=np.expand_dims(heat_map,axis=-1)

        
        # print(np.max(self.all_heat_map[:,:,2]))


        cv2.imshow("cv img",cv_img)
        cv2.imshow("ball img",self.ball_obs_for_show)
        # cv2.resize()
        cv2.imshow("ball_obs",cv2.resize(cv2.resize(self.ball_obs_for_show,(16,16),interpolation=cv2.INTER_AREA),(256,256),))

        cv2.imshow("heat_map",cv2.resize(self.all_heat_map[:,:,:3],dsize=(240,240),interpolation=cv2.INTER_AREA))
        cv2.imshow("life_heat",cv2.resize(self.all_heat_map[:,:,-1]*50,dsize=(240,240),interpolation=cv2.INTER_AREA))
        
        cv2.waitKey(20)
        print(np.max(self.life_heat_map_img))




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
        # print(*target_v)
        self.ball.linearVelocity=b2Vec2(float(target_v[0]),float(target_v[1]))
    def _get_position(self,x):
        return list(x.position)
    def _get_ball_site(self):
        return self._get_position(self.ball)
    def _get_goal_site(self):
        return [self._get_position(goal) for goal in self.goal_list]
    def _update_target_pos(self,target_pos):
        gamma=0.5
        self.target_pos=self.target_pos*gamma+target_pos*(1-gamma)


        
class four_goal_env(fourroom_env):
    def reset(self,):
        self.world = b2World(gravity=(0, 0), doSleep=True)
        # if self.show:
        self.screen = pygame.display.set_mode(self.img_size)
        self.renderer=PygameDraw(surface=self.screen)
        self.world.renderer=self.renderer
        self.clock = pygame.time.Clock()

        "draw wall"
        door_list=[[10,5],[10,15],[5,10],[15,10]]
        wall_list=[]

        for x in range(20):
            for y in [0,19]:
                if not [x,y] in wall_list:
                    wall_list.append([x,y])
        for y in range(20):
            for x in [0,19]:
                if not [x,y] in wall_list:
                    wall_list.append([x,y])
        for wall in wall_list:
            if not wall in door_list:
                self._add_wall(grid_y=wall[1],grid_x=wall[0])

        

        "set_goal"
        self.goal_list=[]


        goal_site=[3,3]
        goal=self._add_goal(goal_site[1],goal_site[0])
        self.goal_list.append(goal)
        goal_site=[16,16]
        goal=self._add_goal(goal_site[1],goal_site[0])
        self.goal_list.append(goal)
        goal_site=[3,16]
        goal=self._add_goal(goal_site[1],goal_site[0])
        self.goal_list.append(goal)
        goal_site=[16,3]
        goal=self._add_goal(goal_site[1],goal_site[0])
        self.goal_list.append(goal)

        self.real_goal_index=np.random.randint(0,4)    
        "bg_img"
        self.background_img=self.render_img()   
        self.background_img=np.ascontiguousarray(self.background_img)
        # print(self.background_img)

        "ep_heat_map_img"
        self.ep_heat_map_img=np.zeros([24,24],dtype=np.float32)

        "set ball"
        while 1:
            ball_site=list(np.random.randint(1,19,[2]))
            if ball_site in wall_list:
                continue
            else:
                self.ball=self._add_ball(ball_site[1],ball_site[0])
                break
        "init count"
        self.steps_num=0

        self.target_pos=np.array([0.5,0.5])

        self.get_info()
        obs=self.get_obs()
        return obs
    def reward_func(self):
        reward=0
        if np.linalg.norm(self.info['ball_site_norm']-self.info['real_goal_site_norm'])<=0.05:
            reward=1
            done=True
        else:
            for i in range(len(self.goal_list)):
                if i==self.real_goal_index:
                    continue
                elif np.linalg.norm(self.info['ball_site_norm']-self.info['goal_site_norm_list'][i])<=0.05:
                        reward=-0.05
                else:
                    pass
            done=False or self.steps_num>=self.max_steps_num

        ball_site_norm=self.info["ball_site_norm"]
        ball_block_site=(ball_site_norm*self.grid_size).astype(np.int)
        reward+=(1-self.heat_map_std_func(self.ep_heat_map_img[ball_block_site[0]][self.grid_size[1]-ball_block_site[1]]))*0.001
        reward+=(1-self.life_heat_map_img[ball_block_site[0]][self.grid_size[1]-ball_block_site[1]]*100)*0.001
        return reward,done

    def heat_map_std_func(self,x):
        return np.tanh(0.2*x)
    def get_obs(self):
        cv_img=self.render_img_cv()
        
        
        ball_site=self.info["ball_site"]
        ball_site_norm=self.info["ball_site_norm"]
        real_goal_site_norm=self.info["real_goal_site_norm"]
        ball_obs=self._crop_img(cv_img,np.array([self.img_size[1]-ball_site[1],ball_site[0]],dtype=np.int),[256,256])
        self.ball_obs_for_show=ball_obs
        ball_obs=cv2.resize(ball_obs,(16,16),interpolation=cv2.INTER_AREA)
        ball_obs=np.transpose(ball_obs,[2,0,1]).astype(np.float32)/255


        reward_map=np.zeros([24,24],dtype=np.float32)
        site_map=np.zeros([24,24],dtype=np.float32)

        ball_block_site=(ball_site_norm*self.grid_size).astype(np.int)
        goal_block_site=(real_goal_site_norm*self.grid_size).astype(np.int)

        site_map[ball_block_site[0]][self.grid_size[1]-ball_block_site[1]]+=1.0
        reward_map[goal_block_site[0]][self.grid_size[1]-goal_block_site[1]]+=1.0

        self.ep_heat_map_img[ball_block_site[0]][self.grid_size[1]-ball_block_site[1]]+=1

        gamma_life=0.999
        self.life_heat_map_img=self.life_heat_map_img*gamma_life
        self.life_heat_map_img[ball_block_site[0]][self.grid_size[1]-ball_block_site[1]]+=(1-gamma_life)
        
        all_heat_map=np.concatenate([
            np.expand_dims(reward_map,axis=-1),
            np.expand_dims(site_map,axis=-1),
            np.expand_dims(self.heat_map_std_func(self.ep_heat_map_img),axis=-1),
            np.expand_dims(self.life_heat_map_img,axis=-1)*50
            ],
            axis=-1
        )
        self.all_heat_map=all_heat_map
        all_heat_map_transpose=np.transpose(all_heat_map,[2,0,1])


        # heat_map=cv2.resize(heat_map,(16,16))
        # print(np.shape(np.expand_dims(heat_map,0)),np.shape(ball_obs))


        
        # goal_site=self._get_goal_site()
        # goal_site_norm=np.array(goal_site)/self.img_size
        # cv2.imshow("img",cv_img)
        # cv2.imshow("crop_img",ball_obs)
        # cv2.waitKey()

        return {"ball_site":ball_site_norm,"goal_site":real_goal_site_norm,"ball_obs":all_heat_map_transpose,"heat_map":all_heat_map}



if __name__=="__main__":
    e=four_goal_env()
    # e=fourroom_env()
    e.reset()
    for i in range(100000):
        obs,reward,done=e.step(np.random.uniform(-1,1,[2]),[0.5,0.5])
        e.render()
        # e.render_img_cv()
        
        if reward!=0:
            pass
            # print(reward,e.info)
            # cv2.waitKey()
        # print(i)