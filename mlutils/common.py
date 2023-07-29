try:
    import paddle
    import paddle.nn as nn
    import paddle.nn.functional as F
    import xlrd
except:
    pass
# import hyperparam as hp
import numpy as np
import time
import cv2
import os


debug=False


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Location():
    def in_location(self,location):
        """
        基础位置类
        """
        pass
class Point(Location):
    def __init__(self,y,x):
        """
        点位置
        """
        self.y=y
        self.x=x


        # self.point=[y,x]
    @property
    def slice_miny(self):
        return self.y
    @property
    def slice_minx(self):
        return self.x
    @property
    def slice_maxy(self):
        return self.y+1
    @property
    def slice_maxx(self):
        return self.x+1
    def __repr__(self):
        return f"Point y,x=[{self.y},{self.x}]"
    def move(self,move_y,move_x):
        self.y+=move_y
        self.x+=move_x
        return self
        # self.point=[self.y,self.x]
    def equal(self,other_point):
        return other_point.y==self.y and other_point.x==self.x
    def in_rect(self,rect):
        return self.y<=rect.maxy and self.y>=rect.miny and self.x<=rect.maxx and self.x>=rect.minx
    def in_location(self,location):
        if isinstance(location,Rect):
            return self.in_rect(location)
        elif isinstance(location,Point):
            return self.qeual(location)
    
class Rect(Location):
    def __init__(self,miny,minx,maxy,maxx) -> None:
        """
        rect包括边界点
        比如 miny=0,maxy=3
        则y方向边长包括0,1,2,3四个点
        """
        self.miny=miny
        self.minx=minx
        self.maxy=maxy
        self.maxx=maxx

    @property
    def slice_miny(self):
        return self.miny
    @property
    def slice_minx(self):
        return self.minx
    @property
    def slice_maxy(self):
        return self.maxy+1
    @property
    def slice_maxx(self):
        return self.maxx+1
    @property
    def height(self):
        return self.maxy-self.miny+1
    @property
    def width(self):
        return self.maxx-self.minx+1


    def __repr__(self):
        return f"Rect miny,minx,maxy,maxx=[{self.miny},{self.minx},{self.maxy},{self.maxx}]"
    def equal(self,rect):
        return self.miny==rect.miny and self.minx==rect.minx and self.maxy==rect.maxy and self.maxx==rect.maxx
    def contain_point(self,point:Point):
        if point.y<=self.maxy and point.y>=self.miny and point.x<=self.maxx and point.x>=self.minx:
            return True
        else:
            return False
    def in_location(self,location):
        if isinstance(location,Rect):
            return self.equal(location)
        else:
            return False
 

def location_in(x,location_list):
    """
    x:site
    site_rect_list:如果 x为site,list可包含site或rect
    如果 x为rect，list只能为rect
    """
    out=False
    if isinstance(x,Point):
        for location in location_list:
            if not isinstance(location,Location):
                raise ValueError(f"site_rect_in type error:{location} is type {type(location)} not Location")
            out=x.in_location(location) or out
    elif isinstance(x,Rect):
        for location in location_list:
            if not isinstance(location,Rect):
                raise ValueError(f"site_rect_in type error:{location} is type {type(location)} not Rect")
            out=x.in_location(location) or out
    return out