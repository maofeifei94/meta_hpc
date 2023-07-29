import os
import gym
import math
import random
import numpy as np
import sys
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
class show_img_class():
    def __init__(self,use_plt,show):
        self.use_plt_show=use_plt
        self.show=show
        pass

    def get_site(self,bar_size,value_range,number,x):
        site_x=int((x - value_range[0]) / (value_range[1] - value_range[0]) * bar_size[1])
        site_y=int(number*bar_size[0]+bar_size[0]/2)
        return (site_x,site_y)
    def show_avg_fire_move(self,avg_fire,train_mean_fire,sparsity,window_name):
        bar_size=[30,1000]
        img_size=[bar_size[0]*len(avg_fire),bar_size[1]]
        value_range=[0.03,0.07]
        avg_color = (200, 200, 0)
        train_mean_color = (0, 200, 200)
        radiu=int(bar_size[0]/2*0.7)

        img=np.zeros([*img_size,3],dtype=np.uint8)
        for i in range(len(avg_fire)):
            af=avg_fire[i]
            tmf=train_mean_fire[i]
            # print(np.shape(af),np.shape(tmf))
            avg_site=self.get_site(bar_size,value_range,i,af)
            train_mean_site=self.get_site(bar_size,value_range,i,tmf)
            cv2.circle(img,avg_site,radiu,avg_color,-1)
            cv2.circle(img,train_mean_site,radiu,train_mean_color,-1)

        line_x=int((sparsity - value_range[0]) / (value_range[1] - value_range[0]) * bar_size[1])
        cv2.line(img,(line_x,0),(line_x,img_size[0]),(200,0,0),3)
        cv2.imshow(window_name,img)
    def show_mnist_img(self,data,show_mat=[4,4],name="mnist",ratio=4.0):
        show_data=data[:np.prod(show_mat)]
        img_list=[]

        show_img=np.reshape(
            np.transpose(
                np.reshape(
                    show_data,[show_mat[0],show_mat[1],28,28]
                ),[0,2,1,3]),[show_mat[0]*28,show_mat[1]*28]
        )
        show_img=cv2.resize(show_img,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
        cv2.imshow(name,show_img)
        cv2.waitKey(1)




    def show_resize_img(self,data,size,ratio,name):
        img=np.reshape(data,size)
        resized_img=cv2.resize(img,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_NEAREST)
        if self.use_plt_show:
            plt.imshow(resized_img,"gray")
            plt.show()
        else:
            cv2.imshow(name,resized_img)
    def show_hist(self,data,window_name):
        data=np.reshape(data,[-1])
        fig = plt.figure()
        plt.hist(data,bins=100)
        if self.use_plt_show:
            plt.show()
            plt.close(fig)
        else:
            image = self.fig2data(fig)
            if self.show:
                cv2.imshow(window_name, image)
                cv2.waitKey(1)
            plt.close(fig)
            return image
    def show_hist_2d(self,data,window_name,draw_circle_site=None):
        hist_range=[[-10,10],[-10,10]]
        # hist_range=[[-20,20],[-20,20]]
        fig = plt.figure()
        hist_img=plt.hist2d(data[0,:], data[1,:], bins=60,range=hist_range)
        # print(np.shape(data))
        # print(np.sum(hist_img[0]))
        if self.use_plt_show:
            plt.show()
            plt.close(fig)
        else:

            image = self.fig2data(fig)

            if type(draw_circle_site)!=type(None):
                # draw_circle_site=[-5,-5]
                # print(np.shape(image))
                lu_site=(80,60)
                rd_site=(576,427)

                c_x,c_y=draw_circle_site[1],draw_circle_site[0]
                circle_site=(int((c_x+10)/20*(rd_site[0]-lu_site[0])+lu_site[0]),
                             int(-(c_y-10)/20*(rd_site[1]-lu_site[1])+lu_site[1]))
                cv2.line(image,(circle_site[0],lu_site[1]),(circle_site[0],rd_site[1]),(0,0,255),2)
                cv2.line(image, (lu_site[0], circle_site[1]), (rd_site[0], circle_site[1]), (0, 0, 255), 2)
                # cv2.circle(image,circle_site,3,(0,0,255),-1)

            cv2.imshow(window_name, image)
            cv2.waitKey(1)
            plt.close(fig)
    def show_img_3d(self,data=None,window_name="distribution",bins=50):
        img_data = plt.hist2d(data[:, 0], data[:, 1], bins=bins)
        # print(np.shape(img_data[0]))
        # plt.imshow(img_data[0])
        # plt.show()
        Z,X,Y=img_data[0],img_data[1],img_data[2]
        X,Y=np.meshgrid(X[1:],Y[1:])
        # print(np.shape(X),np.shape(Y),np.shape(Z))
        fig = plt.figure()
        ax3 = fig.gca(projection='3d')
        ax3.plot_surface(X, Y, Z, cmap='rainbow')
        # ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
        # print(type(fig))
        # plt.show()
        image=self.fig2data(fig)
        cv2.imshow(window_name,image)
        # cv2.waitKey(1)
        plt.close(fig)
        # plt.show()
    def fig2data(self,fig):
        # draw the renderer
        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        image = np.asarray(image)
        # print(np.shape(image))
        image=np.concatenate([image[:,:,2:3],image[:,:,1:2],image[:,:,0:1]],axis=-1)
        return image