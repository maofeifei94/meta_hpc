import imp
import paddle
import argparse
from collections import deque
import paddle.nn as nn
import paddle.nn.functional as F
import os
import copy
import cv2
import gym
import numpy as np
import sys
import parl
import threading
# from attr import validate
# from scipy.fft import hfft2
import hyperparam as hp
import time
from paddle.jit import to_static
import parl
import paddle
import paddle.optimizer as optim
import paddle.nn as nn
from paddle.distribution import Normal
import numpy as np
from myf_ML_util import timer_tool




view_shape=[2,16,16]
image_shape=[3,64,64]

r_shape=[256,16,16]

h_shape=[128,16,16]
u_shape=[128,64,64]
z_shape=[4,16,16]

generator_depth=5

def nl_func():
    return nn.LeakyReLU(negative_slope=0.1)
class ConvGru(nn.Layer):
    def __init__(self,input_channels,hid_channels) -> None:
        super().__init__()
        self.hid_channels=hid_channels
        self.conv_zt=nn.Conv2D(hid_channels+input_channels,hid_channels,[5,5],[1,1],padding='SAME')
        self.conv_rt=nn.Conv2D(hid_channels+input_channels,hid_channels,[5,5],[1,1],padding='SAME')
        self.conv_ht_middle=nn.Conv2D(hid_channels+input_channels,hid_channels,[5,5],[1,1],padding='SAME')
        pass

    def forward(self,xt,ht_1):
        "[b,c,h,w]"
        ht_xt=paddle.concat([ht_1,xt],axis=1)
        zt=F.sigmoid(self.conv_zt(ht_xt))
        rt=F.sigmoid(self.conv_rt(ht_xt))
        ht_middle=F.tanh(self.conv_ht_middle(paddle.concat([rt*ht_1,xt],axis=1)))
        ht=(1-zt)*ht_1+zt*ht_middle
        return ht

class GenerationCore(nn.Layer):
    def __init__(self,input_channels,hid_channels):
        super().__init__()
        self.gru=ConvGru(input_channels,hid_channels)
        self.h_to_u=nn.Conv2DTranspose(hid_channels,u_shape[0],[4,4],[4,4],padding='SAME')
    def forward(self,core_input_dict):
        representation=core_input_dict["representation"]
        target_view=core_input_dict["target_view"]

        gen_gru_h_pre_layer=core_input_dict["gen_gru_h_pre_layer"]
        gen_u_pre_layer=core_input_dict["gen_u_pre_layer"]

        z_pre_layer=core_input_dict["z_pre_layer"]

        # run gru
        # print(representation.shape,target_view.shape,z_pre_layer.shape)
        gru_input=paddle.concat([representation,target_view,z_pre_layer],axis=1)
        h_t=self.gru(gru_input,gen_gru_h_pre_layer)

        u_t=gen_u_pre_layer+self.h_to_u(h_t)

        output_dict={
            "gen_gru_h":h_t,
            "gen_u":u_t,
        }
        return output_dict

class InferenceCore(nn.Layer):
    def __init__(self,input_channels,hid_channels) -> None:
        super().__init__()
        self.gru=ConvGru(input_channels,hid_channels)
        self.u_to_core=nn.Conv2D(u_shape[0],u_shape[0],[4,4],[4,4],padding="SAME")
        self.img_to_core=nn.Conv2D(image_shape[0],image_shape[0],[4,4],[4,4],padding="SAME")
    def forward(self,core_input_dict):
        representation=core_input_dict["representation"]
        target_view=core_input_dict["target_view"]
        target_img=core_input_dict["target_img"]

        gen_gru_h_pre_layer=core_input_dict["gen_gru_h_pre_layer"]
        gen_u_pre_layer=core_input_dict["gen_u_pre_layer"]

        inf_gru_h_pre_layer=core_input_dict["inf_gru_h_pre_layer"]

        # run gru
        # print(representation,target_view,target_img,gen_gru_h_pre_layer,self.u_to_core(gen_u_pre_layer))
        gru_input=paddle.concat([
            representation,
            target_view,
            self.img_to_core(target_img),
            gen_gru_h_pre_layer,
            self.u_to_core(gen_u_pre_layer)
            ],axis=1)
        h_t=self.gru(gru_input,inf_gru_h_pre_layer)

        output_dict={
            "inf_gru_h":h_t,
        }
        return output_dict

class RepresentationNet(nn.Layer):
    def __init__(self) -> None:
        super().__init__()
        #before concat v
        #[3,64,64]
        self.conv1=nn.Sequential(nn.Conv2D(image_shape[0],256,[2,2],[2,2],padding='SAME'),nl_func())
        #[256,32,32]
        self.conv2=nn.Sequential(nn.Conv2D(256,256,[3,3],[1,1],padding='SAME'),nl_func())
        #[128,32,32]
        self.conv3=nn.Sequential(nn.Conv2D(256,256,[2,2],[2,2],padding='SAME'),nl_func())
        #[256,16,16]
        #concat[2,16,16]
        self.conv4=nn.Sequential(nn.Conv2D(256+view_shape[0],256,[3,3],[1,1],padding='SAME'),nl_func())
        #[128,16,16]
        self.conv5=nn.Sequential(nn.Conv2D(256,256,[3,3],[1,1],padding='SAME'),nl_func())
        #[256,16,16]
        self.conv6=nn.Sequential(nn.Conv2D(256,256,[1,1],[1,1],padding='SAME'),nl_func())
        #[256,16,16]
    def forward(self,img,view):
        x=self.conv1(img)
        # print(1,x.shape)
        x=self.conv2(x)+x #残差连接(skip connection)
        # print(2,x.shape)
        x=self.conv3(x)
        # print(3,x.shape)
        x=paddle.concat([x,view],axis=1)
        # print(4,x.shape)
        x=self.conv4(x) 
        # print(5,x.shape)
        x=self.conv5(x)+x #残差连接(skip connection)
        # print(6,x.shape)
        x=self.conv6(x)
        # print(7,x.shape)
        return x
#重参数技巧的Normal分布
#标准差 std,sigma,scale
#方差   var,sigma**2，
class ReparamNormal():
    def __init__(self,disshape) -> None:
        self.normal_dis=paddle.distribution.Normal(paddle.zeros(disshape),paddle.ones(disshape))
        pass
    def sample(self,mean,log_var):
        std=(log_var/2).exp()
        return self.normal_dis.sample([1])*std+mean
        
class gqnmodel(nn.Layer):

    def __init__(self) -> None:
        super().__init__()

        

        "rnet"
        self.rnet=RepresentationNet()

        "gen inf core"
        self.generator_depth=generator_depth
        self.generator_core_list=[]
        self.inference_core_list=[]
        for i in range(self.generator_depth):
            gen_i=GenerationCore(r_shape[0]+view_shape[0]+z_shape[0],h_shape[0])
            inf_i=InferenceCore(r_shape[0]+view_shape[0]+image_shape[0]+h_shape[0]+u_shape[0],h_shape[0])
            self.generator_core_list.append(gen_i)
            self.inference_core_list.append(inf_i)
            setattr(self,f'gen_{i}',gen_i)
            setattr(self,f'inf_{i}',inf_i)

        
        "gen_h_to_u"
        self.gen_h_to_u_net=nn.Conv2DTranspose(
            in_channels=h_shape[0],out_channels=u_shape[0],
            kernel_size=[4,4],stride=[4,4],padding="SAME"
        )
        "gen_h_to_z_dis"
        self.gen_h_to_z_dis_net=nn.Conv2DTranspose(
            in_channels=h_shape[0],out_channels=z_shape[0]*2,
            kernel_size=[5,5],stride=[1,1],padding="SAME"
        )
        "inf_h_to_z_dis"
        self.inf_h_to_z_dis_net=nn.Conv2DTranspose(
            in_channels=h_shape[0],out_channels=z_shape[0]*2,
            kernel_size=[5,5],stride=[1,1],padding="SAME"
        )

        "u_to_recon_img"
        self.u_to_recon_img_net=nn.Sequential(nn.Conv2D(
            u_shape[0],image_shape[0],
            [1,1],[1,1],padding="SAME"),
            nn.Sigmoid())
        "sample z"
        self.repa_normal=ReparamNormal(z_shape)


        "all_param"
        self.all_param=[]

        self.all_param+=self.rnet.parameters()
        for i,gen_i in enumerate(self.generator_core_list):
            self.all_param+=gen_i.parameters()
        for i,inf_i in enumerate(self.inference_core_list):
            self.all_param+=inf_i.parameters()
        self.all_param+=self.gen_h_to_u_net.parameters()
        self.all_param+=self.gen_h_to_z_dis_net.parameters()
        self.all_param+=self.inf_h_to_z_dis_net.parameters()
        self.all_param+=self.u_to_recon_img_net.parameters()


        "optimizer"
        self.optimizer=optim.Adam(1e-4,parameters=self.all_param)

        # "all net"
        # self.net_dict={}
        # self.net_dict['rnet']=self.rnet
        # for i in range(len(self.generator_core_list)):
        #     gen_i=self.generator_core_list[i]
        #     inf_i=self.inference_core_list[i]
        #     self.net_dict[f'gen_{i}']=gen_i
        #     self.net_dict[f'inf_{i}']=inf_i

        # self.net_dict['gen_h_to_u_net']=self.gen_h_to_u_net
        # self.net_dict['gen_h_to_z_dis_net']=self.gen_h_to_z_dis_net
        # self.net_dict['inf_h_to_z_dis_net']=self.inf_h_to_z_dis_net
        # self.net_dict['u_to_recon_img_net']=self.u_to_recon_img_net
        # self.net_dict={
        #     'rnet':self.rnet
        # }

    def get_pixel_std(self,steps):
        std_i=2.0
        std_f=0.7
        n=2e5
        return max(std_f+(std_i-std_f)*(1-steps/n),std_f)
    def get_core_init_state_dict(self,target_batch_size):
        init_state_dict={
            "gen_gru_h_pre_layer":paddle.zeros([target_batch_size,*h_shape]),
            "gen_u_pre_layer":paddle.zeros([target_batch_size,*u_shape]),
            "inf_gru_h_pre_layer":paddle.zeros([target_batch_size,*h_shape]),
            "z_pre_layer":paddle.zeros([target_batch_size,*z_shape]),
        }
        return init_state_dict



    def get_presentation(self,history_img_batch,history_view_batch):
        #img_batch[b,3,64,64]
        #viewbatch[b,2]
        # print(history_view_batch.shape)
        # history_view_batch=paddle.tile(paddle.reshape(history_view_batch,[-1,view_shape[0],1,1]),[1,1,16,16])
        # print(history_view_batch.shape)
        r_batch=self.rnet(history_img_batch,history_view_batch)
        r=paddle.mean(r_batch,axis=0,keepdim=True)
        return r
    def get_z_param_from_h(self,gen_h,dis_net):
        z_params=dis_net(gen_h)
        z_mean=z_params[:,:z_shape[0]]
        z_log_var=z_params[:,z_shape[0]:]
        return z_mean,z_log_var
    
    def pred_for_train(self,train_input_dict):
        history_img_batch=train_input_dict['history_img_batch']
        history_view_batch=train_input_dict['history_view_batch']

        target_img=train_input_dict['target_img']
        target_view=train_input_dict['target_view']

        "get r"
        r=self.get_presentation(history_img_batch,history_view_batch)
        r=paddle.tile(r,[target_img.shape[0],1,1,1])


        "init input"
        core_init_state_dict=self.get_core_init_state_dict(target_img.shape[0])
        core_input_dict={
            "representation":r,
            "target_view":target_view,
            "target_img":target_img,


            "gen_gru_h_pre_layer":core_init_state_dict["gen_gru_h_pre_layer"],
            "gen_u_pre_layer":core_init_state_dict["gen_u_pre_layer"],
            "inf_gru_h_pre_layer":core_init_state_dict["inf_gru_h_pre_layer"],
            "z_pre_layer":core_init_state_dict["z_pre_layer"],
        }

        gen_z_param_list=[]
        inf_z_param_list=[]
        for i in range(self.generator_depth):
            "select core"
            gen_core_i=self.generator_core_list[i]
            inf_core_i=self.inference_core_list[i]

            "run core"
            gen_core_i_output=gen_core_i(core_input_dict)
            inf_core_i_output=inf_core_i(core_input_dict)

            "get z_param"
            gen_z_mean,gen_z_log_var=self.get_z_param_from_h(gen_core_i_output['gen_gru_h'],self.gen_h_to_z_dis_net)
            inf_z_mean,inf_z_log_var=self.get_z_param_from_h(inf_core_i_output['inf_gru_h'],self.inf_h_to_z_dis_net)
            inf_z_log_var=paddle.zeros_like(inf_z_log_var)
            gen_z_param_list.append([gen_z_mean,gen_z_log_var])
            inf_z_param_list.append([inf_z_mean,inf_z_log_var])
            z=self.repa_normal.sample(inf_z_mean,inf_z_log_var)
            
            "updata core_input"
            core_input_dict['gen_gru_h_pre_layer']=gen_core_i_output['gen_gru_h']
            core_input_dict['gen_u_pre_layer']=gen_core_i_output['gen_u']
            core_input_dict['inf_gru_h_pre_layer']=inf_core_i_output['inf_gru_h']
            core_input_dict['z_pre_layer']=z

        final_u=gen_core_i_output['gen_u']

        recon_img=self.u_to_recon_img_net(final_u)
        train_pred_dict={
            "final_u":final_u,
            "recon_img":recon_img,
            "gen_z_param_list":gen_z_param_list,
            "inf_z_param_list":inf_z_param_list,

        }
        return train_pred_dict
    def pred_for_test(self,test_input_dict):
        history_img_batch=test_input_dict['history_img_batch']
        history_view_batch=test_input_dict['history_view_batch']

        # target_img=test_input_dict['target_img']
        target_view=test_input_dict['target_view']

        "get r"
        r=self.get_presentation(history_img_batch,history_view_batch)
        r=paddle.tile(r,[target_view.shape[0],1,1,1])


        "init input"
        core_init_state_dict=self.get_core_init_state_dict(target_view.shape[0])
        core_input_dict={
            "representation":r,
            "target_view":target_view,
            # "target_img":target_img,

            "gen_gru_h_pre_layer":core_init_state_dict["gen_gru_h_pre_layer"],
            "gen_u_pre_layer":core_init_state_dict["gen_u_pre_layer"],
            # "inf_gru_h_pre_layer":core_init_state_dict["inf_gru_h_pre_layer"],
            "z_pre_layer":core_init_state_dict["z_pre_layer"],
        }

        gen_z_param_list=[]
        # inf_z_param_list=[]
        for i in range(self.generator_depth):
            "select core"
            gen_core_i=self.generator_core_list[i]
            # inf_core_i=self.inference_core_list[i]

            "run core"
            gen_core_i_output=gen_core_i(core_input_dict)
            # inf_core_i_output=inf_core_i(core_input_dict)

            "get z_param"
            gen_z_mean,gen_z_log_var=self.get_z_param_from_h(gen_core_i_output['gen_gru_h'],self.gen_h_to_z_dis_net)
            
            gen_z_param_list.append([gen_z_mean,gen_z_log_var])

            z=self.repa_normal.sample(gen_z_mean,gen_z_log_var)
            
            "updata core_input"
            core_input_dict['gen_gru_h_pre_layer']=gen_core_i_output['gen_gru_h']
            core_input_dict['gen_u_pre_layer']=gen_core_i_output['gen_u']
            # core_input_dict['inf_gru_h_pre_layer']=inf_core_i_output['inf_gru_h']
            core_input_dict['z_pre_layer']=z

        final_u=gen_core_i_output['gen_u']

        recon_img=self.u_to_recon_img_net(final_u)
        test_pred_dict={
            "final_u":final_u,
            "recon_img":recon_img,
            "gen_z_param_list":gen_z_param_list,
            # "inf_z_param_list":inf_z_param_list,
        }
        return test_pred_dict
    def train_one_step(self,train_data,steps):
        train_input_dict=train_data

        train_pred_dict=self.pred_for_train(train_input_dict)
        final_u=train_pred_dict["final_u"]
        recon_img=train_pred_dict["recon_img"]
        gen_z_param_list=train_pred_dict["gen_z_param_list"]
        inf_z_param_list=train_pred_dict["inf_z_param_list"]

        # kl loss
        total_kl_loss=0
        for gen_z_param,inf_z_param in zip(gen_z_param_list,inf_z_param_list):
            gen_z_mean,gen_z_log_var=gen_z_param
            inf_z_mean,inf_z_log_var=inf_z_param
            gen_z_dis=paddle.distribution.Normal(gen_z_mean,paddle.exp(gen_z_log_var/2))
            inf_z_dis=paddle.distribution.Normal(inf_z_mean,paddle.exp(inf_z_log_var/2))
            kl_loss_i=inf_z_dis.kl_divergence(gen_z_dis)
            # print("kl_loss_i.shape",kl_loss_i.shape)
            total_kl_loss+=kl_loss_i
        kl_loss=paddle.mean(total_kl_loss)/len(gen_z_param_list)

        # recon_loss=paddle.sum((recon_img-train_input_dict['target_img'])**2)
        pixel_std=self.get_pixel_std(steps)
        recon_normal=Normal(recon_img,paddle.full_like(recon_img,pixel_std))
        recon_loss=-paddle.mean(recon_normal.log_prob(train_input_dict['target_img']))-np.log((2*np.pi*pixel_std**2)**0.5)
        # print(recon_loss.shape)

        total_loss=recon_loss+kl_loss

        # print(total_loss.numpy(),recon_loss.numpy(),kl_loss.numpy())


        self.optimizer.clear_grad()
        total_loss.backward()
        self.optimizer.step()
        return np.array([total_loss.numpy(),recon_loss.numpy(),kl_loss.numpy()]).reshape([-1])

        # return {
        #     'total_loss':total_loss.numpy(),
        #     "recon_loss":recon_loss.numpy(),
        #     "kl_loss":kl_loss.numpy()
        #     }
    def save_model(self,save_dir,iter_num):
        # print(self.state_dict)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        paddle.save(self.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")
        # for key in self.net_dict.keys():
        #     paddle.save(self.net_dict[key].state_dict(),f"{save_dir}/{iter_num}_{key}_model.pdparams")
    def load_model(self,save_dir,iter_num):
        self.set_state_dict(paddle.load(f"{save_dir}/{iter_num}_model.pdparams"))
        # self.net_dict[key].set_state_dict(paddle.load(path=f"{save_dir}/{iter_num}_{key}_model.pdparams"))


        

        













        




        