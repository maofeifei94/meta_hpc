

from audioop import mul
# from statistics import multimode
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle
import copy
import os
import cv2
import numpy as np
from myf_ML_util import Basic_Module,moving_avg
from hpc import V_Auto_Encoder,Place_imging_module,Reward_HPC_Module
from env.env import multi_circle_env

def nl_func():
    return nn.LeakyReLU(negative_slope=0.1)
class Reslayer(nn.Layer):
    def __init__(self,dim):
        super(Reslayer,self).__init__()
        self.layer=nn.Sequential(nn.Linear(dim,dim),nn.Tanh())
    def forward(self,x):
        return x+self.layer(x)

class fakepfc():
    def __init__(self,) -> None:
        self.net=nn.Sequential(
            nn.Linear(256+2,256),nn.Tanh(),
            nn.Linear(256,256),nn.Tanh(),
            nn.Linear(256,128),nn.Tanh(),
            nn.Linear(128,64),nn.Tanh(),
            nn.Linear(64,1)
        )
        self.optimizer=optim.Adam(learning_rate=0.0003,parameters=self.net.parameters())
        self.avg_loss=moving_avg()
    def train(self,train_data,train_label,vae:V_Auto_Encoder,pim:Place_imging_module,rhm:Reward_HPC_Module):
        print(train_data.shape,train_label.shape)
        pred=self.net(train_data)
        loss=paddle.mean((pred-train_label)**2)
        print("pred abs mean",paddle.mean(paddle.abs(pred)))
        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()
        self.avg_loss.update(np.mean(loss.numpy()))
        return np.mean(loss.numpy())
class slpfc():
    def __init__(self,input_dim=259,gru_hid_dim=256) -> None:
        self.vae_hid_dim=256
        self.input_dim=input_dim
        self.gru_hid_dim=gru_hid_dim

        self.obs_to_gru=nn.Sequential(
            nn.Linear(input_dim,self.gru_hid_dim),nl_func(),
            Reslayer(self.gru_hid_dim),
            Reslayer(self.gru_hid_dim),
            Reslayer(self.gru_hid_dim),
            Reslayer(self.gru_hid_dim),
            Reslayer(self.gru_hid_dim)
        )
        self.gru=nn.GRU(input_size=self.gru_hid_dim,hidden_size=gru_hid_dim)

        self.gru_to_place=nn.Sequential(
            nn.Linear(self.gru_hid_dim,2),
            Reslayer(2),
            Reslayer(2),
            Reslayer(2),
            Reslayer(2),
            nn.Linear(2,2),nn.Sigmoid()
        )

        self.gru_to_answer_net=nn.Sequential(
            Reslayer(self.gru_hid_dim),
            Reslayer(self.gru_hid_dim),
            Reslayer(self.gru_hid_dim),
            Reslayer(self.gru_hid_dim),
            Reslayer(self.gru_hid_dim)
        )

        self.answer_net=nn.Sequential(
            nn.Linear(self.vae_hid_dim+2+self.gru_hid_dim,512),nl_func(),
            Reslayer(512),
            Reslayer(512),
            Reslayer(512),
            nn.Linear(512,1)
        )
        self.optimizer=optim.Adam(learning_rate=0.0003,parameters=[
            *self.obs_to_gru.parameters(),
            *self.gru.parameters(),
            *self.gru_to_place.parameters(),
            *self.gru_to_answer_net.parameters(),
            *self.answer_net.parameters()
            ]
        )
        self.avg_loss=moving_avg()

    def save_model(self,save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        net_list=[self.obs_to_gru,self.gru,self.gru_to_place,self.gru_to_answer_net,self.answer_net,self.optimizer]
        net_name_list=["obs_to_gru","gru","gru_to_place","gru_to_answer_net","answer_net","optim"]

        for net,name in zip(net_list,net_name_list):
            paddle.save(net.state_dict(),f"{save_dir}/newest_{name}_model.pdparams")

        
        
    def load_model(self,save_dir):
        net_list=[self.obs_to_gru,self.gru,self.gru_to_place,self.gru_to_answer_net,self.answer_net,self.optimizer]
        net_name_list=["obs_to_gru","gru","gru_to_place","gru_to_answer_net","answer_net","optim"]

        for net,name in zip(net_list,net_name_list):
            net.set_state_dict(paddle.load(path=f"{save_dir}/newest_{name}_model.pdparams"))
    def train(self,train_data,train_label,vae:V_Auto_Encoder,pim:Place_imging_module,rhm:Reward_HPC_Module):

        vec_to_gru=paddle.zeros([1,self.input_dim])
        h=paddle.zeros([1,1,self.gru_hid_dim],'float32')

        place_list=[]

        

        for i in range(50):
            gru_input=self.obs_to_gru(vec_to_gru)
            gru_input=paddle.reshape(gru_input,[1,1,-1])

            gru_out,h=self.gru(gru_input,h)
            gru_out=paddle.reshape(gru_out,[1,-1])

            ask_place=self.gru_to_place(gru_out)
            place_list.append(ask_place)

            
            pred_vae_hid=pim.input(ask_place)

            pred_reward=rhm.input(paddle.concat([ask_place,pred_vae_hid],axis=-1))

            vec_to_gru=paddle.concat([ask_place,pred_vae_hid,pred_reward],axis=-1)

        train_batch_size,_=train_data.shape

        h_batch=paddle.tile(paddle.reshape(h,[1,-1]),[train_batch_size,1])

        answer=self.answer_net(paddle.concat([train_data,h_batch],axis=-1))

        # print(answer,train_label)
        # all_place=paddle.concat(place_list,axis=0)
        # place1=paddle.reshape(all_place,[1,-1,2])
        # place2=paddle.reshape(all_place,[-1,1,2])
        # psub=paddle.sum((place1-place2)**2,axis=-1)
        # loss_place=-paddle.mean(psub)
        # print("place_loss",loss_place.numpy())

        loss=paddle.mean(paddle.abs(train_label-answer))

        all_loss=loss

        self.optimizer.clear_grad()
        all_loss.backward()
        self.optimizer.step()

        self.avg_loss.update(np.mean(loss.numpy()))
        return np.mean(loss.numpy())
    def test(self,train_data,train_label,vae:V_Auto_Encoder,pim:Place_imging_module,rhm:Reward_HPC_Module,_env:multi_circle_env):
        vec_to_gru=paddle.zeros([1,self.input_dim])
        h=paddle.zeros([1,1,self.gru_hid_dim],'float32')

        test_img=np.zeros([600,600,3],dtype=np.uint8)

        for goal_site in _env.goal_site_list:
            cv2.circle(test_img,(int((goal_site[1]+0.5)*30),int((goal_site[0]+0.5)*30)),15,(150,150,150),-1)


        for i in range(50):
            gru_input=self.obs_to_gru(vec_to_gru)
            gru_input=paddle.reshape(gru_input,[1,1,-1])

            gru_out,h=self.gru(gru_input,h)
            gru_out=paddle.reshape(gru_out,[1,-1])

            ask_place=self.gru_to_place(gru_out)

            
            pred_vae_hid=pim.input(ask_place)

            pred_reward=rhm.input(paddle.concat([ask_place,pred_vae_hid],axis=-1))

            vec_to_gru=paddle.concat([ask_place,pred_vae_hid,pred_reward],axis=-1)

            recon_img=vae.decode(pred_vae_hid).numpy()

            # cv2.circle()
            print(i,ask_place.numpy())
            show_img=np.transpose(recon_img[0],[1,2,0])
            print(np.shape(show_img))

            ball_place=ask_place.numpy()[0]
            show_bg_img=copy.deepcopy(test_img)

            cv2.rectangle(show_bg_img,(int(ball_place[0]*600-64),int(ball_place[1]*600-64)),(int(ball_place[0]*600+64),int(ball_place[1]*600+64)),(200,100,100),3)
            cv2.circle(test_img,(int(ball_place[0]*600),int(ball_place[1]*600)),5,(0,0,255),-1)
            target_site=ball_place
            obs,reward,done=_env.step(None,place=target_site,hpc_train_mode=True)
            ball_img=obs["ball_obs"]
            cv2.imshow("ball_img",ball_img)
            cv2.imshow("pfc test",cv2.resize(show_img,(128,128)))
            cv2.imshow("background_img",show_bg_img)
            cv2.waitKey()

            

