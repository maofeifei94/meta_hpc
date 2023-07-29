
import imp
from cv2 import waitKey
from vae.vae import V_Auto_Encoder
from env.env_war_frog_gqn import multi_circle_env
import numpy as np
import cv2
import os
import random
import paddle
from paddle import nn
from myf_ML_util import moving_avg
from hpc.sparse_ae import Sparse_Autoencoder
from matplotlib import pyplot as plt
def get_data():
    img_num=100
    batch_num=1000
    env=multi_circle_env()
    model=V_Auto_Encoder()
    model.load_model("all_models/vae_model",240000)

    all_z=[]
    with paddle.no_grad():
        for iter in range(batch_num):
            "get data"
            env.reset()
            img_batch=[]
            for i in range(img_num):
                rand_site=np.random.uniform(0,1,[2])
                env_info,reward,done=env.step(None,rand_site,True)

                img=env_info['ask_ball_obs']
                img_batch.append(img)
            
            img_batch=paddle.to_tensor(img_batch)
            img_batch=paddle.transpose(img_batch,perm=[0,3,1,2])
            z_batch=model.encode(img_batch)
            all_z.append(z_batch)
            # print(iter)
        all_z=paddle.concat(all_z,axis=0).detach()
    return all_z

def train_sae():
    data=get_data()
    data_size=data.shape[0]
    data=paddle.reshape(data,[data_size,-1])

    train_batch_size=1000

    sae=Sparse_Autoencoder(data.shape[1])

    npy_save_dir="all_models/sae_npy"
    npy_num=0
    while 1:
        if os.path.exists(f"{npy_save_dir}/loss{npy_num}.npy"):
            npy_num+=1
        else:
            break

    loss_npy_data=[]
    for i in range(1000000000+1):

        rand_slice=paddle.to_tensor(np.random.randint(0,data_size,[train_batch_size]))
        rand_data=paddle.gather(data,rand_slice)
        loss_list=sae.train_one_step(rand_data)
        if i%1000==0:
            loss_npy_data.append([i,loss_list])
            print(i,loss_list)
        if i%10000==0:
            sae.save_model("all_models/sae_model",i)
            
            np.save(f"{npy_save_dir}/loss{npy_num}.npy",loss_npy_data)

def train_sae_for_best_gamma(gamm_fast,gamma_slow):

    
    data=np.meshgrid(np.linspace(0,1,1000),np.linspace(0,1,1000))
    data=np.transpose(np.reshape(data,[2,-1]),[1,0]).astype(np.float32)
    data_size=data.shape[0]
    print(data)
    data=paddle.to_tensor(data)


    train_batch_size=1000

    sae=Sparse_Autoencoder(2,gamm_fast,gamma_slow)

    loss_npy_data=[]
    for i in range(100000+1):

        rand_slice=paddle.to_tensor(np.random.randint(0,data_size,[train_batch_size]))
        rand_data=paddle.gather(data,rand_slice)
        loss_list=sae.train_one_step(rand_data)
        if i%1000==0:
            loss_npy_data.append([i,loss_list])
            print(i,loss_list)
        if i%10000==0:
            sae.save_model("all_models/sae_model",i)
            npy_save_dir="all_models/sae_npy"
            np.save(f"{npy_save_dir}/{sae.avg_fire_gamma_slow}_{sae.avg_fire_gamma_fast}.npy",loss_npy_data)

def get_best_gamma():
    gamma_list=[0.9,0.99,0.999,0.9999]
    for i in range(len(gamma_list)):
        for j in range(i+1,len(gamma_list)):
            train_sae(gamma_list[i],gamma_list[j])

def train_sae_for_best_act(data,act,act_name,neuron_num):
    data_size=data.shape[0]
    data=paddle.reshape(data,[data_size,-1])

    train_batch_size=1000
    npy_save_dir="all_models/sae_npy"

    sae=Sparse_Autoencoder(data.shape[1],act,neuron_num)


    loss_npy_data=[]
    for i in range(100000+1):

        rand_slice=paddle.to_tensor(np.random.randint(0,data_size,[train_batch_size]))
        rand_data=paddle.gather(data,rand_slice)
        loss_list=sae.train_one_step(rand_data)
        if i%1000==0:
            loss_npy_data.append([i,loss_list])
            print(i,loss_list)
        if i%10000==0:
            sae.save_model("all_models/sae_model",i)
            np.save(f"{npy_save_dir}/{act_name}_{neuron_num}.npy",loss_npy_data)

def get_best_act():
    data=get_data()

    act_dict={
        "tanh":nn.Tanh(),
        "relu":nn.ReLU(),
        "leakyrelu":nn.LeakyReLU(0.1),
        "swish":nn.Swish(),
        "selu":nn.SELU(),
        "gelu":nn.GELU(),
    }

    neuron_num_list=[2000,1000,3000]
    for neuron_num in neuron_num_list:
        for key in act_dict.keys():
            print(f"train {key} {neuron_num}")
            train_sae_for_best_act(data,act_dict[key],key,neuron_num)

    
if __name__=="__main__":
    train_sae()
