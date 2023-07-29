
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
from hpc.sparse_vae import Sparse_VAE
from matplotlib import pyplot as plt

def train():
    img_num=128

    env=multi_circle_env()
    model=Sparse_VAE()
    model_num=0
    if model_num==0 or model_num is None:
        pass
    else:
        model.load_model("all_models/svae_model",model_num)


    for iter in range(model_num,999999999):
        "get data"
        env.reset()
        img_list=[]
        view_list=[]
        for i in range(img_num):
            rand_site=np.random.uniform(0,1,[2])
            env_info,reward,done=env.step(None,rand_site,True)
            # print(env_info.keys())
            img=env_info['ask_ball_obs']
            view=env_info['ask_site_norm']
            # print(view,np.max(img),np.shape(img))
            # cv2.imshow("img",img)
            # cv2.waitKey(10)
            model.rpm_collect(img)


        output_dict=model.learn(1000)
        total_loss=output_dict["total_loss"]
        vae_loss=output_dict["vae_loss"]
        sae_loss=output_dict["sae_loss"]
        vae_recon_img=output_dict["vae_recon_img"]
        train_img=output_dict["train_img"]

        if iter%100==0:
            print(iter,total_loss,vae_loss,sae_loss)
        if iter%10000==0:
            model.save_model("all_models/svae_model",iter)
        # print()

    # pass

if __name__=="__main__":
    train()
