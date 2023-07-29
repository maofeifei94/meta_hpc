
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
from hpc.sparse_cae_mnist import Sparse_cae
from matplotlib import pyplot as plt
from mlutils.data import mnist_data

def train():
    img_num=128

    env=mnist_data()

    model=Sparse_cae()
    
    model_num=0
    if model_num==0 or model_num is None:
        pass
    else:
        model.load_model("all_models/scae_model",model_num)


    for iter in range(model_num+1,999999999):
        img_batch=env.get_rand_batch(img_num)
        # print(np.shape(img_batch))
        for i in range(img_num):
            model.rpm_collect(img_batch[i])


        output_dict=model.learn(1000)
        total_loss=output_dict["total_loss"]
        cae_loss=output_dict["cae_loss"]
        sae_loss=output_dict["sae_loss"]
        cae_recon_img=output_dict["cae_recon_img"]
        train_img=output_dict["train_img"]

        if iter%100==0:
            print(iter,total_loss,cae_loss,sae_loss)
        if iter%10000==0:
            model.save_model("all_models/scae_model",iter)
        # print()

    # pass

if __name__=="__main__":
    train()
