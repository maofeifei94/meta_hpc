
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
from matplotlib import pyplot as plt
from mlutils.data import mnist_data

from paddle_mnist_fc_sae.mnist_fc_sae import fc_sae

def train():
    batch_size=128

    env=mnist_data()

    model=fc_sae()
    
    model_num=0
    if model_num==0 or model_num is None:
        pass
    else:
        model.load_model("all_models/scae_model",model_num)


    for iter in range(model_num+1,999999999):
        img_batch=env.get_rand_batch(batch_size)
        img_batch=paddle.flatten(paddle.to_tensor(img_batch,dtype='float32'),1,3)
        loss_list,std_loss_list=model.learn(img_batch)

        if iter%100==0:
            print(iter,loss_list,std_loss_list)
        # print()

    # pass

if __name__=="__main__":
    train()
