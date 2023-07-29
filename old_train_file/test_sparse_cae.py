
import imp
from vae.vae import V_Auto_Encoder
from env.env_war_frog_gqn import multi_circle_env
import numpy as np

import os
import random
import paddle
from paddle import nn
from myf_ML_util import moving_avg
from hpc.sparse_cae import Sparse_cae



def train():
    img_num=128

    env=multi_circle_env()
    model=Sparse_cae()
    model_num=350000
    if model_num==0 or model_num is None:
        pass
    else:
        model.load_model("all_models/scae_model",model_num)


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

            output_dict=model.pred_and_recon([img])
            total_loss=output_dict["total_loss"]
            vae_loss=output_dict["cae_loss"]
            sae_loss=output_dict["sae_loss"]
            vae_recon_img=output_dict["cae_recon_img"]
            test_img=output_dict["test_img"]
            sae_sparse_hid=output_dict["sae_sparse_hid"]
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]=""

            sparse_hid=sae_sparse_hid.numpy()[0]
            print(np.sum(sparse_hid<0.05)/2000,np.sum(sparse_hid>0.95)/2000)


            from matplotlib import pyplot as plt
            plt.hist(sae_sparse_hid.numpy()[0],bins=100)
            plt.show()


            test_img=np.transpose(vae_recon_img.numpy(),[0,2,3,1])[0]
            show_img=np.concatenate([img,test_img],axis=1)
            # print(np.)
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]="/home/kylin/Program/python37_v2/lib/python3.7/site-packages/cv2/qt/plugins"
            import cv2
            cv2.imshow("img",cv2.resize(show_img,None,fx=5,fy=5))
            cv2.waitKey()

        # print()

    # pass

if __name__=="__main__":
    train()
