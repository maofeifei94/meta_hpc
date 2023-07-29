from vae.vae import V_Auto_Encoder
from env.env_war_frog_gqn import multi_circle_env
import numpy as np
import cv2
import random
import paddle
from myf_ML_util import moving_avg

def train():
    img_num=128

    env=multi_circle_env()
    model=V_Auto_Encoder()
    model_num=0
    if model_num==0 or model_num is None:
        pass
    else:
        model.load_model("all_models/vae_model",model_num)


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


        img,recon_img=model.learn()

        if iter%100==0:
            print(iter,[model.avg_loss_recon,model.avg_loss_kl])
        if iter%10000==0:
            model.save_model("all_models/vae_model",iter)
        # print()

    # pass

if __name__=="__main__":
    train()