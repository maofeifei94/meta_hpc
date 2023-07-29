
from cv2 import waitKey
from vae.vae import V_Auto_Encoder
from env.env_war_frog_gqn import multi_circle_env
import numpy as np
import cv2
import random
import paddle
from myf_ML_util import moving_avg

# import os
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]=""

from matplotlib import pyplot as plt

def test():
    img_num=128

    env=multi_circle_env()
    model=V_Auto_Encoder()


    model.load_model("all_models/vae_model",240000)


    for iter in range(0,999999999):
        "get data"
        env.reset()
        full_img=env.render_full_img_cv()

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

            # def plot_d_L2():
            #     pa_img=paddle.to_tensor([img])
            #     pa_img=paddle.transpose(pa_img,perm=[0,3,1,2])
            #     mean,log_var=model.encoder(pa_img)

            #     d_z_list=[]
            #     d_img_list=[]

            #     for k in range(1000):
            #         z=model.reparameterize(mean,log_var)
            #         recon_img=model.decoder(z)

            #         d_z=paddle.mean((z-mean)**2)**0.5
            #         d_img=paddle.mean((recon_img-pa_img)**2)**0.5
            #         d_z_list.append(d_z.numpy()[0])
            #         d_img_list.append(d_img.numpy()[0])
            #     print(d_z_list,d_img_list)
            #     plt.scatter(d_z_list,d_img_list)
            #     # plt.hist(d_z_list)
            #     # plt.hist(d_img_list)
            #     plt.show()
            # plot_d_L2()

            for j in range(5):
                recon_img=model.ae([img])[0].numpy()
                recon_img=np.transpose(recon_img,[1,2,0])
                cv2.imshow("img",cv2.resize(np.concatenate([img,recon_img],axis=1),None,fx=4,fy=4,interpolation=cv2.INTER_AREA))
                cv2.waitKey()

            


        

        

    # pass

if __name__=="__main__":
    test()