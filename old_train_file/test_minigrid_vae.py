

import numpy as np
import cv2
import time
from multiprocessing import Process,Queue,Pipe
import threading
import os

def test():
    from mlutils.ml import moving_avg
    from vae.minigrid_vae import MinigridVae
    
    from env.minigrid import MiniGrid_fourroom
    
    net=MinigridVae()
    env=MiniGrid_fourroom()
    avg_loss=moving_avg(0.99)
    # print(os.path.exists("all_models/minigrid_vae_model/110000_model.pdparams"))
    net.load_model("all_models/minigrid_vae_model",36000)
    

    for i in range(1000000000000):
        obs=env.reset()
        done=False
        a_history=[]
        s_history=[]
        render_img_list=[]

        env_time=time.time()
        img_list=[]
        while not done:
            action=np.clip(np.random.randint(0,4),0,2)
            obs,reward,done=env.step(action)
            img_list.append(np.transpose(obs,[2,0,1]))
            
        recon_img_list=net.test(np.array(img_list))

        for _j in range(len(recon_img_list)):
            ori_img=np.transpose(img_list[_j],[1,2,0])
            recon_img=(np.transpose(recon_img_list[_j],[1,2,0])*255).astype(np.uint8)

            cv2.imshow("ori_img",cv2.resize(ori_img,None,fx=5,fy=5,interpolation=cv2.INTER_AREA))
            cv2.imshow("recon_img",cv2.resize(recon_img,None,fx=5,fy=5,interpolation=cv2.INTER_AREA))
            cv2.waitKey()










if __name__=="__main__":
    # train_pred()
    test()
