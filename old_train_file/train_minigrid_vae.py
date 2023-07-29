

import numpy as np
import cv2
import time
from multiprocessing import Process,Queue,Pipe
import threading


def train():
    from mlutils.ml import moving_avg
    from vae.minigrid_vae import MinigridVae
    
    from env.minigrid import MiniGrid_fourroom
    
    net=MinigridVae()
    env=MiniGrid_fourroom()
    avg_loss=moving_avg(0.99)
    # prednet.load_model("all_models/minigrid_model",22000)

    for i in range(1000000000000):
        obs=env.reset()
        done=False
        a_history=[]
        s_history=[]
        render_img_list=[]

        env_time=time.time()
        while not done:
            action=np.clip(np.random.randint(0,4),0,2)
            obs,reward,done=env.step(action)
            net.rpm_collect({"img":np.transpose(obs,[2,0,1])})
            
        loss_list=net.learn()
        if loss_list is None:
            continue
        avg_loss.update(loss_list)

        if i%100==0:
                print(i,avg_loss,net.optimizer.get_lr())
        if i%200==0:
            # print(np.argmax(a_history,axis=-1),np.array(s_history)[:,1].astype(np.int))
            # render_pred(prednet.label,prednet.pred_result)
            pass
        if i%2000==0:
            net.save_model("all_models/minigrid_vae_model",i)




if __name__=="__main__":
    # train_pred()
    train()
