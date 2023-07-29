from env.env_pred_kl import multi_circle_env
import numpy as np
import cv2
from pfc.gru_pred_conv import GruPredHid 
from mlutils.ml import moving_avg
import time
from gym_minigrid.wrappers import *
class render():
    def __init__(self) -> None:
        pass
    def reset(self):
        pass
    def step(self,ball_index,action,pred_prob):
        pass
def render_pred(label,pred):
    ratio=50
    cv2.imshow("label",cv2.resize(label,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA))
    cv2.imshow("pred",cv2.resize(pred,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA))
    cv2.waitKey(100)

def test(env,prednet):
    prednet.reset_pred()
    obs=env.reset()
    done=False

    while not done:
        action=np.random.randint(0,env.box_num)
        obs,reward,done=env.step(action)
        # print(np.argwhere(obs==1))
        prednet.pred(np.concatenate(obs))
def train_pred():
    env = gym.make('MiniGrid-FourRooms-v0')
    prednet=GruPredHid()
    avg_loss=moving_avg(0.99)

    import os
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=''
    from matplotlib import pyplot as plt
    
    for i in range(1,100000000000):
        for j in range(100):
            obs=env.reset()
            done=False
            a_history=[]
            s_history=[]
            s_last=None

            t1=time.time()
            while not done:
                # action=np.random.uniform(-1,1,[2])
                action=min(np.random.randint(0,4),2)
                action_vec=np.zeros([3])
                action_vec[action]=1

                agent_obs=np.zeros([7,7,3],np.uint8)
                agent_obs[:7,:7]=obs['image']
                # print(np.max(obs["image"]),obs['image'].dtype)
                s_history.append(np.transpose(agent_obs,[2,0,1]))
                a_history.append(action_vec)

                obs,reward,done,info=env.step(action)
                
                # print(np.argwhere(obs==1))
                # history.append(np.concatenate(obs))
                env.render()

                # plt.imshow(obs['image'])
                # plt.show()
                
                print(np.shape(obs['image']))
                print(obs['image'][:,:,0])
                print(obs['image'][:,:,1])
                print(obs['image'][:,:,2],)
                # input()
                # env.render()
                # cv2.imshow("out_ball_obs",cv2.resize(agent_obs,(128,128),interpolation=cv2.INTER_AREA)*40)
                # cv2.waitKey()
                # print(obs)
            print(np.shape(a_history),np.shape(s_history))
            print("cost",time.time()-t1)
            prednet.rpm_collect({
                "action_history":a_history,
                "obs_history":s_history,
                # "obs_last":obs["ball_obs"]
            })

        loss_list=prednet.learn()
        # loss_list=prednet.train_history(s_history,a_history)
        avg_loss.update(loss_list)
        
        if i%100==0:
            print(i,avg_loss,prednet.optimizer.get_lr())
            # test(env,prednet)
        if i%200==0:
            # print(np.argmax(a_history,axis=-1),np.array(s_history)[:,1].astype(np.int))
            # render_pred(prednet.label,prednet.pred_result)
            pass
        if i%2000==0:
            prednet.save_model("all_models/prednet_model",i)
if __name__=="__main__":
    train_pred()
