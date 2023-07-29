from env.env_pred_goal import FindBallEnv
import numpy as np
import cv2
from pfc.gru_pred import *
from mlutils.ml import moving_avg

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
    env=FindBallEnv()
    prednet=GruPredHid()
    avg_loss=moving_avg(0.99)
    
    for i in range(1,100000000000):
        for j in range(100):
            obs=env.reset()
            done=False
            a_history=[]
            s_history=[]
            while not done:
                action=np.random.randint(0,env.box_num)
                obs,reward,done=env.step(action)
                # print(np.argwhere(obs==1))
                # history.append(np.concatenate(obs))
                a_history.append(obs[0])
                s_history.append(obs[1])
                # print(obs)
            # print(np.shape(history),np.shape(obs))
            # print(np.argmax(a_history,axis=-1),np.array(s_history)[:,1].astype(np.int))
            # print(np.array(s_history)[:,1].astype(np.int))
            prednet.rpm_collect({
                "action_history":a_history,
                "obs_history":s_history
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
