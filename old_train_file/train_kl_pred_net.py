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
def std_model(a_his,s_his):
    ball_index=None
    check_info=np.zeros([10])
    loss_list=[]
    pred_list=[]
    for i in range(len(a_his)):
        a=a_his[i]
        s=s_his[i]
        check_index=np.where(a==1)[0][0]
        contain_ball=float(s)
        "pred"
        if ball_index is None:
            pass
            if check_info[check_index]==1:
                pred=0
            else:
                pred=1/(10-np.sum(check_info))
        else:
            if check_index==ball_index:
                pred=1
            else:
                pred=0
        loss=(contain_ball-pred)**2
        loss_list.append(loss)
        pred_list.append(pred)
        "update"
        if contain_ball:
            ball_index=check_index
        else:
            check_info[check_index]=1
        # print(pred,loss)
        # print("i=",i,"check",check_index,"Ball",contain_ball,"pred",round(pred,2),"loss",round(loss,2),"mark",check_info)
    return loss_list,pred_list
def train_pred():
    env=FindBallEnv()
    prednet=KLPredNet()
    avg_loss=moving_avg(0.99)
    std_avg_loss=moving_avg(0.9999)
    
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
                s_history.append(obs[1][1:])
                # print(obs)
            # print(np.shape(history),np.shape(obs))
            # print(np.argmax(a_history,axis=-1),np.array(s_history)[:,1].astype(np.int))
            # print(np.array(s_history)[:,1].astype(np.int))
            prednet.rpm_collect({
                "action_history":a_history,
                "obs_history":s_history
            })
        # std_loss_list,_=std_model(a_history,s_history)
        # std_avg_loss.update(np.mean(std_loss_list))

        loss_list=prednet.learn()

        if loss_list is None:
            continue
        # loss_list=prednet.train_history(s_history,a_history)
        avg_loss.update(loss_list)


        
        if i%100==0:
            print(i,avg_loss,"std loss",std_avg_loss,prednet.optimizer.get_lr(),prednet.rpm._size)
            # test(env,prednet)
        if i%200==0:
            # print(np.argmax(a_history,axis=-1),np.array(s_history)[:,1].astype(np.int))
            # render_pred(prednet.label,prednet.pred_result)
            pass
        if i%2000==0:
            prednet.save_model("all_models/kl_prednet_model",i)
if __name__=="__main__":
    train_pred()
