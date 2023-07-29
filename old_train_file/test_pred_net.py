from env.env_pred_goal import FindBallEnv
import numpy as np
import cv2
from pfc.gru_pred import GruPredHid
from mlutils.ml import moving_avg
from matplotlib import pyplot as plt
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
def render_surprise(action_list,state_list,surprise_list,name):
    bg_grid_shape=np.array([11,20])
    block_size=np.array([50,50])
    bg_img=np.zeros([*(bg_grid_shape*block_size),3],np.uint8)
    # print(np.shape(bg_img))
    print(np.shape(surprise_list))

    action_list=np.reshape(action_list,[-1,10])
    state_list=np.reshape(state_list,[-1,2])
    for t in range(len(action_list)):
        color=[50,100,200] if state_list[t][1]==0 else [50,200,50]
        x_min=t*block_size[1]
        x_max=(t+1)*block_size[1]
        action_index=np.argmax(action_list[t])
        y_min=action_index*block_size[0]
        y_max=(action_index+1)*block_size[0]
        bg_img[y_min:y_max,x_min:x_max]=color
      
            # print(str(round(surprise_list[t],3)),((t-1)*block_size[1],bg_grid_shape[0]*block_size[0]),cv2.FONT_HERSHEY_COMPLEX,10,(255,255,255),1)
        cv2.putText(bg_img,str(round(surprise_list[t],3)),(int((t+0.0)*block_size[1]),int((bg_grid_shape[0]-0.75+0.5*(t%2))*block_size[0])),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    cv2.imshow(name,bg_img)
    # cv2.waitKey()
def std_model(a_his,s_his):
    ball_index=None
    check_info=np.zeros([10])
    loss_list=[]
    pred_list=[]
    for i in range(len(a_his)):
        a=a_his[i]
        s=s_his[i]
        check_index=np.where(a==1)[0][0]
        contain_ball=np.where(s==1)[0][0]
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

        print("i=",i,"check",check_index,"Ball",contain_ball,"pred",round(pred,2),"loss",round(loss,2),"mark",check_info)
    return loss_list,pred_list

def test_std_model():

    env=FindBallEnv()
    avg_loss=moving_avg()
    for i in range(1,100000000000):
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

        loss_list,pred_list=std_model(a_history,s_history)
        avg_loss.update(np.mean(loss_list))
        if i%1000==0:
            print(i,avg_loss)


def test():
    import os
    # os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]=""
    env=FindBallEnv()
    prednet=GruPredHid()
    avg_loss=moving_avg(0.99)

    prednet.load_model("all_models/prednet_model",118000)
    
    for i in range(1,100000000000):
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
        
        kl_loss_list,kl_pred_list=prednet.test(np.array([s_history],np.float32),np.array([a_history],np.float32))
        kl_loss_list=np.reshape(kl_loss_list.numpy(),[-1])
        kl_pred_list=np.reshape(kl_pred_list,[-1])

        std_pred_list=std_model(a_history,s_history)
        # print(kl_loss_list)
        # plt.subplot(2,1,1)
        # plt.imshow(np.transpose(np.reshape(a_history,[-1,10]),[1,0]))
        # plt.subplot(2,1,2)
        # plt.plot(np.reshape(s_history,[-1,2])[1:,1])
        # plt.plot(np.log(kl_loss_list))
        # plt.show()
        # print(kl_loss_list.numpy())
        render_surprise(a_history,s_history,kl_loss_list,"loss")
        render_surprise(a_history,s_history,kl_pred_list,"kl_pred")
        render_surprise(a_history,s_history,std_pred_list,"std_pred")
        cv2.waitKey()

if __name__=="__main__":
    test()
    # test_std_model()
