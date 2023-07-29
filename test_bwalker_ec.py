from mlutils.ml import one_hot
import numpy as np
import cv2
import time
from multiprocessing import Process,Queue,Pipe
import threading
import random
import os
import zipfile
from mlutils.data import npy_compress


def train():
    from mlutils.ml import moving_avg
    from ec.seq_vae_pred import SeqVaePred
    from ec.hyperparam.bwalker_seq_pred_hp import svaepred_info
    import paddle

    train_batchsize=128
    train_history_len=128
    load_num=130000

    avg_loss=moving_avg(0.999)
    svaepred=SeqVaePred(svaepred_info)

    if load_num is not None:
        svaepred.load_model("all_models/bwalker_svaepred",load_num)

    npy_data=np.load("/home/kylin/下载/process_0.npy",allow_pickle=True).item()
    paddle_data={key:paddle.to_tensor(np.expand_dims(npy_data[key],0),'float32') for key in npy_data.keys()}


    _,data_max_len,_=paddle_data["state"].shape


    for i in range(load_num+1 if load_num is not None else 0,1000000000000):
        t1=time.time()
        rand_data=paddle_data
        batch_state=[]
        batch_reward=[]
        batch_action=[]
        for b in range(1):
            rand_slice=np.random.randint(0,data_max_len-train_history_len)
            batch_state.append(rand_data["state"][:,rand_slice:rand_slice+train_history_len])
            batch_reward.append(rand_data["reward"][:,rand_slice:rand_slice+train_history_len])
            batch_action.append(rand_data["action"][:,rand_slice:rand_slice+train_history_len])
        batch_state=paddle.concat(batch_state,axis=0)
        batch_reward=paddle.concat(batch_reward,axis=0)
        batch_action=paddle.concat(batch_action,axis=0)
        t2=time.time()

        pred,pred_target=svaepred.test(paddle.concat([batch_state,batch_reward],axis=-1),batch_action,None,None)
        t3=time.time()

        for j in range(pred.shape[1]):
            print("pred",{_k:_num for _k,_num in enumerate(np.round(pred[0,j],3))})
            print("pred_target",{_k:_num for _k,_num in enumerate(np.round(pred_target[0,j],3))})
            loss=np.abs(np.round(pred[0,j],3)-np.round(pred_target[0,j],3))
            print(np.mean(loss**2))
            print("loss",{_k:_num for _k,_num in enumerate(np.round(loss,3))})
        input()

if __name__=="__main__":
    train()


