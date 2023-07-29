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

def combine_npy():
    data_dir="/home/aistudio/work/meta_hpc/all_data/bwalker"
    for p_num in range(4):
        p_data_list=[]
        for npy_num in range(14):
            npy_path=f"{data_dir}/process_{p_num}/{npy_num}.npy"
            npy_data=np.load(npy_path,allow_pickle=True).item()
            # print(p_num,npy_num,[[key,npy_data[key].shape] for key in npy_data.keys()])
            p_data_list.append(npy_data)
        p_all_data={key:np.concatenate([data[key] for data in p_data_list],axis=0) for key in npy_data.keys()}
        print(p_num,npy_num,[[key,p_all_data[key].shape] for key in p_all_data.keys()])
        np.save(f"{data_dir}/process_{p_num}.npy",p_all_data)
def zip_npy():
    data_dir="/home/aistudio/work/meta_hpc/all_data/bwalker"
    z_file_path=f"{data_dir}/bwalker_data.zip"
    z_file=zipfile.ZipFile(z_file_path,'w',zipfile.ZIP_LZMA)
    # z_file.write(f"{data_dir}/bwalker_data")
    for p_num in range(4):
        z_file.write(f"{data_dir}/bwalker_data/process_{p_num}.npy",arcname="process_{p_num}.npy")

def read_npy():
    npy_dir="/home/aistudio/data/data189637"
    npy_data_list=[]
    for i in range(4):
        npy_path=f"{npy_dir}/process_{i}.npy"
        npy_data=np.load(npy_path,allow_pickle=True).item()
        npy_data_list.append(npy_data)
    return npy_data_list
def train():
    from mlutils.ml import moving_avg
    from ec.seq_vae_pred_halfattention import SeqVaePred
    from ec.hyperparam.bwalker_seq_pred_halfattention_hp import svaepred_info
    import paddle
    from mlutils.logger import mllogger

    log=mllogger()

    train_batchsize=128
    train_history_len=128
    load_num=None

    avg_loss=moving_avg(0.999)
    svaepred=SeqVaePred(svaepred_info)
    if load_num is not None:
        svaepred.load_model("all_models/bwalker_svaepred",load_num)
    npy_data_list=read_npy()
    paddle_data_list=[{key:paddle.to_tensor(np.expand_dims(data[key],0),'float32') for key in data.keys()} for data in npy_data_list]

    print(paddle_data_list[0]["state"].shape,paddle_data_list[0]["action"].shape)

    _,data_max_len,_=paddle_data_list[0]["state"].shape
    print(data_max_len)

    for i in range(load_num+1 if load_num is not None else 0,1000000000000):
        t1=time.time()
        rand_data_index=np.random.randint(0,4)
        rand_data=paddle_data_list[rand_data_index]
        batch_state=[]
        batch_reward=[]
        batch_action=[]
        for b in range(train_batchsize):
            rand_slice=np.random.randint(0,data_max_len-train_history_len)
            batch_state.append(rand_data["state"][:,rand_slice:rand_slice+train_history_len])
            batch_reward.append(rand_data["reward"][:,rand_slice:rand_slice+train_history_len])
            batch_action.append(rand_data["action"][:,rand_slice:rand_slice+train_history_len])
        batch_state=paddle.concat(batch_state,axis=0)
        batch_reward=paddle.concat(batch_reward,axis=0)
        batch_action=paddle.concat(batch_action,axis=0)
        t2=time.time()

        loss=svaepred.train(paddle.concat([batch_state,batch_reward],axis=-1),batch_action,None,None)
        t3=time.time()

        log.log_dict({"iter":i,"pred_loss":loss[1],"kl_loss":loss[2]})

        avg_loss.update(np.array(loss))
        if i%100==0:
            print(i,avg_loss,t2-t1,t3-t2)
        if i%10000==0:
            svaepred.save_model("all_models/bwalker_svaepred",i)






if __name__=="__main__":
    train()


