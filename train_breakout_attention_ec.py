from mlutils.logger import mllogger
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


def read_npy(npy_dir,process_num):
    npy_data_list=[]
    for i in range(process_num):
        npy_path=f"{npy_dir}/process_{i}.npy"
        npy_data=np.load(npy_path,allow_pickle=True).item()
        npy_data_list.append(npy_data)
    return npy_data_list
def train():
    from mlutils.ml import moving_avg,one_hot
    from ec.seq_vae_pred_attention import SeqVaePred
    from ec.hyperparam.breakout_seq_pred_attention_hp import svaepred_info
    import paddle
    from mlutils.logger import mllogger
    from queue import Queue
    from threading import Thread


    train_log=mllogger("log")
    for attr in dir(svaepred_info):
        if attr.startswith("__") and attr.endswith("__"):
            pass
        else:
            print(f"{attr}={getattr(svaepred_info,attr)}")
            train_log.log_str(f"{attr}={getattr(svaepred_info,attr)}")


    train_batchsize=128
    train_history_len=128
    load_num=None
    process_num=4
    npy_dir="all_data/breakout"

    avg_loss=moving_avg(0.999)
    svaepred=SeqVaePred(svaepred_info)
    if load_num is not None:
        svaepred.load_model("all_models/breakout_svaepred",load_num)
    npy_data_list=read_npy(npy_dir,process_num)

    data_list=[]

    "数据处理"
    for data in npy_data_list:
        data_dict={}
        for key in ["state","reward","action"]:
            if key=="action":
                print(int(svaepred_info.action_vec_dim),np.expand_dims(data[key][:,0],axis=0).shape)
                data_dict[key]=np.expand_dims(one_hot(data[key][:,0].astype(np.int32),int(svaepred_info.action_vec_dim)),axis=0)
            else:
                data_dict[key]=np.expand_dims(data[key],axis=0)
        data_list.append(data_dict)
    # data_list=[{key:np.expand_dims(data[key],axis=0) for key in data.keys} for data in npy_data_list]
    print([[key,data_list[0][key].shape] for key in data_list[0].keys()])
    # input()
    # data_max_len=npy_data_list
    _,data_max_len,_=data_list[0]['state'].shape

    def train_thread(queue_dict):
        queue_data=queue_dict['data']

        for i in range(load_num+1 if load_num is not None else 0,1000000000000):
            t1=time.time()
            # rand_data_index=np.random.randint(0,process_num)
            # rand_data=data_list[rand_data_index]
            # batch_state=[]
            # batch_reward=[]
            # batch_action=[]
            # for b in range(train_batchsize):
            #     rand_slice=np.random.randint(1,data_max_len-train_history_len)
            #     batch_state.append(rand_data["state"][:,rand_slice:rand_slice+train_history_len])
            #     batch_reward.append(rand_data["reward"][:,rand_slice-1:rand_slice-1+train_history_len])
            #     batch_action.append(rand_data["action"][:,rand_slice:rand_slice+train_history_len])
            batch_data_dict=queue_data.get()
            batch_state=batch_data_dict['state']
            batch_reward=batch_data_dict['reward']
            batch_action=batch_data_dict['action']


            t2=time.time()

            loss=svaepred.train(paddle.concat([batch_state,batch_reward],axis=-1),batch_action,None,None)
            t3=time.time()

            avg_loss.update(np.array(loss))
            if i%100==0:
                train_log.log_dict({"iter":i,"pred_loss":loss[1],"kl_loss":loss[2]})
                print(i,avg_loss,t2-t1,t3-t2)
            if i%10000==0:
                svaepred.save_model("all_models/breakout_svaepred",i)
    def data_thread(queue_dict):
        queue_data=queue_dict['data']
        while 1:
            rand_data_index=np.random.randint(0,process_num)
            rand_data=data_list[rand_data_index]
            batch_state=[]
            batch_reward=[]
            batch_action=[]
            for b in range(train_batchsize):
                rand_slice=np.random.randint(1,data_max_len-train_history_len)
                batch_state.append(rand_data["state"][:,rand_slice:rand_slice+train_history_len])
                batch_reward.append(rand_data["reward"][:,rand_slice-1:rand_slice-1+train_history_len])
                batch_action.append(rand_data["action"][:,rand_slice:rand_slice+train_history_len])

            batch_state=paddle.to_tensor(np.concatenate(batch_state,axis=0),dtype='float32')
            batch_reward=paddle.to_tensor(np.concatenate(batch_reward,axis=0),dtype='float32')
            batch_action=paddle.to_tensor(np.concatenate(batch_action,axis=0),dtype='float32')
            queue_data.put({
                "state":batch_state,
                "reward":batch_reward,
                "action":batch_action
            })
            # print(queue_data.qsize())

    queue_dict={
        'data':Queue(maxsize=20),
    }
    p_list=[]

    p_list.append(Thread(target=train_thread,args=(queue_dict,)))
    p_list.append(Thread(target=data_thread,args=(queue_dict,)))
    [p.start() for p in p_list]
    [p.join() for p in p_list]
    pass






if __name__=="__main__":
    train()


