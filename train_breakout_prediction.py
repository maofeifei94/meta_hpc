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
    from ec.gru_prediction import GruPred
    from ec.hyperparam.gru_prediction_hp import svaepred_info
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


    # train_batchsize=128
    # train_history_len=128
    load_num=None
    process_num=1
    npy_dir="all_data/breakout"
    update_h_history_len=16384
    update_h_interval=40000
    queue_max_size=10
    save_interval=10000

    "model"
    avg_loss=moving_avg(0.999)
    svaepred=GruPred(svaepred_info)
    
    # print("mean before load=",svaepred.get_sd_mean())
    if load_num is not None:
        svaepred.load_model("all_models/breakout_svaepred",load_num)
    # print("mean after load=",svaepred.get_sd_mean())




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
            data_dict[key]=data_dict[key]
        data_list.append(data_dict)
    # data_list=[{key:np.expand_dims(data[key],axis=0) for key in data.keys} for data in npy_data_list]
    print([[key,data_list[0][key].shape] for key in data_list[0].keys()])
    # input()
    # data_max_len=npy_data_list
    _,data_max_len,_=data_list[0]['state'].shape



    

    
            
            


    def train_thread(queue_dict):
        queue_data=queue_dict['data']
        queue_signal=queue_dict['signal']


        # update_gru_h(data_list,svaepred)
        print("finish init gru h")
        # print(data_list[0]["global_h"].shape)
        queue_signal.put("start")

        for i in range(load_num+1 if load_num is not None else 1,1000000000000):
            # if i%update_h_interval==0:
            #     queue_signal.put("fresh")
            #     update_gru_h(data_list,svaepred)
            #     print(i,"update gru h finish")
                
                


            t1=time.time()

            batch_data_dict=queue_data.get()
            batch_state=batch_data_dict['state']
            batch_reward=batch_data_dict['reward']
            batch_action=batch_data_dict['action']
            batch_global_h=batch_data_dict['global_h']
            batch_local_h=batch_data_dict['local_h']


            t2=time.time()

            # loss=svaepred.train_with_warmup(paddle.concat([batch_state,batch_reward],axis=-1),batch_action,batch_global_h)
            # t3=time.time()
            loss=svaepred.train_with_warmup(batch_state,batch_action,batch_global_h)
            t3=time.time()

            avg_loss.update(np.array(loss))
            if i%100==0:
                train_log.log_dict({"iter":i,"pred_loss":loss[0]})
                print(i,avg_loss,t2-t1,t3-t2)
            if i%save_interval==0:
                svaepred.save_model("all_models/breakout_svaepred",i)
                # print("mean after save=",svaepred.get_sd_mean())
    def data_thread(queue_dict):
        queue_data=queue_dict['data']
        queue_signal=queue_dict['signal']
        queue_signal.get()
        print("start data")
        while 1:
            if not queue_signal.empty():
                queue_signal.get()
                for _ in range(queue_data.qsize()):
                    queue_data.get()

            rand_data_index=np.random.randint(0,process_num)
            rand_data=data_list[rand_data_index]
            batch_state=[]
            batch_reward=[]
            batch_action=[]
            batch_local_h=[]
            batch_global_h=[]
            for b in range(svaepred_info.train_batch_size):
                rand_slice=np.random.randint(1,data_max_len-svaepred_info.len_max_history)
                batch_state.append(rand_data["state"][:,rand_slice:rand_slice+svaepred_info.len_max_history])
                batch_reward.append(rand_data["reward"][:,rand_slice-1:rand_slice-1+svaepred_info.len_max_history])
                batch_action.append(rand_data["action"][:,rand_slice:rand_slice+svaepred_info.len_max_history])
                batch_local_h.append(np.zeros([1,1,svaepred_info.gru_dim]))
                batch_global_h.append(np.zeros([1,1,svaepred_info.gru_dim]))

            batch_state=paddle.to_tensor(np.concatenate(batch_state,axis=0),dtype='float32')
            batch_reward=paddle.to_tensor(np.concatenate(batch_reward,axis=0),dtype='float32')
            batch_action=paddle.to_tensor(np.concatenate(batch_action,axis=0),dtype='float32')
            batch_local_h=paddle.to_tensor(np.concatenate(batch_local_h,axis=1),dtype='float32')
            batch_global_h=paddle.to_tensor(np.concatenate(batch_global_h,axis=1),dtype='float32')

            queue_data.put({
                "state":batch_state,
                "reward":batch_reward,
                "action":batch_action,
                "local_h":batch_local_h,
                "global_h":batch_global_h
            })
            # print(queue_data.qsize())

    queue_dict={
        'data':Queue(maxsize=queue_max_size),
        'signal':Queue(maxsize=queue_max_size),
    }
    p_list=[]

    p_list.append(Thread(target=train_thread,args=(queue_dict,)))
    p_list.append(Thread(target=data_thread,args=(queue_dict,)))
    [p.start() for p in p_list]
    [p.join() for p in p_list]
    pass






if __name__=="__main__":
    train()


