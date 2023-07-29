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


def mp_npy_compress():
    num=4
    def npy_process(index,pnum):
        npy_compress(index,pnum).check_npy_and_compress()
    
    p_list=[]
    for i in range(num):
        p_list.append(Process(target=npy_process,args=(i,num)))

    [p.start() for p in p_list]
    [p.join() for p in p_list]



class npy_data_reader():
    def __init__(self) -> None:
        self.data_dir="all_data/minigrid_history"
        self.nc=npy_compress()
        self.reset()
    def reset(self):
        self.current_index=0
        self.npy_num_list=self.get_all_npy_zip_num(self.data_dir)
        self.npy_file_num=len(self.npy_num_list)
        
        random.shuffle(self.npy_num_list)
        print(f"start new npy loop with file name {self.npy_file_num}")
        # print(self.npy_num_list)
        # print("npy start new loop with data num ",self.npy_num)
        # print(os.listdir(self.data_dir))
    def get_all_npy_zip_num(self,data_dir):
        num_list=[]
        for f in os.listdir(data_dir):
            if f[-8:]==".npy.zip":
                num=int(f[:-8])
                num_list.append(num)
        return num_list

    def get_data(self):
        current_npy_name=f"{self.npy_num_list[self.current_index]}.npy"

        current_npy_path=f"{self.data_dir}/{current_npy_name}"

        current_zip_name=f"{current_npy_name}.zip"

        current_zip_path=f"{self.data_dir}/{current_zip_name}"

        # print("decompress")
        self.nc.decompress(current_zip_path,self.data_dir,current_npy_name)
        current_data=np.load(current_npy_path,allow_pickle=True).item()
        self.nc.delete_f(current_npy_path)
        self.current_index+=1
        if self.current_index>=self.npy_file_num:
            self.reset()
        return current_data
def train():
    from mlutils.ml import moving_avg
    from pfc.gru_pred_conv import GruPredHid 

    prednet=GruPredHid()
    avg_loss=moving_avg(0.99)
    ndr=npy_data_reader()

    for i in range(1000000000000):
        if i%50==0:
            data=ndr.get_data()

            obs_feature_history=data["obs_feature_history"]
            action_history=data["action_history"]
            if np.shape(action_history)[-1]==1:
                action_history=np.squeeze(one_hot(action_history,3))
            # print("obs shape",np.shape(obs_feature_history))
            # print("act shape",np.shape(action_history))
            for _d in range(len(obs_feature_history)):
                # print("collect")
                prednet.rpm_collect({
                    "obs_feature_history":obs_feature_history[_d],
                    "action_history":action_history[_d]
                })

        # print("learn")
        loss_list=prednet.learn()
        if loss_list is None:
            continue
        avg_loss.update(loss_list)
        # print("learn cost",time.time()-t_learn)
    
        if i%100==0:
            print(i,avg_loss,prednet.optimizer.get_lr())
        if i%200==0:
            # print(np.argmax(a_history,axis=-1),np.array(s_history)[:,1].astype(np.int))
            # render_pred(prednet.label,prednet.pred_result)
            pass
        if i%2000==0:
            prednet.save_model("all_models/minigrid_model",i)





if __name__=="__main__":
    train()


