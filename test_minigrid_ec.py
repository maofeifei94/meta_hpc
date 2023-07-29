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
    def __init__(self,data_dir="all_data/minigrid_img8x8_history") -> None:
        self.data_dir=data_dir
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
    from ec.seq_vae_pred_minigrid import SeqVaePred
    from ec.hyperparam.minigrid_seq_pred_hp import svaepred_info
    import paddle

    avg_loss=moving_avg(0.999)
    ndr=npy_data_reader("C:/Users/admin/Desktop/code/minigrid_data/minigrid_img8x8_history")
    svaepred=SeqVaePred(svaepred_info)
    svaepred.load_model("all_models/minigrid_svaepred",520000)
    # paddle_sa_list=[]
    # max_size=400
    while 1:
        data=ndr.get_data()
        s_history=np.reshape(data["s_history"],[*np.shape(data["s_history"])[:2],-1])
        action_history=data["action_history"]
        test_s=paddle.to_tensor(s_history,'float32')/255
        test_a=paddle.to_tensor(action_history,'float32')
        pred,pred_target=svaepred.test(test_s,test_a)
        pred=np.reshape(pred,[1000,99,3,8,8])
        pred_target=np.reshape(pred_target,[1000,99,3,8,8])

        for i in range(5):
            for j in range(20):
                print(i,j)
                img_pred=np.transpose(pred[i,j],[1,2,0])
                img_target=np.transpose(pred_target[i,j],[1,2,0])
                img_show=np.concatenate([img_pred,img_target],axis=1)
                cv2.imshow("img_show",cv2.resize(img_show,None,fx=20,fy=20,interpolation=cv2.INTER_AREA))
                cv2.waitKey()

if __name__=="__main__":
    train()


