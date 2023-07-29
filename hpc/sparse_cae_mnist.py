from turtle import forward
import os
import numpy as np
import paddle.nn as nn
import paddle
import paddle.optimizer as optim
import paddle.nn.functional as F
from myf_ML_util import dense_block,Reslayer
from myf_ML_util import Basic_Module, ReplayMemory,dense_block,DataFormat
from hpc.sparse_ae_mnist import Sparse_Autoencoder
from cae.cae_mnist import Conv_Auto_Encoder
class Sparse_cae(nn.Layer):
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)

        self.sparse_ae=Sparse_Autoencoder(49*2)
        self.cae=Conv_Auto_Encoder()
        self.rpm=ReplayMemory([DataFormat("img",[28,28,1],np.float32)],100000)
        self.optimizer=optim.RMSProp(0.0003,parameters=[*self.sparse_ae.parameters(),*self.cae.parameters()])
    def rpm_collect(self,img):
        "收集训练数据"
        self.rpm.collect({'img':img})
        pass
    def rpm_clear(self):
        self.rpm.clear()
        pass
    def pred_and_recon(self,img):
        test_img=paddle.to_tensor(img)
        test_img=paddle.transpose(test_img,perm=[0,3,1,2])

        cae_z=self.cae.encoder(test_img)

        "sparse_ae process"
        sparse_ae_loss_pred=self.sparse_ae.pred_with_loss(paddle.flatten(cae_z,1,3))
        sae_loss_recon=sparse_ae_loss_pred["loss_recon"]
        sae_loss_data_fire=sparse_ae_loss_pred["loss_data_fire"]
        sae_loss_cell_fire=sparse_ae_loss_pred["loss_cell_fire"]
        sae_loss_cell_fire_fast=sparse_ae_loss_pred["loss_cell_fire_fast"]
        sae_loss_cell_fire_slow=sparse_ae_loss_pred["loss_cell_fire_slow"]
        sae_recon_hid_code=sparse_ae_loss_pred["recon_hid_code"]
        sae_sparse_hid=sparse_ae_loss_pred["sparse_hid"]
        sae_loss=sae_loss_recon+(sae_loss_data_fire+sae_loss_cell_fire)*1.0

        "recon_z"
        rand_ratio=1.0
        recon_z=paddle.reshape(sae_recon_hid_code,cae_z.shape)*rand_ratio+cae_z*(1-rand_ratio)

        "recon_img"
        cae_recon_img=self.cae.decoder(paddle.reshape(recon_z,cae_z.shape))
        cae_loss_recon=0.5*paddle.mean((test_img-cae_recon_img)**2)
        # loss_recon=paddle.mean(paddle.abs(img-recon_img))
        cae_loss=cae_loss_recon


        total_loss=sae_loss+cae_loss


        output_dict={
            "total_loss":total_loss.numpy()[0],
            "cae_loss":[cae_loss.numpy()[0],cae_loss_recon.numpy()[0]],
            "sae_loss":[sae_loss.numpy()[0],sae_loss_recon.numpy()[0],sae_loss_data_fire.numpy()[0],sae_loss_cell_fire_fast.numpy()[0],sae_loss_cell_fire_slow.numpy()[0]],
            "cae_recon_img":cae_recon_img,
            "test_img":test_img,
            "sae_sparse_hid":sae_sparse_hid,

        }
        return output_dict


    def learn(self,train_batch_size):
        train_data=self.rpm.sample_batch(train_batch_size)
        train_img=paddle.to_tensor(train_data['img'])
        train_img=paddle.transpose(train_img,perm=[0,3,1,2])

        cae_z=self.cae.encode(train_img)

        "sparse_ae process"
        sparse_ae_loss_pred=self.sparse_ae.pred_with_loss(paddle.flatten(cae_z,1,3))
        sae_loss_recon=sparse_ae_loss_pred["loss_recon"]
        sae_loss_data_fire=sparse_ae_loss_pred["loss_data_fire"]
        sae_loss_cell_fire=sparse_ae_loss_pred["loss_cell_fire"]
        sae_loss_cell_fire_fast=sparse_ae_loss_pred["loss_cell_fire_fast"]
        sae_loss_cell_fire_slow=sparse_ae_loss_pred["loss_cell_fire_slow"]
        sae_loss_cell_fire_current=sparse_ae_loss_pred["loss_cell_fire_current"]
        sae_recon_hid_code=sparse_ae_loss_pred["recon_hid_code"]
        sae_loss=sae_loss_recon*100.0+(sae_loss_data_fire*100+sae_loss_cell_fire*100)

        "recon_z"
        rand_ratio=0.5
        recon_z=paddle.reshape(sae_recon_hid_code,cae_z.shape)*rand_ratio+cae_z*(1-rand_ratio)

        "recon_img"
        cae_recon_img=self.cae.decode(recon_z)
        cae_loss_recon=0.5*paddle.mean((train_img-cae_recon_img)**2)
        # loss_recon=paddle.mean(paddle.abs(img-recon_img))
        cae_loss=cae_loss_recon*1000


        total_loss=sae_loss+cae_loss

        self.optimizer.clear_grad()
        total_loss.backward()
        self.optimizer.step()

        output_dict={
            "total_loss":total_loss.numpy()[0],
            "cae_loss":[cae_loss.numpy()[0],cae_loss_recon.numpy()[0]],
            "sae_loss":[sae_loss.numpy()[0],sae_loss_recon.numpy()[0],sae_loss_data_fire.numpy()[0],sae_loss_cell_fire_current.numpy()[0],sae_loss_cell_fire_fast.numpy()[0],sae_loss_cell_fire_slow.numpy()[0]],
            "cae_recon_img":cae_recon_img,
            "train_img":train_img

        }
        return output_dict


    def save_model(self,save_dir,iter_num):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"start_save_model {save_dir}/{iter_num}_model.pdparams")
        paddle.save(self.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")
        print(f"save model {save_dir}/{iter_num}_model.pdparams success")
        paddle.save(self.state_dict(),f"{save_dir}/newest_model.pdparams")

    def load_model(self,save_dir,iter_num):
        print(f"start load_model {save_dir}/{iter_num}_model.pdparams")
        self.set_state_dict(paddle.load(path=f"{save_dir}/{iter_num}_model.pdparams"))
        print(f"load model {save_dir}/{iter_num}_model.pdparams success")





        







        






