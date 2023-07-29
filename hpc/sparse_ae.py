from turtle import forward
import os
import numpy as np
import paddle.nn as nn
import paddle
import paddle.optimizer as optim
import paddle.nn.functional as F
from myf_ML_util import dense_block,Reslayer

#经测试，0.99，0.999为最好的组合。Resnet下激活函数Tanh为最好，非Resnet也是tanh最好，但远远不及Resnet
class Sparse_Autoencoder(nn.Layer):
    def __init__(self,hid_vec_dim,nl_func=nn.Tanh(),gamma_fast=0.99,gamma_slow=0.999,) -> None:
        super().__init__()
        # nl_func=nn.LeakyReLU(negative_slope=0.1)
        # nl_func=nl_func
        # hid_vec_dim=512
        encoder_hid_dim=512
        self.encoder_sparse_dim=2000
        self.sparse_ratio=0.05

        
        encoder=nn.Sequential(
                    nn.Sequential(nn.Linear(hid_vec_dim,encoder_hid_dim),nl_func),
                    Reslayer(nn.Sequential(nn.Linear(encoder_hid_dim,encoder_hid_dim),nl_func)),
                    Reslayer(nn.Sequential(nn.Linear(encoder_hid_dim,encoder_hid_dim),nl_func)),
                    Reslayer(nn.Sequential(nn.Linear(encoder_hid_dim,encoder_hid_dim),nl_func)),
                    Reslayer(nn.Sequential(nn.Linear(encoder_hid_dim,encoder_hid_dim),nl_func)),
                    nn.Sequential(nn.Linear(encoder_hid_dim,self.encoder_sparse_dim),nn.Sigmoid())
                )
        decoder=nn.Linear(self.encoder_sparse_dim,hid_vec_dim,bias_attr=False)

        self.encoder=encoder
        self.decoder=decoder

        self.optimizer=optim.Adam(0.0003,parameters=[*self.encoder.parameters(),*self.decoder.parameters()])

        self.avg_fire_gamma_slow=gamma_slow
        self.avg_fire_rate_slow=paddle.ones([self.encoder_sparse_dim])*self.sparse_ratio
        self.avg_fire_gamma_fast=gamma_fast
        self.avg_fire_rate_fast=paddle.ones([self.encoder_sparse_dim])*self.sparse_ratio

        self.avg_fire_gamma_9999=0.9999
        self.avg_fire_rate_9999=paddle.ones([self.encoder_sparse_dim])*self.sparse_ratio
        
    def norm_hid(self,hid_code):
        # return self.k*hid_code/torch.sum(hid_code,dim=-1,keepdim=True)
        hid2=hid_code**2
        return self.encoder_sparse_dim*self.sparse_ratio*hid2/paddle.sum(hid2,axis=-1,keepdim=True)
    def kl_loss(self,x,ratio):
        return paddle.mean(-(ratio*paddle.log(x)+(1-ratio)*paddle.log(1-x)))
    def pred_with_loss(self,hid_code_batch):
        sparse_hid=self.encoder(hid_code_batch)
        recon_hid_code=self.decoder(self.norm_hid(sparse_hid))

        loss_recon=paddle.mean((hid_code_batch-recon_hid_code)**2)

        # fire_per_data=paddle.mean(sparse_hid,axis=1,keepdim=True)
        # loss_data_fire=-paddle.mean(self.sparse_ratio*paddle.log(fire_per_data)+(1-self.sparse_ratio)*paddle.log(1-fire_per_data))
        code1=paddle.mean(sparse_hid**2,axis=1,keepdim=True)
        code2=paddle.mean((1-sparse_hid)**2,axis=1,keepdim=True)
        loss_data_fire=(self.kl_loss(code1,self.sparse_ratio)+self.kl_loss(code2,1-self.sparse_ratio))/2
        

        fire_per_cell=paddle.mean(sparse_hid,axis=0,keepdim=True)

        "fast"
        avg_fire_rate_fast=self.avg_fire_rate_fast*self.avg_fire_gamma_fast+(1-self.avg_fire_gamma_fast)*fire_per_cell
        loss_cell_fire_fast=-paddle.mean(self.sparse_ratio*paddle.log(avg_fire_rate_fast+1e-8)+(1-self.sparse_ratio)*paddle.log(1-avg_fire_rate_fast+1e-8))
        self.avg_fire_rate_fast=avg_fire_rate_fast.detach()

        "slow"
        avg_fire_rate_slow=self.avg_fire_rate_slow*self.avg_fire_gamma_slow+(1-self.avg_fire_gamma_slow)*fire_per_cell
        loss_cell_fire_slow=-paddle.mean(self.sparse_ratio*paddle.log(avg_fire_rate_slow+1e-8)+(1-self.sparse_ratio)*paddle.log(1-avg_fire_rate_slow+1e-8))
        self.avg_fire_rate_slow=avg_fire_rate_slow.detach()
        # print(paddle.min(fire_per_cell).numpy(),paddle.max(fire_per_cell).numpy())

        "current"
        loss_cell_fire_current=-paddle.mean(self.sparse_ratio*paddle.log(fire_per_cell+1e-8)+(1-self.sparse_ratio)*paddle.log(1-fire_per_cell+1e-8))
        # "9999"
        # self.avg_fire_rate_9999=self.avg_fire_rate_9999*self.avg_fire_gamma_9999+(1-self.avg_fire_gamma_9999)*fire_per_cell
        # self.avg_fire_rate_9999=self.avg_fire_rate_9999.detach()
        # loss_cell_fire_9999=-paddle.mean(self.sparse_ratio*paddle.log(self.avg_fire_rate_9999+1e-8)+(1-self.sparse_ratio)*paddle.log(1-self.avg_fire_rate_9999+1e-8))


        loss_ratio1=0.1
        loss_cell_fire=loss_ratio1*loss_cell_fire_fast/(1-self.avg_fire_gamma_fast)+(1-loss_ratio1)*loss_cell_fire_slow/(1-self.avg_fire_gamma_slow)
        # loss_cell_fire=loss_cell_fire_current

        loss_and_pred_dict={
            "loss_recon":loss_recon,
            "loss_data_fire":loss_data_fire,
            "loss_cell_fire_slow":loss_cell_fire_slow,
            "loss_cell_fire_fast":loss_cell_fire_fast,
            "loss_cell_fire_current":loss_cell_fire_current,
            "loss_cell_fire":loss_cell_fire,
            "recon_hid_code":recon_hid_code,
            "sparse_hid":sparse_hid,

        }
        return loss_and_pred_dict


    def train_one_step(self,hid_code_batch):
        loss_and_pred_dict=self.pred_with_loss(hid_code_batch)

        loss_recon=loss_and_pred_dict["loss_recon"]
        loss_data_fire=loss_and_pred_dict["loss_data_fire"]
        loss_cell_fire=loss_and_pred_dict["loss_cell_fire"]
        loss_cell_fire_slow=loss_and_pred_dict["loss_cell_fire_slow"]
        loss_cell_fire_fast=loss_and_pred_dict["loss_cell_fire_fast"]

        loss=loss_recon+(loss_data_fire+loss_cell_fire)*1.0
        loss_list=[loss.numpy(),loss_recon.numpy(),loss_data_fire.numpy(),loss_cell_fire_slow.numpy()]
        loss_list=np.reshape(loss_list,[-1])
        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()

        return loss_list
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