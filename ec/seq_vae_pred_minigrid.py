import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
import numpy as np
import os
from mlutils.ml import dense_block,ReparamNormal,gauss_KL,DataFormat,ReplayMemory
from mlutils.ml import Basic_Module, ReplayMemory,dense_block,DataFormat,Reslayer
from mlutils.ml import ModelSaver
nl_func=nn.LeakyReLU(0.1)
class svaepred_info():
    nl_func=nn.LeakyReLU(0.1)

    "input dim"
    gloinfo_gauss_dim=256
    state_vec_dim=256
    action_vec_dim=4

    "train param"
    train_lr=0.0001
    train_history_len=256
    train_history_warmup_len=64

    # gloinfo
    "input fc"
    gloinfo_ifc_input_dim=state_vec_dim+action_vec_dim
    gloinfo_ifc_mid_dim=128
    gloinfo_ifc_mid_layers=3
    gloinfo_ifc_output_dim=128
    "gru"
    gloinfo_gru_dim=128
    "gauss_fc"
    gloinfo_gfc_input_dim=gloinfo_gru_dim
    gloinfo_gfc_mid_dim=128
    gloinfo_gfc_mid_layers=3
    gloinfo_gfc_output_dim=gloinfo_gauss_dim*2

    # locinfo
    "input fc"
    locinfo_ifc_input_dim=state_vec_dim+action_vec_dim
    locinfo_ifc_mid_dim=128
    locinfo_ifc_mid_layers=3
    locinfo_ifc_output_dim=128
    "gru"
    locinfo_gru_dim=128

    #predfc
    pred_fc_input_dim=locinfo_gru_dim+gloinfo_gauss_dim
    pred_fc_mid_dim=256
    pred_fc_mid_layers=4
    pred_fc_output_dim=state_vec_dim




    


class SeqVaePred(nn.Layer,ModelSaver):
    def __init__(self,info:svaepred_info):
        super().__init__()
        self.info=info
        "global info vae"
        self.gloinfo_input_fc=dense_block([self.info.gloinfo_ifc_input_dim,*[self.info.gloinfo_ifc_mid_dim]*self.info.gloinfo_ifc_mid_layers,self.info.gloinfo_ifc_output_dim],act_output=nl_func)
        self.gloinfo_gru=nn.GRU(self.info.gloinfo_ifc_output_dim,self.info.gloinfo_gru_dim)
        self.gloinfo_gauss=dense_block([self.info.gloinfo_gfc_input_dim,*[self.info.gloinfo_gfc_mid_dim]*self.info.gloinfo_gfc_mid_layers,self.info.gloinfo_gfc_output_dim])

        "location info"
        self.locinfo_input_fc=dense_block([self.info.locinfo_ifc_input_dim,*[self.info.locinfo_ifc_mid_dim]*self.info.locinfo_ifc_mid_layers,self.info.locinfo_ifc_output_dim],act_output=nl_func)
        self.locinfo_gru=nn.GRU(self.info.locinfo_ifc_output_dim,self.info.locinfo_gru_dim)

        "pred"
        self.pred_fc=dense_block([self.info.pred_fc_input_dim,*[self.info.pred_fc_mid_dim]*self.info.pred_fc_mid_layers,self.info.pred_fc_output_dim],act_output=nn.Sigmoid())

        self.optimizer=optim.Adam(self.info.train_lr,parameters=[
            *self.gloinfo_input_fc.parameters(),
            *self.gloinfo_gru.parameters(),
            *self.gloinfo_gauss.parameters(),
            *self.locinfo_input_fc.parameters(),
            *self.locinfo_gru.parameters(),
            *self.pred_fc.parameters(),
            ]) 
        self.reset()
    def btd_layer(self,net,x):
        batch,time,dim=x.shape
        return net(x.reshape([batch*time,dim])).reshape([batch,time,-1])
    def bttd_layer(self,net,x):
        batch,time1,time2,dim=x.shape
        return net(x.reshape([batch*time1*time2,dim])).reshape([batch,time1,time2,-1])
    def btchw_layer(self,convnet,x):
        batch,time,c,h,w=x.shape
        conv_result=convnet(x.reshape([batch*time,c,h,w]))
        _,newc,newh,neww=conv_result.shape
        return conv_result.reshape([batch,time,newc,newh,neww])
    def bttchw_layer(self,convnet,x):
        batch,time1,time2,c,h,w=x.shape
        conv_result=convnet(x.reshape([batch*time1*time2,c,h,w]))
        _,newc,newh,neww=conv_result.shape
        return conv_result.reshape([batch,time1,time2,newc,newh,neww])

    def _cal_global_h(self,s_history,a_history,init_h):
        sa_his=paddle.concat([s_history,a_history],axis=-1)
        sa_feature=self.btd_layer(self.gloinfo_input_fc,sa_his)
        global_h,_=self.gloinfo_gru(sa_feature,init_h)
        return global_h
    def _cal_local_h(self,s_history,a_history,init_h):
        sa_his=paddle.concat([s_history,a_history],axis=-1)
        sa_feature=self.btd_layer(self.locinfo_input_fc,sa_his)
        local_h,_=self.locinfo_gru(sa_feature,init_h)
        return local_h
    def _cal_global_gauss(self,global_h):
        global_gauss_param=self.btd_layer(self.gloinfo_gauss,global_h)
        # global_gauss_mean,gloal_gauss_logvar=global_gauss_param[:,:,:self.info.gloinfo_gauss_dim],global_gauss_param[:,:,self.info.gloinfo_gauss_dim:]
        return global_gauss_param
    def _cal_pred(self,gauss_sample,local_h):
        return self.btd_layer(self.pred_fc,paddle.concat([gauss_sample,local_h],axis=-1))
    def _sample_from_gauss_param(self,gauss_param,sample_num=1):
        gauss_mean,gauss_logvar=gauss_param[:,:,:self.info.gloinfo_gauss_dim],gauss_param[:,:,self.info.gloinfo_gauss_dim:]
        gauss_sample=ReparamNormal(gauss_mean.shape).sample(gauss_mean,gauss_logvar,sample_num)
        return gauss_sample
    def reset(self):
        self.global_h=paddle.zeros([1,1,self.info.gloinfo_gru_dim])
        self.local_h=paddle.zeros([1,1,self.info.locinfo_gru_dim])
        ec_vec=np.zeros([self.info.gloinfo_gauss_dim*2+self.info.locinfo_gru_dim])
        return ec_vec
    def forward(self,s_history,a_history):
        if isinstance(s_history,paddle.Tensor):
            s_history=s_history
            a_history=a_history
        else:
            s_history=paddle.to_tensor(s_history,'float32')
            a_history=paddle.to_tensor(a_history,'float32')
        s_history=s_history.reshape([1,1,-1])
        a_history=a_history.reshape([1,1,-1])
        global_h=self._cal_global_h(s_history,a_history,self.global_h)
        global_gauss=self._cal_global_gauss(global_h)
        local_h=self._cal_local_h(s_history,a_history,self.local_h)
        self.global_h,self.local_h=global_h,local_h
        ec_vec=paddle.concat([global_gauss,local_h],axis=-1).numpy().reshape([-1])
        return ec_vec
    def train_rmp(self,rpm):
        pass
    def train(self,s_history,a_history,pre_global_h,pre_local_h):
        if isinstance(s_history,paddle.Tensor):
            pass
        else:
            s_history=paddle.to_tensor(s_history,'float32')
            a_history=paddle.to_tensor(a_history,'float32')

        _b,_t,_=s_history.shape

        if pre_global_h is None:
            pre_global_h=paddle.zeros([1,_b,self.info.gloinfo_gru_dim])
            pre_local_h=paddle.zeros([1,_b,self.info.locinfo_gru_dim])
        elif isinstance(pre_global_h,paddle.Tensor):
            pass
        else:
            pre_global_h=paddle.to_tensor(pre_global_h,'float32')
            pre_local_h=paddle.to_tensor(pre_local_h,'float32')


        global_h=self._cal_global_h(s_history,a_history,pre_global_h)
        global_gauss_param=self._cal_global_gauss(global_h)
        local_h=self._cal_local_h(s_history,a_history,pre_local_h)

        global_gauss_sample=self._sample_from_gauss_param(global_gauss_param)

        final_global_info=global_gauss_sample[:,-1:,:]

        "pred"
        pred=self._cal_pred(paddle.tile(final_global_info,[1,_t,1]),local_h)
        pred=pred[:,:-1,:]
        pred_target=s_history[:,1:]
        pred_loss=paddle.mean((pred-pred_target).pow(2))

        "kl"
        mean_all=global_gauss_param[:,:,:self.info.gloinfo_gauss_dim]
        sigma_all=paddle.exp(0.5*global_gauss_param[:,:,self.info.gloinfo_gauss_dim:])
        mean_final=mean_all[:,-1:]
        sigma_final=sigma_all[:,-1:]
        kl_loss=paddle.mean(gauss_KL(mean_final,sigma_final,mean_all,sigma_all))

        total_loss=pred_loss+self.info.train_kl_loss_ratio*kl_loss

        self.optimizer.clear_grad()
        total_loss.backward()
        self.optimizer.step()

        return [total_loss.numpy()[0],pred_loss.numpy()[0],kl_loss.numpy()[0]]
    def train_multi_sample(self,s_history,a_history,pre_global_h,pre_local_h):
        if isinstance(s_history,paddle.Tensor):
            pass
        else:
            s_history=paddle.to_tensor(s_history,'float32')
            a_history=paddle.to_tensor(a_history,'float32')

        _b,_t,_=s_history.shape

        if pre_global_h is None:
            pre_global_h=paddle.zeros([1,_b,self.info.gloinfo_gru_dim])
            pre_local_h=paddle.zeros([1,_b,self.info.locinfo_gru_dim])
        elif isinstance(pre_global_h,paddle.Tensor):
            pass
        else:
            pre_global_h=paddle.to_tensor(pre_global_h,'float32')
            pre_local_h=paddle.to_tensor(pre_local_h,'float32')


        global_h=self._cal_global_h(s_history,a_history,pre_global_h)
        global_gauss_param=self._cal_global_gauss(global_h)
        local_h=self._cal_local_h(s_history,a_history,pre_local_h)

        "change sample_num"
        sample_num=20
        global_gauss_sample=self._sample_from_gauss_param(global_gauss_param,sample_num=sample_num)
        final_global_info=global_gauss_sample[:,:,-1:,:]
        final_global_info=paddle.tile(final_global_info,[1,1,_t,1])

        local_h=paddle.tile(paddle.reshape(local_h,[1,*local_h.shape]),[sample_num,1,1,1])

        

        # print("final_global_info.shape,local_h.shape=",final_global_info.shape,local_h.shape)
        "pred"
        pred=self._cal_pred(final_global_info.reshape([sample_num*_b,_t,-1]),local_h.reshape([sample_num*_b,_t,-1]))
        pred=pred[:,:-1,:].reshape([sample_num,_b,_t-1,-1])
        # print("pred.shape=",pred.shape)
        pred_target=s_history[:,1:]
        pred_target=paddle.reshape(pred_target,[1,*pred_target.shape])
        pred_loss=paddle.mean((pred-pred_target).pow(2))

        "kl"
        mean_all=global_gauss_param[:,:,:self.info.gloinfo_gauss_dim]
        sigma_all=paddle.exp(0.5*global_gauss_param[:,:,self.info.gloinfo_gauss_dim:])
        mean_final=mean_all[:,-1:]
        sigma_final=sigma_all[:,-1:]
        kl_loss=paddle.mean(gauss_KL(mean_final,sigma_final,mean_all,sigma_all))

        total_loss=pred_loss+self.info.train_kl_loss_ratio*kl_loss

        self.optimizer.clear_grad()
        total_loss.backward()
        self.optimizer.step()

        return [total_loss.numpy()[0],pred_loss.numpy()[0],kl_loss.numpy()[0]]
    def test(self,s_history,a_history,pre_global_h=None,pre_local_h=None):
        if isinstance(s_history,paddle.Tensor):
            pass
        else:
            s_history=paddle.to_tensor(s_history,'float32')
            a_history=paddle.to_tensor(a_history,'float32')

        _b,_t,_=s_history.shape

        if pre_global_h is None:
            pre_global_h=paddle.zeros([1,_b,self.info.gloinfo_gru_dim])
            pre_local_h=paddle.zeros([1,_b,self.info.locinfo_gru_dim])
        elif isinstance(pre_global_h,paddle.Tensor):
            pass
        else:
            pre_global_h=paddle.to_tensor(pre_global_h,'float32')
            pre_local_h=paddle.to_tensor(pre_local_h,'float32')


        global_h=self._cal_global_h(s_history,a_history,pre_global_h)
        global_gauss_param=self._cal_global_gauss(global_h)
        local_h=self._cal_local_h(s_history,a_history,pre_local_h)

        global_gauss_sample=self._sample_from_gauss_param(global_gauss_param)

        final_global_info=global_gauss_sample[:,-1:,:]

        "pred"
        pred=self._cal_pred(paddle.tile(final_global_info,[1,_t,1]),local_h)
        pred=pred[:,:-1,:]
        pred_target=s_history[:,1:]
        pred_loss=paddle.mean((pred-pred_target).pow(2))

        "kl"
        mean_all=global_gauss_param[:,:,:self.info.gloinfo_gauss_dim]
        sigma_all=paddle.exp(0.5*global_gauss_param[:,:,self.info.gloinfo_gauss_dim:])
        mean_final=mean_all[:,-1:]
        sigma_final=sigma_all[:,-1:]
        kl_loss=paddle.mean(gauss_KL(mean_final,sigma_final,mean_all,sigma_all))

        total_loss=pred_loss+self.info.train_kl_loss_ratio*kl_loss



        return pred.numpy(),pred_target.numpy()

    def save_model(self,save_dir,iter_num):
        "调用ModelSaver类的方法"
        super().save_model(model=self,save_dir=save_dir,iter_num=iter_num)
    def load_model(self,save_dir,iter_num):
        "调用ModelSaver类的方法"
        super().load_model(model=self,save_dir=save_dir,iter_num=iter_num)
    def update_model(self,state_dict):
        "调用ModelSaver类的方法"
        super().update_model(model=self,state_dict=state_dict)
    def update_model_from_np(self,state_dict_np):
        "调用ModelSaver类的方法"
        super().update_model_from_np(model=self,state_dict_np=state_dict_np)
    def send_model_to_np(self):
        "调用ModelSaver类的方法"
        return super().send_model_to_np(model=self)