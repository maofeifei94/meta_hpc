import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
import numpy as np
import os
from mlutils.ml import dense_block,ReparamNormal,gauss_KL,DataFormat,ReplayMemory
from mlutils.ml import Basic_Module, ReplayMemory,dense_block,DataFormat,Reslayer,res_block,ResNormBlock,HierarchicalGru
from mlutils.ml import ModelSaver,GroupGru
nl_func=nn.Tanh()
class svaepred_info():
    len_warmup=128
    len_max_history=512
    
    train_batch_size=4

    "input dim"
    state_vec_dim=128
    action_vec_dim=4

    "train param"
    train_lr=0.0001
    train_kl_loss_ratio=1e-8
    # gloinfo

    "input fc"
    ifc_input_dim=state_vec_dim+action_vec_dim
    ifc_mid_dim=128
    ifc_mid_layers=4
    ifc_output_dim=128
    "gru"
    gru_dim=256
    
    "pred fc"
    ofc_input_dim=gru_dim
    ofc_mid_dim=128
    ofc_mid_layers=6
    ofc_output_dim=state_vec_dim
class GruPred(nn.Layer,ModelSaver):
    def __init__(self,info:svaepred_info):
        super().__init__()
        self.info=info
        "global info vae"
        self.ifc=self.resnorm_net(
            self.info.ifc_input_dim,
            self.info.ifc_mid_dim,
            self.info.ifc_mid_layers,
            self.info.ifc_output_dim,
            True,
            nl_func)
        self.gru=GroupGru(self.info.ifc_output_dim,self.info.gru_dim,1)
        self.ofc=self.resnorm_net(
            self.info.ofc_input_dim,
            self.info.ofc_mid_dim,
            self.info.ofc_mid_layers,
            self.info.ofc_output_dim,
            False,
            None)
       
       
        self.optimizer=optim.Adam(self.info.train_lr,parameters=[
            *self.ifc.parameters(),
            *self.gru.parameters(),
            *self.ofc.parameters(),
            ]) 
    def res_net(self,input_dim,mid_dim,mid_layers,output_dim,act_output):
        return res_block([input_dim,*[mid_dim]*mid_layers,output_dim],act=nl_func,act_output=act_output)
    def resnorm_net(self,input_dim,mid_dim,mid_layers,output_dim,last_layer_norm,act_output):
        return ResNormBlock([input_dim,*[mid_dim]*mid_layers,output_dim],act=nl_func,last_layer_norm=last_layer_norm,act_output=act_output)
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
    def _cal_h(self,s_history,a_history,init_h):
        sa_his=paddle.concat([s_history,a_history],axis=-1)
        sa_feature=self.btd_layer(self.ifc,sa_his)
        gru_h,next_gru_h=self.gru(sa_feature,init_h)
        return gru_h,next_gru_h
    def _cal_pred(self,gru_h):
        pred=self.btd_layer(self.ofc,gru_h)
        return pred
    def train_with_warmup(self,s_history,a_history,pre_h):
        if isinstance(s_history,paddle.Tensor):
            pass
        else:
            s_history=paddle.to_tensor(s_history,'float32')
            a_history=paddle.to_tensor(a_history,'float32')

        _b,_t,_=s_history.shape
        gru_h,_=self._cal_h(s_history,a_history,pre_h)
        pred=self._cal_pred(gru_h)[:,self.info.len_warmup-1:-1]
        target=s_history[:,self.info.len_warmup:]
        total_loss=paddle.mean((pred-target)**2)
        self.optimizer.clear_grad()
        total_loss.backward()
        self.optimizer.step()

        return [total_loss.numpy()[0]]

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