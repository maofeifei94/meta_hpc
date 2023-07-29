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

    "input dim"
    gloinfo_gauss_dim=128
    state_vec_dim=128+1
    action_vec_dim=4

    "train param"
    train_lr=0.0001
    train_kl_loss_ratio=0

    # gloinfo
    "input fc"
    gloinfo_ifc_input_dim=state_vec_dim+action_vec_dim
    gloinfo_ifc_mid_dim=128
    gloinfo_ifc_mid_layers=2
    gloinfo_ifc_output_dim=128
    "gru"
    gloinfo_gru_dim=128
    "gauss_fc"
    gloinfo_gfc_input_dim=gloinfo_gru_dim
    gloinfo_gfc_mid_dim=128
    gloinfo_gfc_mid_layers=4
    gloinfo_gfc_output_dim=gloinfo_gauss_dim*2

    # locinfo
    "input fc"
    locinfo_ifc_input_dim=state_vec_dim+action_vec_dim
    locinfo_ifc_mid_dim=128
    locinfo_ifc_mid_layers=4
    locinfo_ifc_output_dim=128
    "gru"
    locinfo_gru_dim=128

    #predfc
    "pred fc"
    pred_fc_input_dim=locinfo_gru_dim+gloinfo_gauss_dim
    pred_fc_mid_dim=128
    pred_fc_mid_layers=6
    pred_fc_output_dim=state_vec_dim




    


class SeqVaePred(nn.Layer,ModelSaver):
    def __init__(self,info:svaepred_info):
        super().__init__()
        self.info=info
        "global info vae"
       
        self.gloinfo_input_fc=self.resnorm_net(self.info.gloinfo_ifc_input_dim,self.info.gloinfo_ifc_mid_dim,self.info.gloinfo_ifc_mid_layers,self.info.gloinfo_ifc_output_dim,True,nl_func)
        # self.gloinfo_gru=nn.GRU(self.info.gloinfo_ifc_output_dim,self.info.gloinfo_gru_dim)
        # self.gloinfo_gru=GroupGru(self.info.gloinfo_ifc_output_dim,self.info.gloinfo_gru_dim,1)
        self.gloinfo_gru=GroupGru(self.info.gloinfo_ifc_output_dim,self.info.gloinfo_gru_dim,1)
        
        self.gloinfo_gauss=self.resnorm_net(self.info.gloinfo_gfc_input_dim,self.info.gloinfo_gfc_mid_dim,self.info.gloinfo_gfc_mid_layers,self.info.gloinfo_gfc_output_dim,False,None)
        "location info"
        # self.locinfo_input_fc=res_block([self.info.locinfo_ifc_input_dim,*[self.info.locinfo_ifc_mid_dim]*self.info.locinfo_ifc_mid_layers,self.info.locinfo_ifc_output_dim],act=nl_func,act_output=nl_func)
        self.locinfo_input_fc=self.resnorm_net(self.info.locinfo_ifc_input_dim,self.info.locinfo_ifc_mid_dim,self.info.locinfo_ifc_mid_layers,self.info.locinfo_ifc_output_dim,True,nl_func)
        # self.locinfo_gru=nn.GRU(self.info.locinfo_ifc_output_dim,self.info.locinfo_gru_dim)
        self.locinfo_gru=GroupGru(self.info.locinfo_ifc_output_dim,self.info.locinfo_gru_dim,1)
        "pred"
        self.pred_fc=self.resnorm_net(
            self.info.pred_fc_input_dim,self.info.pred_fc_mid_dim,
            self.info.pred_fc_mid_layers,self.info.pred_fc_output_dim,False,act_output=None)
        

        self.optimizer=optim.Adam(self.info.train_lr,parameters=[
            *self.gloinfo_input_fc.parameters(),
            *self.gloinfo_gru.parameters(),
            *self.gloinfo_gauss.parameters(),
            *self.locinfo_input_fc.parameters(),
            *self.locinfo_gru.parameters(),
            *self.pred_fc.parameters(),
            ]) 
        self.train_step_num=0
        self.reset()
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

    def _cal_global_h(self,s_history,a_history,init_h):
        sa_his=paddle.concat([s_history,a_history],axis=-1)
        sa_feature=self.btd_layer(self.gloinfo_input_fc,sa_his)
        global_h,next_gru_h=self.gloinfo_gru(sa_feature,init_h)
        # print(global_h.shape)
        return global_h,next_gru_h
    def _cal_local_h(self,s_history,a_history,init_h):
        sa_his=paddle.concat([s_history,a_history],axis=-1)
        sa_feature=self.btd_layer(self.locinfo_input_fc,sa_his)
        local_h,next_local_gru_h=self.locinfo_gru(sa_feature,init_h)
        return local_h,next_local_gru_h
    def _cal_global_gauss(self,global_h):
        global_gauss_param=self.btd_layer(self.gloinfo_gauss,global_h)
        # global_gauss_mean,gloal_gauss_logvar=global_gauss_param[:,:,:self.info.gloinfo_gauss_dim],global_gauss_param[:,:,self.info.gloinfo_gauss_dim:]
        return global_gauss_param
    def _cal_pred(self,gauss_sample,local_h):
        _b,_t_loc,_=local_h.shape
        _b,_t_glo,_=gauss_sample.shape
        # print(local_h.shape)
        # print(gauss_sample.shape)
        gauss_sample=gauss_sample.reshape([_b,_t_glo,1,-1]).tile([1,1,_t_loc//_t_glo,1]).reshape([_b,_t_loc,-1])
        # print(gauss_sample.shape)
        return self.btd_layer(self.pred_fc,paddle.concat([local_h,gauss_sample],axis=-1))
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
    def train_with_warmup(self,s_history,a_history,pre_global_h,pre_local_h):
        optimizer_warmup_steps=1000
        if self.train_step_num<optimizer_warmup_steps:
            self.optimizer.set_lr(0.0)
        elif self.train_step_num==optimizer_warmup_steps:
            self.optimizer.set_lr(self.info.train_lr)
        else:
            pass
        self.train_step_num+=1
        # print(self.optimizer.get_lr())

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
        
        global_h,_=self._cal_global_h(s_history,a_history,pre_global_h)
        global_gauss_param=self._cal_global_gauss(global_h)
        global_gauss_sample=self._sample_from_gauss_param(global_gauss_param)
        # final_global_info=global_gauss_sample[:,-1:,:]
        
        local_h,_=self._cal_local_h(s_history,a_history,pre_local_h)

        "gather"
        len_warmup=self.info.len_warmup
        len_max_history=self.info.len_max_history
        len_pred=self.info.len_pred
        len_kl=self.info.len_kl
        interval_pred=self.info.interval_pred
        interval_kl=self.info.interval_kl

        index_global_info=paddle.to_tensor(np.concatenate([[glo_i] for glo_i in range(len_warmup+len_pred,len_max_history,interval_pred)]),'int32')
        index_loc_info=paddle.to_tensor(np.concatenate([list(range(glo_i-len_pred,glo_i)) for glo_i in range(len_warmup+len_pred,len_max_history,interval_pred)]),'int32')
        index_pred_target=index_loc_info+1

        index_kl_start=paddle.to_tensor(list(range(0,len_max_history-len_kl)),'int32').reshape([-1,1])
        index_kl_start=paddle.tile(index_kl_start,[1,len_kl//interval_kl]).reshape([-1])
        # index_kl_final=paddle.to_tensor(list(range(len_kl,len_max_history)),'int32')
        index_kl_final=paddle.to_tensor([list(range(interval_kl,len_kl+interval_kl,interval_kl)) for _ in range(len_max_history-len_kl)],'int32').reshape([-1])

        glo_info_for_pred=paddle.gather(global_gauss_sample,index_global_info,axis=1)
        loc_info_for_pred=paddle.gather(local_h,index_loc_info,axis=1)
        # print(glo_info_for_pred.shape,loc_info_for_pred.shape)
        "pred"
        pred=self._cal_pred(glo_info_for_pred,loc_info_for_pred)
        pred_target=paddle.gather(s_history,index_pred_target,axis=1)
        pred_loss=paddle.mean((pred-pred_target).pow(2))

        "kl"
        gauss_param_start=paddle.gather(global_gauss_param,index_kl_start,axis=1)
        gauss_param_final=paddle.gather(global_gauss_param,index_kl_final,axis=1)

        mean_start=gauss_param_start[:,:,:self.info.gloinfo_gauss_dim]
        sigma_start=paddle.exp(gauss_param_start[:,:,self.info.gloinfo_gauss_dim:])
        mean_final=gauss_param_final[:,:,:self.info.gloinfo_gauss_dim]
        sigma_final=paddle.exp(gauss_param_final[:,:,self.info.gloinfo_gauss_dim:])

        kl_loss=paddle.mean(gauss_KL(mean_final,sigma_final,mean_start,sigma_start))
        
        total_loss=pred_loss+self.info.train_kl_loss_ratio*kl_loss

        self.optimizer.clear_grad()
        total_loss.backward()
        self.optimizer.step()

        return [total_loss.numpy()[0],pred_loss.numpy()[0],kl_loss.numpy()[0]]
    def train_with_warmup_onesteppred(self,s_history,a_history,pre_global_h,pre_local_h):
        optimizer_warmup_steps=1000
        if self.train_step_num<optimizer_warmup_steps:
            self.optimizer.set_lr(0.0)
        elif self.train_step_num==optimizer_warmup_steps:
            self.optimizer.set_lr(self.info.train_lr)
        else:
            pass
        self.train_step_num+=1
        # print(self.optimizer.get_lr())

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
        
        global_h,_=self._cal_global_h(s_history,a_history,pre_global_h)
        global_gauss_param=self._cal_global_gauss(global_h)
        global_gauss_sample=self._sample_from_gauss_param(global_gauss_param)
        # final_global_info=global_gauss_sample[:,-1:,:]
        
        local_h,_=self._cal_local_h(s_history,a_history,pre_local_h)

        "gather"
        len_warmup=self.info.len_warmup
        len_max_history=self.info.len_max_history
        len_pred=self.info.len_pred
        interval_pred=self.info.interval_pred


        

       

        # print(glo_info_for_pred.shape,loc_info_for_pred.shape)
        "pred"
        index_global_info=paddle.to_tensor(np.concatenate([[glo_i] for glo_i in range(len_warmup+len_pred,len_max_history,interval_pred)]),'int32')
        index_loc_info=paddle.to_tensor(np.concatenate([list(range(glo_i-len_pred,glo_i)) for glo_i in range(len_warmup+len_pred,len_max_history,interval_pred)]),'int32')
        index_pred_target=index_loc_info+1

        glo_info_for_pred=paddle.gather(global_gauss_sample,index_global_info,axis=1)
        loc_info_for_pred=paddle.gather(local_h,index_loc_info,axis=1)

        pred=self._cal_pred(glo_info_for_pred,loc_info_for_pred)
        pred_target=paddle.gather(s_history,index_pred_target,axis=1)
        pred_loss=paddle.mean((pred-pred_target).pow(2))

        "kl"
        kl_steps_list=[1,2,4,8,16,32,64,128,256]
        index_kl_start_list=[]
        index_kl_final_list=[]
        for kl_step in kl_steps_list:
            index_kl_start_part=paddle.to_tensor(list(range(0,len_max_history-kl_step)),'int32')
            index_kl_final_part=index_kl_start_part+kl_step
            index_kl_start_list.append(index_kl_start_part)
            index_kl_final_list.append(index_kl_final_part)
        index_kl_start=paddle.concat(index_kl_start_list,axis=0)
        index_kl_final=paddle.concat(index_kl_final_list,axis=0)

        gauss_param_start=paddle.gather(global_gauss_param,index_kl_start,axis=1)
        gauss_param_final=paddle.gather(global_gauss_param,index_kl_final,axis=1)

        mean_start=gauss_param_start[:,:,:self.info.gloinfo_gauss_dim]
        sigma_start=paddle.exp(gauss_param_start[:,:,self.info.gloinfo_gauss_dim:])
        mean_final=gauss_param_final[:,:,:self.info.gloinfo_gauss_dim]
        sigma_final=paddle.exp(gauss_param_final[:,:,self.info.gloinfo_gauss_dim:])

        kl_loss=paddle.mean(gauss_KL(mean_final,sigma_final,mean_start,sigma_start))
        
        total_loss=pred_loss+self.info.train_kl_loss_ratio*kl_loss

        self.optimizer.clear_grad()
        total_loss.backward()
        self.optimizer.step()

        return [total_loss.numpy()[0],pred_loss.numpy()[0],kl_loss.numpy()[0]]
    
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
        global_gauss_sample=self._sample_from_gauss_param(global_gauss_param)
        final_global_info=paddle.tile(global_gauss_sample[:,-1:,:],[1,_t,1])
        
        local_h=self._cal_local_h(s_history,a_history,pre_local_h)

        "pred"
        pred=self._cal_pred(final_global_info,local_h)
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
        sample_num=2
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