import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
from paddle.jit import to_static
import numpy as np
import os
from mlutils.ml import dense_block,ReparamNormal,gauss_KL,DataFormat,ReplayMemory
from mlutils.ml import Basic_Module, ReplayMemory,dense_block,DataFormat,Reslayer,res_block,ResNormBlock,HierarchicalStepGru,HierarchicalGru
from mlutils.ml import ModelSaver,GroupGru

nl_func=nn.Tanh()
class svaepred_info():

    "input dim"
    gloinfo_gauss_dim=256
    state_vec_dim=24+1
    action_vec_dim=4

    "train param"
    train_lr=0.0001
    train_kl_loss_ratio=1e-10

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
    "pred fc"
    Q_dim=96
    
    V_dim=64
    V_num=64
    
    K_dim=Q_dim*V_num

    # pred_fc_input_dim=locinfo_gru_dim+gloinfo_gauss_dim
    # pred_fc_mid_dim=256
    # pred_fc_mid_layers=4
    # pred_fc_output_dim=state_vec_dim

    pred_Qnet_input_dim=locinfo_gru_dim
    pred_Qnet_mid_dim=256
    pred_Qnet_mid_layers=2
    pred_Qnet_output_dim=Q_dim

    pred_Knet_input_dim=gloinfo_gauss_dim
    pred_Knet_mid_dim=256
    pred_Knet_mid_layers=2
    pred_Knet_output_dim=K_dim

    pred_Vnet_input_dim=gloinfo_gauss_dim
    pred_Vnet_mid_dim=256
    pred_Vnet_mid_layers=2
    pred_Vnet_output_dim=V_dim




    


class SeqVaePred(nn.Layer,ModelSaver):
    def __init__(self,info:svaepred_info):
        super().__init__()
        self.info=info
        "global info vae"
       
        self.gloinfo_input_fc=self.resnorm_net(self.info.gloinfo_ifc_input_dim,self.info.gloinfo_ifc_mid_dim,self.info.gloinfo_ifc_mid_layers,self.info.gloinfo_ifc_output_dim,True,nl_func)
        # self.gloinfo_gru=nn.GRU(self.info.gloinfo_ifc_output_dim,self.info.gloinfo_gru_dim)
        # self.gloinfo_gru=GroupGru(self.info.gloinfo_ifc_output_dim,self.info.gloinfo_gru_dim,1)
        self.gloinfo_gru=GroupGru(self.info.gloinfo_ifc_output_dim,self.info.gloinfo_gru_dim,1)
        print("gloinfo gru use HierarchicalGru 4,4,8")
    
        self.gloinfo_gauss=self.resnorm_net(self.info.gloinfo_gfc_input_dim,self.info.gloinfo_gfc_mid_dim,self.info.gloinfo_gfc_mid_layers,self.info.gloinfo_gfc_output_dim,False,None)
        "location info"
        # self.locinfo_input_fc=res_block([self.info.locinfo_ifc_input_dim,*[self.info.locinfo_ifc_mid_dim]*self.info.locinfo_ifc_mid_layers,self.info.locinfo_ifc_output_dim],act=nl_func,act_output=nl_func)
        self.locinfo_input_fc=self.resnorm_net(self.info.locinfo_ifc_input_dim,self.info.locinfo_ifc_mid_dim,self.info.locinfo_ifc_mid_layers,self.info.locinfo_ifc_output_dim,True,nl_func)
        # self.locinfo_gru=nn.GRU(self.info.locinfo_ifc_output_dim,self.info.locinfo_gru_dim)
        self.locinfo_gru=GroupGru(self.info.locinfo_ifc_output_dim,self.info.locinfo_gru_dim,1)
        "pred"
        # self.pred_fc=dense_block([self.info.pred_fc_input_dim,*[self.info.pred_fc_mid_dim]*self.info.pred_fc_mid_layers,self.info.pred_fc_output_dim],act_output=None)
        self.pred_Qnet=self.resnorm_net(
            self.info.pred_Qnet_input_dim,self.info.pred_Qnet_mid_dim,
            self.info.pred_Qnet_mid_layers,self.info.pred_Qnet_output_dim,False,act_output=None)
        self.pred_Knet=self.resnorm_net(
            self.info.pred_Knet_input_dim,self.info.pred_Knet_mid_dim,
            self.info.pred_Knet_mid_layers,self.info.pred_Knet_output_dim,False,act_output=None)
        self.pred_Vnet=self.resnorm_net(
            self.info.pred_Vnet_input_dim,self.info.pred_Vnet_mid_dim,
            self.info.pred_Vnet_mid_layers,self.info.pred_Vnet_output_dim,False,act_output=None)
        self.pred_fc=self.resnorm_net(
            self.info.pred_fc_input_dim,self.info.pred_fc_mid_dim,
            self.info.pred_fc_mid_layers,self.info.pred_fc_output_dim,False,act_output=None)
        # print("gloinfo_input_fc",self.gloinfo_input_fc.state_dict().keys())
        # print("locinfo_gru",self.locinfo_gru.state_dict().keys())
        # print("locinfo_gru",self.locinfo_gru.parameters())
        self.optimizer=optim.Adam(self.info.train_lr,parameters=[
            *self.gloinfo_input_fc.parameters(),
            *self.gloinfo_gru.parameters(),
            *self.gloinfo_gauss.parameters(),
            *self.locinfo_input_fc.parameters(),
            *self.locinfo_gru.parameters(),
            *self.pred_Qnet.parameters(),
            *self.pred_Knet.parameters(),
            *self.pred_Vnet.parameters(),
            *self.pred_fc.parameters(),
            ]) 
        # print(self.state_dict().keys())
        # print(self.optimizer._param_dict.keys())
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
        Qmat=self.btd_layer(self.pred_Qnet,local_h).reshape([_b,_t_loc,1,self.info.Q_dim])
        "global的mat只需计算一次"
        Kmat=self.btd_layer(self.pred_Knet,gauss_sample).reshape([_b,_t_glo,1,self.info.Q_dim,self.info.V_num])
        Kmat=paddle.tile(Kmat,[1,1,_t_loc//_t_glo,1,1]).reshape([_b,_t_loc,self.info.Q_dim,self.info.V_num])
        Vmat=self.btd_layer(self.pred_Vnet,gauss_sample).reshape([_b,_t_glo,1,self.info.V_num,self.info.V_dim])
        Vmat=paddle.tile(Vmat,[1,1,_t_loc//_t_glo,1,1]).reshape([_b,_t_loc,self.info.V_num,self.info.V_dim])

        attention=F.softmax(paddle.matmul(Qmat,Kmat)/(Kmat.shape[-1]**0.5),axis=-1)
        Vvec=paddle.squeeze(paddle.matmul(attention,Vmat),axis=2)

        return self.btd_layer(self.pred_fc,paddle.concat([Vvec,local_h],axis=-1))
    def _sample_from_gauss_param(self,gauss_param,sample_num=1):
        gauss_mean,gauss_logvar=gauss_param[:,:,:self.info.gloinfo_gauss_dim],gauss_param[:,:,self.info.gloinfo_gauss_dim:]
        gauss_sample=ReparamNormal(gauss_mean.shape).sample(gauss_mean,gauss_logvar,sample_num)
        return gauss_sample
    def reset(self):
        self.global_h=paddle.zeros([1,1,self.info.gloinfo_gru_dim])
        self.local_h=paddle.zeros([1,1,self.info.locinfo_gru_dim])
        ec_vec=np.zeros([self.info.gloinfo_gauss_dim*2+self.info.locinfo_gru_dim])
        return ec_vec
    @to_static
    def forward(self,s_history,a_history,reward):
        "只支持batchsize=1"

        # s_history=s_history if isinstance(s_history,paddle.Tensor) else paddle.to_tensor(s_history,'float32')
        # a_history=a_history if isinstance(a_history,paddle.Tensor) else paddle.to_tensor(a_history,'float32')
        # r_history=reward if isinstance(reward,paddle.Tensor) else paddle.to_tensor(reward,'float32')
        # print(s_history,r_history)
        # s_history=s_history.reshape([1,1,-1])
        # a_history=a_history.reshape([1,1,-1])
        # r_history=r_history.reshape([1,1,-1])

        s_history=paddle.concat([s_history,reward],axis=-1)

        global_h,_=self._cal_global_h(s_history,a_history,self.global_h)
        global_gauss_param=self._cal_global_gauss(global_h)
        
        local_h,_=self._cal_local_h(s_history,a_history,self.local_h)

        self.global_h,self.local_h=global_h,local_h
        ec_vec=paddle.concat([global_gauss_param,local_h],axis=-1).numpy().reshape([-1])
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


        global_h,_=self._cal_global_h(s_history,a_history,pre_global_h)
        global_gauss_param=self._cal_global_gauss(global_h)
        global_gauss_sample=self._sample_from_gauss_param(global_gauss_param)
        final_global_info=global_gauss_sample[:,-1:,:]
        
        local_h,_=self._cal_local_h(s_history,a_history,pre_local_h)

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


        global_h,_=self._cal_global_h(s_history,a_history,pre_global_h)
        global_gauss_param=self._cal_global_gauss(global_h)
        local_h,_=self._cal_local_h(s_history,a_history,pre_local_h)

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


        global_h,_=self._cal_global_h(s_history,a_history,pre_global_h)
        global_gauss_param=self._cal_global_gauss(global_h)
        local_h,_=self._cal_local_h(s_history,a_history,pre_local_h)

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
    def get_sd_mean(self):
        mean_value=0
        for key in self.state_dict().keys():
            mean_value+=paddle.mean(paddle.abs(self.state_dict()[key]))
        return mean_value.numpy()
    def save_model(self,save_dir,iter_num):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"start_save_model {save_dir}/{iter_num}_model.pdparams")
        paddle.save(self.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")
        print(f"save model {save_dir}/{iter_num}_model.pdparams success")
        
    def load_model(self,save_dir,iter_num):
        print(f"start load_model {save_dir}/{iter_num}_model.pdparams")
        try:
            params_state = paddle.load(path=f"{save_dir}/{iter_num}_model.pdparams")
            self.set_state_dict(params_state)
            print(f"load model {save_dir}/{iter_num}_model.pdparams success")
        except:
            print(f"load Error:load model {save_dir}/{iter_num}_model.pdparams failed")
    def update_model(self,state_dict):
        "调用ModelSaver类的方法"
        super().update_model(model=self,state_dict=state_dict)
    def update_model_from_np(self,state_dict_np):
        "调用ModelSaver类的方法"
        super().update_model_from_np(model=self,state_dict_np=state_dict_np)
    def send_model_to_np(self):
        "调用ModelSaver类的方法"
        return super().send_model_to_np(model=self)