import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
from paddle.jit import to_static
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
    pred_fc_input_dim=locinfo_gru_dim+gloinfo_gauss_dim+action_vec_dim
    pred_fc_mid_dim=128
    pred_fc_mid_layers=6
    pred_fc_output_dim=state_vec_dim

class Softabs(nn.Layer):
    def forward(self,x):
        abs_x=paddle.abs(x)
        return paddle.where(abs_x>1,abs_x*0.5+0.5,abs_x.pow(2))
class Equallayer(nn.Layer):
    def forward(self,x):
        return x
class ResNormBlock(nn.Layer):
    def __init__(self,input_dim,output_dim,act=nn.GELU()):
        super().__init__()
        self.norm1=nn.LayerNorm([input_dim],weight_attr=False,bias_attr=False)
        self.act1=act
        self.layer1=nn.Linear(input_dim,output_dim,weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01)))
        self.norm2=nn.LayerNorm([output_dim],weight_attr=False,bias_attr=False)
        self.act2=act
        self.layer2=nn.Linear(output_dim,output_dim,weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01)),)
        self.seq_layer=nn.Sequential(
            self.norm1,
            self.act1,
            self.layer1,
            self.norm2,
            self.act2,
            self.layer2
        )
        if input_dim==output_dim:
            self.shortcut_layer=Equallayer()
        else:
            self.shortcut_layer=nn.Linear(input_dim,output_dim)
    def forward(self,x):
        return self.shortcut_layer(x)+self.seq_layer(x)
class ResNormNet(nn.Layer):
    def __init__(self,fc_num_list,act,act_output=None):
        super().__init__()
        print("act=",act)
        self.fc_num_list=fc_num_list
        #get dense_layer_list
        layer_list=[]
        for i in range(1, len(fc_num_list)):
            input_dim,output_dim=fc_num_list[i-1],fc_num_list[i]
            layer_list.append(ResNormBlock(input_dim,output_dim,act=act))
        if act_output is not None:
            layer_list.append(act_output)
        self.net=nn.Sequential(*layer_list)

    def forward(self,x):
        return self.net(x)
    


class SeqVaePred(nn.Layer,ModelSaver):
    def __init__(self,info:svaepred_info):
        super().__init__()
        self.info=info
        "global info vae"
        self.gloinfo_input_fc=self.resnorm_net(self.info.gloinfo_ifc_input_dim,self.info.gloinfo_ifc_mid_dim,self.info.gloinfo_ifc_mid_layers,self.info.gloinfo_ifc_output_dim,nl_func)
        # self.gloinfo_gru=nn.GRU(self.info.gloinfo_ifc_output_dim,self.info.gloinfo_gru_dim)
        # self.gloinfo_gru=GroupGru(self.info.gloinfo_ifc_output_dim,self.info.gloinfo_gru_dim,1)
        self.gloinfo_gru=GroupGru(self.info.gloinfo_ifc_output_dim,self.info.gloinfo_gru_dim,1)
        
        self.gloinfo_gauss=self.resnorm_net(self.info.gloinfo_gfc_input_dim,self.info.gloinfo_gfc_mid_dim,self.info.gloinfo_gfc_mid_layers,self.info.gloinfo_gfc_output_dim,None)
        "location info"
        self.locinfo_input_fc=self.resnorm_net(self.info.locinfo_ifc_input_dim,self.info.locinfo_ifc_mid_dim,self.info.locinfo_ifc_mid_layers,self.info.locinfo_ifc_output_dim,nl_func)
        # self.locinfo_input_fc=self.gloinfo_input_fc
        # self.locinfo_gru=nn.GRU(self.info.locinfo_ifc_output_dim,self.info.locinfo_gru_dim)
        self.locinfo_gru=GroupGru(self.info.locinfo_ifc_output_dim,self.info.locinfo_gru_dim,1)
        "pred"
        self.pred_fc=self.resnorm_net(
            input_dim=self.info.pred_fc_input_dim,mid_dim=self.info.pred_fc_mid_dim,
            mid_layers=self.info.pred_fc_mid_layers,output_dim=self.info.pred_fc_output_dim,act_output=None)
        "kl_pred"
        self.klpred_input_fc=self.resnorm_net(self.info.klpred_ifc_input_dim,self.info.klpred_ifc_mid_dim,self.info.klpred_ifc_mid_layers,self.info.klpred_ifc_output_dim,nl_func)
        # self.klpred_input_fc=self.gloinfo_input_fc
        self.klpred_gru=GroupGru(self.info.klpred_ifc_output_dim,self.info.klpred_gru_dim,1)
        self.klpred_fc=self.resnorm_net(
            input_dim=self.info.klpred_fc_input_dim,mid_dim=self.info.klpred_fc_mid_dim,
            mid_layers=self.info.klpred_fc_mid_layers,output_dim=self.info.klpred_fc_output_dim,act_output=Softabs())

        self.klpred_moving_avg=self.create_parameter([1],dtype='float32')
        self.klpred_moving_avg.set_value(np.array([1.0],np.float32))
        self.add_parameter("klpred_moving_avg", self.klpred_moving_avg)
        self.klpred_moving_gamma=self.info.klpred_moving_gamma

        self.optimizer=optim.Adam(self.info.train_lr,parameters=[
            *self.gloinfo_input_fc.parameters(),
            *self.gloinfo_gru.parameters(),
            *self.gloinfo_gauss.parameters(),
            *self.locinfo_input_fc.parameters(),
            *self.locinfo_gru.parameters(),
            *self.pred_fc.parameters(),
            *self.klpred_input_fc.parameters(),
            *self.klpred_gru.parameters(),
            *self.klpred_fc.parameters(),
            ]) 
        self.train_step_num=0
        self.reset()
    def resnorm_net(self,input_dim,mid_dim,mid_layers,output_dim,act_output):
        return ResNormNet([input_dim,*[mid_dim]*mid_layers,output_dim],act=nl_func,act_output=act_output)
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
    def _cal_klpred_h(self,s_history,a_history,init_h):
        sa_his=paddle.concat([s_history,a_history],axis=-1)
        sa_feature=self.btd_layer(self.klpred_input_fc,sa_his)
        klpred_h,next_klpred_gru_h=self.klpred_gru(sa_feature,init_h)
        return klpred_h,next_klpred_gru_h
    def _cal_klpred(self,klpred_h,a_history):
        return self.btd_layer(self.klpred_fc,paddle.concat([klpred_h,a_history],axis=-1))
    def _cal_global_gauss(self,global_h):
        global_gauss_param=self.btd_layer(self.gloinfo_gauss,global_h)
        # global_gauss_mean,gloal_gauss_logvar=global_gauss_param[:,:,:self.info.gloinfo_gauss_dim],global_gauss_param[:,:,self.info.gloinfo_gauss_dim:]
        return global_gauss_param
    def _cal_pred(self,gauss_sample,local_h,a_history):
        return self.btd_layer(self.pred_fc,paddle.concat([local_h,gauss_sample,a_history],axis=-1))
    def _sample_from_gauss_param(self,gauss_param,sample_num=1):
        gauss_mean,gauss_logvar=gauss_param[:,:,:self.info.gloinfo_gauss_dim],gauss_param[:,:,self.info.gloinfo_gauss_dim:]
        gauss_sample=ReparamNormal(gauss_mean.shape).sample(gauss_mean,gauss_logvar,sample_num)
        return gauss_sample
    def reset(self):
        self.global_h=paddle.zeros([1,1,self.info.gloinfo_gru_dim])
        self.local_h=paddle.zeros([1,1,self.info.locinfo_gru_dim])
        self.klpred_h=paddle.zeros([1,1,self.info.klpred_gru_dim])
        ec_vec=np.zeros([self.info.gloinfo_gauss_dim*2+self.info.locinfo_gru_dim])
        self.pre_gauss_param=None
        return ec_vec
    def pred_env(self,s_history,a_history):
        with paddle.no_grad():
            if not isinstance(s_history,paddle.Tensor):
                s_history=paddle.to_tensor(s_history,'float32')
            if not isinstance(a_history,paddle.Tensor):  
                a_history=paddle.to_tensor(a_history,'float32')

            pre_glo_h=self.global_h
            pre_loc_h=self.local_h
            pre_klpred_h=self.klpred_h

            global_h,next_global_gru_h=self._cal_global_h(s_history,a_history,pre_glo_h)
            global_gauss_param=self._cal_global_gauss(global_h)
            # global_gauss_sample=self._sample_from_gauss_param(global_gauss_param)
            local_h,next_local_gru_h=self._cal_local_h(s_history,a_history,pre_loc_h)
            klpred_h,next_klpred_gru_h=self._cal_klpred_h(s_history,a_history,pre_klpred_h)
            klpred=self._cal_klpred(klpred_h)

            self.global_h=next_global_gru_h.detach()
            self.local_h=next_local_gru_h.detach()
            self.klpred_h=next_klpred_gru_h.detach()

            "inner reward"
            current_gauss_mean=global_gauss_param[:,:,:self.info.gloinfo_gauss_dim]
            current_gauss_sigma=global_gauss_param[:,:,self.info.gloinfo_gauss_dim:]
            if self.pre_gauss_param is None:
                inner_reward=0
            else:
                inner_reward=klpred.reshape([-1]).numpy()[0]
                if np.isnan(inner_reward):
                    inner_reward=0
            inner_reward=np.clip(inner_reward,0,200)

            self.pre_gauss_param={'mean':current_gauss_mean,'sigma':current_gauss_sigma}
            return paddle.concat([global_gauss_param,local_h],axis=-1),pre_glo_h,pre_loc_h,pre_klpred_h,inner_reward
    def pred_h(self,s_history,a_history,pre_global_h,pre_local_h,pre_klpred_h):
        if isinstance(s_history,paddle.Tensor):
            pass
        else:
            s_history=paddle.to_tensor(s_history,'float32')
            a_history=paddle.to_tensor(a_history,'float32')

        _b,_t,_=s_history.shape

        if pre_global_h is None:
            pre_global_h=paddle.zeros([1,_b,self.info.gloinfo_gru_dim])
            pre_local_h=paddle.zeros([1,_b,self.info.locinfo_gru_dim])
            pre_klpred_h=paddle.zeros([1,_b,self.info.klpred_gru_dim])
        elif isinstance(pre_global_h,paddle.Tensor):
            pre_global_h=paddle.transpose(pre_global_h,[1,0,2])
            pre_local_h=paddle.transpose(pre_local_h,[1,0,2])
            pre_klpred_h=paddle.transpose(pre_klpred_h,[1,0,2])
        else:
            pre_global_h=paddle.to_tensor(pre_global_h,'float32')
            pre_local_h=paddle.to_tensor(pre_local_h,'float32')
            pre_klpred_h=paddle.to_tensor(pre_klpred_h,'float32')
            pre_global_h=paddle.transpose(pre_global_h,[1,0,2])
            pre_local_h=paddle.transpose(pre_local_h,[1,0,2])
            pre_klpred_h=paddle.transpose(pre_klpred_h,[1,0,2])
        with paddle.no_grad():
            global_h,_=self._cal_global_h(s_history,a_history,pre_global_h)
            local_h,_=self._cal_local_h(s_history,a_history,pre_local_h)
            klpred_h,_=self._cal_klpred_h(s_history,a_history,pre_klpred_h)
            return global_h.numpy(),local_h.numpy(),klpred_h.numpy()
        
    def train_with_warmup(self,s_history,a_history,pre_global_h,pre_local_h,pre_klpred_h):
        # print(s_history.shape,a_history.shape,pre_global_h.shape,pre_local_h.shape)
        optimizer_warmup_steps=200
        if self.train_step_num>optimizer_warmup_steps:
            pass
        elif self.train_step_num==optimizer_warmup_steps:
            self.optimizer.set_lr(self.info.train_lr)
        elif self.train_step_num<optimizer_warmup_steps:
            self.optimizer.set_lr(0.0)
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
            pre_klpred_h=paddle.zeros([1,_b,self.info.klpred_gru_dim])
        elif isinstance(pre_global_h,paddle.Tensor):
            pre_global_h=paddle.transpose(pre_global_h,[1,0,2])
            pre_local_h=paddle.transpose(pre_local_h,[1,0,2])
            pre_klpred_h=paddle.transpose(pre_klpred_h,[1,0,2])
        else:
            pre_global_h=paddle.to_tensor(pre_global_h,'float32')
            pre_local_h=paddle.to_tensor(pre_local_h,'float32')
            pre_klpred_h=paddle.to_tensor(pre_klpred_h,'float32')
            pre_global_h=paddle.transpose(pre_global_h,[1,0,2])
            pre_local_h=paddle.transpose(pre_local_h,[1,0,2])
            pre_klpred_h=paddle.transpose(pre_klpred_h,[1,0,2])
        # print(s_history.shape,a_history.shape,pre_global_h.shape,pre_local_h.shape)
        global_h,_=self._cal_global_h(s_history,a_history,pre_global_h)
        global_gauss_param=self._cal_global_gauss(global_h)
        global_gauss_sample=self._sample_from_gauss_param(global_gauss_param)
        local_h,_=self._cal_local_h(s_history,a_history,pre_local_h)

        "pred"
        global_gauss_sample_gap=paddle.reshape(global_gauss_sample[:,1:],[_b,(_t-1)//self.info.recon_seq_len,self.info.recon_seq_len,-1])[:,:,-1:,:]
        global_gauss_sample_recon=paddle.tile(global_gauss_sample_gap,[1,1,self.info.recon_seq_len,1]).reshape([_b,_t-1,-1])
        local_h_recon=local_h[:,:-1,:]
        a_history_recon=a_history[:,1:,:]
        pred=self._cal_pred(global_gauss_sample_recon,local_h_recon,a_history_recon)
        # pred=self._cal_pred(global_gauss_sample[:,1:,:],local_h[:,:-1,:],a_history[:,1:,:])
        pred_target=s_history[:,1:,:]
        pred_loss=paddle.mean((pred-pred_target).pow(2))

        "kl"
        # global_gauss_param_gap=paddle.concat([global_gauss_param[:,0:1],paddle.reshape(global_gauss_param[:,1:],[_b,(_t-1)//self.info.recon_seq_len,self.info.recon_seq_len,-1])[:,:,-1,:]],axis=1)
        global_gauss_param_start=global_gauss_param[:,:-self.info.recon_seq_len]
        global_gauss_param_final=global_gauss_param[:,self.info.recon_seq_len:]
        mean_start=global_gauss_param_start[:,:,:self.info.gloinfo_gauss_dim]
        sigma_start=paddle.exp(global_gauss_param_start[:,:,self.info.gloinfo_gauss_dim:])
        mean_final=global_gauss_param_final[:,:,:self.info.gloinfo_gauss_dim]
        sigma_final=paddle.exp(global_gauss_param_final[:,:,self.info.gloinfo_gauss_dim:])

        kl_loss=paddle.mean(paddle.sum(gauss_KL(mean_final,sigma_final,mean_start,sigma_start),axis=-1,keepdim=True))
        
        "klpred"
        klpred_h,_=self._cal_klpred_h(s_history,a_history,pre_klpred_h)
        kl_pred=self._cal_klpred(klpred_h[:,:-1],a_history[:,1:])

        klpred_gauss_param_start=global_gauss_param[:,:-1]
        klpred_gauss_param_final=global_gauss_param[:,1:]

        klpred_mean_start=klpred_gauss_param_start[:,:,:self.info.gloinfo_gauss_dim]
        klpred_sigma_start=paddle.exp(klpred_gauss_param_start[:,:,self.info.gloinfo_gauss_dim:])
        klpred_mean_final=klpred_gauss_param_final[:,:,:self.info.gloinfo_gauss_dim]
        klpred_sigma_final=paddle.exp(klpred_gauss_param_final[:,:,self.info.gloinfo_gauss_dim:])
        
        klpred_target_original=paddle.sum(gauss_KL(klpred_mean_final,klpred_sigma_final,klpred_mean_start,klpred_sigma_start),axis=-1,keepdim=True).detach()
        self.klpred_moving_avg.set_value(self.klpred_moving_avg*self.klpred_moving_gamma+(1-self.klpred_moving_gamma)*paddle.mean(klpred_target_original))
        # print(self.klpred_moving_avg.numpy()[0])
        klpred_target=klpred_target_original/(self.klpred_moving_avg.detach())

        klpred_loss=paddle.mean((klpred_target-kl_pred).pow(2))

        total_loss=pred_loss+self.info.train_kl_loss_ratio*kl_loss+klpred_loss

        #打印结果
        # pred_numpy=pred.numpy()[0]
        # pred_target_numpy=pred_target.numpy()[0]
        # kl_loss_numpy=paddle.mean(gauss_KL(
        #     global_gauss_param[:,1:,:self.info.gloinfo_gauss_dim],
        #     global_gauss_param[:,1:,self.info.gloinfo_gauss_dim:],
        #     global_gauss_param[:,:-1,:self.info.gloinfo_gauss_dim],
        #     global_gauss_param[:,:-1:,self.info.gloinfo_gauss_dim:],
        #     ),axis=-1).numpy()[0]
        # a_numpy=a_history.numpy()[0][1:]
        # klpred_numpy=kl_pred.numpy()[0]
        # klpred_target_numpy=klpred_target.numpy()[0]
        # pred_loss_numpy=np.mean((pred_numpy-pred_target_numpy)**2,axis=-1)
        # print(klpred_numpy.reshape([-1]),klpred_target_numpy.reshape([-1]),np.mean(klpred_target_numpy))
        # for i in range(len(pred_numpy)):
        #     print("a=",np.argmax(a_numpy[i]),"s=",pred_target_numpy[i][np.argmax(a_numpy[i])],"pred=",pred_numpy[i][np.argmax(a_numpy[i])],"kl=",kl_loss_numpy[i],"klpred=",klpred_numpy[i])

        self.optimizer.clear_grad()
        total_loss.backward()
        self.optimizer.step()

        info_dict={
            'total_loss':total_loss.numpy()[0],
            'pred_loss':pred_loss.numpy()[0],
            'kl_loss':kl_loss.numpy()[0],
            'klpred_loss':klpred_loss.numpy()[0],
            'klpred_target_original':paddle.mean(klpred_target_original).numpy()[0],
            'klpred_moving_avg':self.klpred_moving_avg.numpy()[0],
            'klpred_mean':paddle.mean(kl_pred).numpy()[0],
            'klpred_target_mean':paddle.mean(klpred_target).numpy()[0],
        }

        return info_dict
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