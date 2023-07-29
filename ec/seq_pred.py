import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
import numpy as np
import os
from mlutils.ml import dense_block,ReparamNormal,gauss_KL,DataFormat,ReplayMemory
from mlutils.ml import Basic_Module, ReplayMemory,dense_block,DataFormat,Reslayer
from mlutils.ml import ModelSaver
class spred_info():
    nl_func=nn.LeakyReLU(0.1)

    state_vec_dim=256
    action_vec_dim=4
    seq_vae_gauss_vec_dim=256

    ifc_input_dim=state_vec_dim+action_vec_dim+seq_vae_gauss_vec_dim
    ifc_mid_layer_num=3
    ifc_mid_dim=256
    ifc_output_dim=256

    gru_input_dim=ifc_output_dim
    gru_hid_dim=256

    ofc_input_dim=gru_hid_dim
    ofc_mid_layer_num=3
    ofc_mid_dim=256
    ofc_output_dim=state_vec_dim
"输入0-t步的state，和t+1步的seq_vae_h，来预测t+1步的state。这样智能体的隐含状态中必须包含当前的位置和环境的变量，才能完整预测"
class SeqPred(nn.Layer,ModelSaver):
    def __init__(self):
        super().__init__()
        info=spred_info
        self.info=info
        "encoder"
        self.input_fc=dense_block([info.ifc_input_dim,*[info.ifc_mid_dim]*info.ifc_mid_layer_num,info.ifc_output_dim],act_output=info.nl_func)
        self.gru=nn.GRU(info.gru_input_dim,info.gru_hid_dim)
        self.output_fc=dense_block([info.ofc_input_dim,*[info.ofc_mid_dim]*info.ofc_mid_layer_num,info.ofc_output_dim],act_output=info.nl_func)

        self.optimizer=optim.Adam(0.0001,parameters=[
            *self.input_fc.parameters(),
            *self.gru.parameters(),
            *self.output_fc.parameters(),
            ])
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
    def pred_one_step(self,state,action,svae_code_paddle,h_paddle):
        state_paddle=paddle.to_tensor(np.reshape(state,[1,1,-1]),'float32')
        action_paddle=paddle.to_tensor(np.reshape(action,[1,1,-1]),'float32')
        
        input_paddle=paddle.concat([state_paddle,action_paddle,svae_code_paddle],axis=-1)
        input_feature=self.btd_layer(self.input_fc,input_paddle)
        gru_out=self.gru(input_feature,h_paddle)
        return gru_out.detach()
    def pred_history(self,s_history,a_history,seq_vae_history):
        """
        s_history时间范围是0~T-1，共T步

        t时间的s_his,a_his，以及t+1步的seq_vae_history来预测t+1步的s_his.
        """
        if isinstance(s_history,paddle.Tensor):
            s_his=s_history
            a_his=a_history
            s_vae_his=seq_vae_history
        else:
            s_his=paddle.to_tensor(s_history,'float32')
            a_his=paddle.to_tensor(a_history,'float32')
        sa_his=paddle.concat([s_his,a_his],axis=-1)
        _b,_t,_=s_his.shape
        sa_his_for_input=sa_his[:,:-1]
        seq_vae_his_for_input=seq_vae_history[:,1:]

        sa_svae_his_for_input=paddle.concat([sa_his_for_input,seq_vae_his_for_input],axis=-1)

        sa_feature=self.btd_layer(self.input_fc,sa_svae_his_for_input)
        gru_out,_=self.gru(sa_feature,paddle.zeros([1,_b,self.info.gru_hid_dim]))
        pred=self.btd_layer(self.output_fc,gru_out)
        return pred
    def train(self,s_history,a_history,seq_vae_history):
        if isinstance(s_history,paddle.Tensor):
            s_his=s_history
            a_his=a_history
            s_vae_his=seq_vae_history
        else:
            s_his=paddle.to_tensor(s_history,'float32')
            a_his=paddle.to_tensor(a_history,'float32')
        sa_his=paddle.concat([s_his,a_his],axis=-1)

        pred_s_list=self.pred_history(s_his,a_his,s_vae_his)

        recon_loss=paddle.mean((pred_s_list-s_his[:,1:])**2)

        total_loss=recon_loss
        
        self.optimizer.clear_grad()
        total_loss.backward()
        self.optimizer.step()
        return [total_loss.numpy()[0],recon_loss.numpy()[0]]

    
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
        super().send_model_to_np(model=self)