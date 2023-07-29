import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
import numpy as np
import os
from mlutils.ml import dense_block,ReparamNormal,gauss_KL,DataFormat,ReplayMemory
from mlutils.ml import Basic_Module, ReplayMemory,dense_block,DataFormat,Reslayer
from mlutils.ml import ModelSaver

class svae_info():
    nl_func=nn.LeakyReLU(0.1)

    gauss_dim=256
    state_vec_dim=256
    action_vec_dim=4

    encoder_ifc_input_dim=state_vec_dim+action_vec_dim
    encoder_ifc_mid_layer_num=3
    encoder_ifc_mid_dim=256
    encoder_ifc_output_dim=256

    encoder_gru_input_dim=encoder_ifc_output_dim
    encoder_gru_hid_dim=256

    encoder_gfc_input_dim=encoder_gru_hid_dim
    encoder_gfc_mid_layer_num=3
    encoder_gfc_mid_dim=256
    encoder_gfc_output_dim=gauss_dim

    decoder_ifc_input_dim=state_vec_dim+action_vec_dim
    decoder_ifc_mid_layer_num=3
    decoder_ifc_mid_dim=256
    decoder_ifc_output_dim=256

    decoder_gru_input_dim=decoder_ifc_output_dim
    decoder_gru_hid_dim=gauss_dim

    decoder_ofc_input_dim=decoder_gru_hid_dim
    decoder_ofc_mid_layer_num=3
    decoder_ofc_mid_dim=256
    decoder_ofc_output_dim=state_vec_dim




class SeqVAE(nn.Layer,ModelSaver):
    def __init__(self):
        super().__init__()
        info=svae_info
        "encoder"
        self.encoder_input_fc=dense_block([info.encoder_ifc_input_dim,*[info.encoder_ifc_mid_dim]*info.encoder_ifc_mid_layer_num,info.encoder_ifc_output_dim],act_output=info.nl_func)
        self.encoder_gru=nn.GRU(info.encoder_gru_input_dim,info.encoder_gru_hid_dim)
        self.encoder_gauss=dense_block([info.encoder_gfc_input_dim,*[info.encoder_gfc_mid_dim]*info.encoder_gfc_mid_layer_num,info.encoder_gfc_output_dim*2])
        "decoder"
        self.decoder_input_fc=dense_block([info.decoder_ifc_input_dim,*[info.decoder_ifc_mid_dim]*info.decoder_ifc_mid_layer_num,info.decoder_ifc_output_dim],act_output=info.nl_func)
        self.decoder_gru=nn.GRU(info.decoder_gru_input_dim,info.decoder_gru_hid_dim)
        self.decoder_output_fc=dense_block([info.decoder_ofc_input_dim,*[info.decoder_ofc_mid_dim]*info.decoder_ofc_mid_layer_num,info.decoder_ofc_output_dim],act_output=nn.Sigmoid())
        
        self.optimizer=optim.Adam(0.0001,parameters=[
            *self.encoder_input_fc.parameters(),
            *self.encoder_gru.parameters(),
            *self.encoder_gauss.parameters(),
            *self.decoder_input_fc.parameters(),
            *self.decoder_gru.parameters(),
            *self.decoder_output_fc.parameters(),
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
    def sample_with_gauss_param(self,s_history,a_history):
        if isinstance(s_history,paddle.Tensor):
            s_his=s_history
            a_his=a_history
        else:
            s_his=paddle.to_tensor(s_history,'float32')
            a_his=paddle.to_tensor(a_history,'float32')
        sa_his=paddle.concat([s_his,a_his],axis=-1)
        _b,_t,_=s_his.shape

        "feature"
        sa_feature_encoder=self.btd_layer(self.encoder_input_fc,sa_his)
        "gru"
        encoder_output_his,_=self.encoder_gru(sa_feature_encoder,paddle.zeros([1,sa_his.shape[0],self.gru_dim]))
        "gauss_sample"
        "[init，step0,step1 ... stepn-1]"
        encoder_gru_hid=paddle.concat([paddle.zeros([sa_his.shape[0],1,self.gru_dim]),encoder_output_his],axis=-2)
        encoder_gauss_param=self.btd_layer(self.encoder_gauss,encoder_gru_hid)
        gauss_param_mean,gauss_param_log_var=encoder_gauss_param[:,:,:self.gru_dim],encoder_gauss_param[:,:,self.gru_dim:]
        encoder_hid=ReparamNormal(gauss_param_mean.shape).sample(gauss_param_mean,gauss_param_log_var)
        return encoder_hid,gauss_param_mean,gauss_param_log_var

    def sample_recon_with_gauss_param(self,s_history,a_history):
        if isinstance(s_history,paddle.Tensor):
            s_his=s_history
            a_his=a_history
        else:
            s_his=paddle.to_tensor(s_history,'float32')
            a_his=paddle.to_tensor(a_history,'float32')
        sa_his=paddle.concat([s_his,a_his],axis=-1)
        _b,_t,_=s_his.shape

        encoder_hid,gauss_param_mean,gauss_param_log_var=self.sample_with_gauss_param(s_history,a_history)

        "decode"
        "feature"
        "[step0,step1 ... stepn-1]"
        sa_feature_decoder=self.btd_layer(self.decoder_input_fc,sa_his)
        "gru"
        decoder_gru_hid,_=self.decoder_gru(sa_feature_decoder,paddle.reshape(encoder_hid[:,-1:,:],[1,_b,self.gru_dim]))
        "output"
        decoder_output=self.btd_layer(self.decoder_output_fc,decoder_gru_hid)#[b,t-1,d]

        return decoder_output,gauss_param_mean,gauss_param_log_var
    def train(self,train_s_history,train_a_history):
        if isinstance(train_s_history,paddle.Tensor):
            s_his=train_s_history
            a_his=train_a_history
        else:
            s_his=paddle.to_tensor(train_s_history,'float32')
            a_his=paddle.to_tensor(train_a_history,'float32')
        sa_his=paddle.concat([s_his,a_his],axis=-1)

        "sample recon"
        decoder_output,gauss_param_mean,gauss_param_log_var=self.sample_recon_with_gauss_param(s_his,a_his)

        "recon last step"
        label=s_his[:,1:]
        recon_loss=paddle.mean((decoder_output[:,:-1]-label)**2)

        "kl loss"
        "[stepn-1]"
        mean0=gauss_param_mean[:,-1:]
        sigma0=paddle.exp(0.5*gauss_param_log_var[:,-1:])
        "[init，step0,step1 ... stepn-1]"
        mean1=gauss_param_mean
        sigma1=paddle.exp(0.5*gauss_param_log_var)
        kl_loss1=paddle.mean(gauss_KL(mean0,sigma0,mean1,sigma1))

        total_loss=recon_loss+0.00001*kl_loss1
        
        self.optimizer.clear_grad()
        total_loss.backward()
        self.optimizer.step()

        return [total_loss.numpy()[0],recon_loss.numpy()[0],kl_loss1.numpy()[0]]


        

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