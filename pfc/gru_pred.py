import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
import numpy as np
import os
from mlutils.ml import dense_block,ReparamNormal,gauss_KL,DataFormat,ReplayMemory

class GruPredHid(nn.Layer):
    def __init__(self):
        super().__init__()
        self.train_batchsize=1024
        self.train_start_size=100000
        self.gru_dim=256
        self.action_dim=10
        self.state_dim=2

        self.encoder_input_fc=dense_block([self.action_dim+self.state_dim,32,32,32])
        self.encoder_gru=nn.GRU(32,self.gru_dim)
        self.encoder_gauss=dense_block([self.gru_dim,2*self.gru_dim,2*self.gru_dim,2*self.gru_dim])

        self.decoder_input_fc=dense_block([self.action_dim,32,32,32])
        self.decoder_gru=nn.GRU(32,self.gru_dim)
        self.decoder_output_fc=dense_block([self.gru_dim,32,32,32,self.state_dim])
       
        self.dis_pred_fc=dense_block([self.gru_dim,32,32,32,self.action_dim])


        "pred kl"
        self.kl_pred_input_fc=dense_block([self.action_dim+self.state_dim,32,32,32])
        self.kl_pred_gru=nn.GRU(32,self.gru_dim)
        self.kl_pred_output_fc=dense_block([self.gru_dim+self.action_dim,32,1],act_output=nn.Softplus())

        self.lr_decay=paddle.optimizer.lr.MultiStepDecay(learning_rate=0.001, milestones=[2000,5000], gamma=1/10**0.5)

        self.optimizer=optim.Adam(self.lr_decay,parameters=[
            *self.encoder_input_fc.parameters(),
            # *self.en1coder.parameters(),
            *self.encoder_gru.parameters(),
            *self.encoder_gauss.parameters(),
            *self.decoder_input_fc.parameters(),
            *self.decoder_gru.parameters(),
            *self.decoder_output_fc.parameters(),
            # *self.future_pred_fc.parameters(),
            *self.dis_pred_fc.parameters(),
            # *self.decoder.parameters()
            *self.kl_pred_input_fc.parameters(),
            *self.kl_pred_gru.parameters(),
            *self.kl_pred_output_fc.parameters(),
            ])
        
        self.rpm=ReplayMemory(
            [
                DataFormat("action_history",[20,10],np.float32),
                DataFormat("obs_history",[20,2],np.float32)
                ],5000000)

    def rpm_collect(self,data_dict):
        self.rpm.collect(data_dict)
    def btd_layer(self,net,x):
        batch,time,dim=x.shape
        return net(x.reshape([batch*time,dim])).reshape([batch,time,-1])
    def bttd_layer(self,net,x):
        batch,time,time,dim=x.shape
        return net(x.reshape([batch*time*time,dim])).reshape([batch,time,time,-1])
    def learn(self):
        if len(self.rpm)>=self.train_start_size:
            train_dict=self.rpm.sample_batch(self.train_batchsize)
            return self.train_history(train_dict["obs_history"],train_dict["action_history"])
        else:
            pass
    def save_model(self,save_dir,iter_num):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        paddle.save(self.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")

    def load_model(self,save_dir,iter_num):
        self.set_state_dict(paddle.load(f"{save_dir}/{iter_num}_model.pdparams"))
    def train_history(self,s_history,a_history):
        s_his=paddle.to_tensor(s_history)
        a_his=paddle.to_tensor(a_history)
        sa_his=paddle.concat([s_his,a_his],axis=-1)


        "encode"
        sa_his_feature=self.btd_layer(self.encoder_input_fc,sa_his)
        encoder_output_his,_=self.encoder_gru(sa_his_feature,paddle.zeros([1,sa_his_feature.shape[0],self.gru_dim]))

        "[init，step0,step1 ... stepn-1]"
        encoder_gru_hid=paddle.concat([paddle.zeros([sa_his_feature.shape[0],1,self.gru_dim]),encoder_output_his],axis=-2)
        encoder_gauss_param=self.btd_layer(self.encoder_gauss,encoder_gru_hid)
        gauss_param_mean,gauss_param_log_var=encoder_gauss_param[:,:,:self.gru_dim],encoder_gauss_param[:,:,self.gru_dim:]
        encoder_hid=ReparamNormal(gauss_param_mean.shape).sample(gauss_param_mean,gauss_param_log_var)

        "recon last step"
        a_his_feature=self.btd_layer(self.decoder_input_fc,a_his)
        _b,_t,_d=a_his_feature.shape
        "[step0,step1 ... stepn-1]"
        decoder_pred_gru,_=self.decoder_gru(a_his_feature,paddle.reshape(encoder_hid[:,-1:,:],[1,_b,self.gru_dim]))
        pred=self.btd_layer(self.decoder_output_fc,decoder_pred_gru)

        label=s_his
        recon_loss=paddle.mean(F.softmax_with_cross_entropy(pred,label,soft_label=True))*2

        "kl loss"
        "[stepn-1]"
        mean0=gauss_param_mean[:,-1:]
        sigma0=paddle.exp(0.5*gauss_param_log_var[:,-1:])
        "[init，step0,step1 ... stepn-1]"
        mean1=gauss_param_mean
        sigma1=paddle.exp(0.5*gauss_param_log_var)
        kl_loss1=paddle.mean(gauss_KL(mean0,sigma0,mean1,sigma1))

        mean2=gauss_param_mean[:,1:]
        sigma2=paddle.exp(0.5*gauss_param_log_var[:,1:])
        mean3=gauss_param_mean[:,:-1]
        sigma3=paddle.exp(0.5*gauss_param_log_var[:,:-1])
        kl_loss2=paddle.mean(gauss_KL(mean2,sigma2,mean3,sigma3))

        "kl pred"
        kl_pred_sa_his_feature=self.btd_layer(self.kl_pred_input_fc,sa_his)
        kl_pred_gru_hid,_=self.kl_pred_gru(kl_pred_sa_his_feature,paddle.zeros([1,kl_pred_sa_his_feature.shape[0],self.gru_dim]))
        "[init，step0,step1 ... stepn-1]"
        kl_pred_gru_hid=paddle.concat([paddle.zeros([kl_pred_sa_his_feature.shape[0],1,self.gru_dim]),kl_pred_gru_hid],axis=-2)
        "[step0,step1 ... stepn-1]"
        kl_pred_output=self.btd_layer(self.kl_pred_output_fc,paddle.concat([kl_pred_gru_hid[:,:-1],a_his],axis=-1))

        "kl_pred_loss"
        "[stepn-1]"
        mean2=gauss_param_mean[:,1:]
        sigma2=paddle.exp(0.5*gauss_param_log_var[:,1:])
        mean3=gauss_param_mean[:,:-1]
        sigma3=paddle.exp(0.5*gauss_param_log_var[:,:-1])

        kl_target=paddle.sum(gauss_KL(mean2,sigma2,mean3,sigma3),axis=-1,keepdim=True).detach()
        # print(kl_target.shape)
        kl_pred_loss=paddle.mean((kl_target-kl_pred_output)**2)

        total_loss=recon_loss+kl_loss1+kl_loss2+kl_pred_loss

        self.optimizer.clear_grad()
        total_loss.backward()
        self.optimizer.step()
        self.lr_decay.step()
        # print(total_loss)
        # self.pred_result=np.transpose(F.softmax(pred).numpy(),[1,0])
        # self.label=np.transpose(np.array(s_history),[1,0])
        # print(F.softmax(decoder_pred).numpy(),np.array(s_history))
        return [recon_loss.numpy()[0],kl_loss1.numpy()[0],kl_loss2.numpy()[0],kl_pred_loss.numpy()[0]]

    def test(self,s_history,a_history):
        s_his=paddle.to_tensor(s_history)
        a_his=paddle.to_tensor(a_history)
        sa_his=paddle.concat([s_his,a_his],axis=-1)

        # print("s_his,a_his",s_his.shape,a_his.shape)
        "encode"
        sa_his_feature=self.btd_layer(self.encoder_input_fc,sa_his)
        # print(sa_his)
        encoder_output_his,_=self.encoder_gru(sa_his_feature,paddle.zeros([1,sa_his_feature.shape[0],self.gru_dim]))

        "[init，step0,step1 ... stepn-1]"
        encoder_gru_hid=paddle.concat([paddle.zeros([sa_his_feature.shape[0],1,self.gru_dim]),encoder_output_his],axis=-2)
        encoder_gauss_param=self.btd_layer(self.encoder_gauss,encoder_gru_hid)
        gauss_param_mean,gauss_param_log_var=encoder_gauss_param[:,:,:self.gru_dim],encoder_gauss_param[:,:,self.gru_dim:]
        # encoder_hid=ReparamNormal(gauss_param_mean.shape).sample(gauss_param_mean,gauss_param_log_var)
        "kl pred"
        kl_pred_sa_his_feature=self.btd_layer(self.kl_pred_input_fc,sa_his)
        kl_pred_gru_hid,_=self.kl_pred_gru(kl_pred_sa_his_feature,paddle.zeros([1,kl_pred_sa_his_feature.shape[0],self.gru_dim]))
        # kl_pred_output_input=
        "[init，step0,step1 ... stepn-1]"
        kl_pred_gru_hid=paddle.concat([paddle.zeros([kl_pred_sa_his_feature.shape[0],1,self.gru_dim]),kl_pred_gru_hid],axis=-2)
        "[step0,step1 ... stepn-1]"
        kl_pred_output=self.btd_layer(self.kl_pred_output_fc,paddle.concat([kl_pred_gru_hid[:,:-1],a_his],axis=-1))

        "kl_pred_loss"
        "[stepn-1]"
        mean2=gauss_param_mean[:,1:]
        sigma2=paddle.exp(0.5*gauss_param_log_var[:,1:])
        mean3=gauss_param_mean[:,:-1]
        sigma3=paddle.exp(0.5*gauss_param_log_var[:,:-1])

        kl_target=paddle.sum(gauss_KL(mean2,sigma2,mean3,sigma3),axis=-1,keepdim=True).detach()
        print(sa_his)
        print(kl_pred_output,kl_target)


        "recon"
        # a_his_feature=self.btd_layer(self.decoder_input_fc,a_his)
        # decoder_pred_gru,_=self.decoder_gru(a_his_feature,paddle.transpose(encoder_hid[:,-1:],[1,0,2]))
        
        # pred=self.btd_layer(self.decoder_output_fc,decoder_pred_gru)

        # loss_ratio=s_his[:,:,1:]*(5-5.0/9.0)+5.0/9.0
        # recon_loss=paddle.mean(F.softmax_with_cross_entropy(pred,s_his,soft_label=True)*loss_ratio)

        "kl loss"
        mean0=gauss_param_mean[:,1:]
        sigma0=paddle.exp(0.5*gauss_param_log_var[:,1:])
        mean1=gauss_param_mean[:,:-1]
        sigma1=paddle.exp(0.5*gauss_param_log_var[:,:-1])
        kl_loss_list=paddle.sum(gauss_KL(mean0,sigma0,mean1,sigma1),axis=-1)

        return kl_loss_list,kl_pred_output.numpy()
