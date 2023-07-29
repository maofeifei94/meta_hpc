import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
import numpy as np
import os
from mlutils.ml import dense_block,ReparamNormal,gauss_KL,DataFormat,ReplayMemory
from mlutils.ml import Basic_Module, ReplayMemory,dense_block,DataFormat,Reslayer
from train_pred_net import train_pred

v_encoder_hid_c=4
v_z_c=4
v_decoder_hic_c=4
def nl_func():
    return nn.LeakyReLU(0.1)

class Conv_Encoder_net(nn.Layer):
    def __init__(self,img_shape):
        super(Conv_Encoder_net,self).__init__()
        hid_c=v_encoder_hid_c
        self.conv_net=nn.Sequential(
            nn.Conv2D(
                in_channels=img_shape[0],out_channels=hid_c,
                kernel_size=[3,3],stride=[2,2],padding='SAME'),nl_func(),
            nn.Conv2D(
                in_channels=hid_c,out_channels=hid_c,
                kernel_size=[3,3],stride=[2,2],padding='SAME'),nl_func(),
            nn.Conv2D(
                in_channels=hid_c,out_channels=hid_c,
                kernel_size=[3,3],stride=[2,2],padding='SAME'),nl_func(),
            # Reslayer(
            #     nn.Conv2D(
            #         in_channels=hid_c,out_channels=hid_c,
            #         kernel_size=[3,3],stride=[1,1],padding='SAME'),nl_func()
            # ),
            # Reslayer(
            #     nn.Conv2D(
            #         in_channels=hid_c,out_channels=hid_c,
            #         kernel_size=[3,3],stride=[1,1],padding='SAME'),nl_func()
            # ),
            nn.Conv2D(
                in_channels=hid_c,out_channels=v_z_c,
                kernel_size=[3,3],stride=[1,1],padding='SAME'),
            nl_func()
        )
    def forward(self,img):
        x=img
        x=self.conv_net(x)
        return x

class Conv_Decoder_net(nn.Layer):
    def __init__(self):
        super(Conv_Decoder_net,self).__init__()
        hid_c=v_decoder_hic_c
        self.deconv_net=nn.Sequential(
            nn.Conv2D(
                in_channels=v_z_c,out_channels=hid_c,
                kernel_size=[3,3],stride=[1,1],padding='SAME'),nl_func(),
            # Reslayer(
            #     nn.Conv2D(
            #         in_channels=hid_c,out_channels=hid_c,
            #         kernel_size=[3,3],stride=[1,1],padding='SAME'),nl_func()
            # ),
            nn.Conv2DTranspose(
                in_channels=hid_c,out_channels=hid_c,
                kernel_size=[3,3],stride=2,padding="SAME"),nl_func(),
            nn.Conv2DTranspose(
                in_channels=hid_c,out_channels=hid_c,
                kernel_size=[3,3],stride=2,padding="SAME"),nl_func(),
            nn.Conv2DTranspose(
                in_channels=hid_c,out_channels=hid_c,
                kernel_size=[3,3],stride=2,padding="SAME"),nl_func(),
            # nn.Conv2DTranspose(
            #     in_channels=hid_c,out_channels=hid_c,
            #     kernel_size=[3,3],stride=1,padding="SAME"),nl_func(),
            nn.Conv2DTranspose(
                in_channels=hid_c,out_channels=3,
                kernel_size=[3,3],stride=1,padding="SAME"),nn.Sigmoid(),
            )
    def forward(self,img):
        return self.deconv_net(img)
class GruPredHid(nn.Layer):
    
    def __init__(self):
        super().__init__()
        self.train_batchsize=256
        self.train_start_size=20000
        self.gru_dim=256
        self.conv_feature_dim=256
        self.conv_feature_shape=[4,8,8]
        self.action_dim=3
        self.state_dim=2
        self.history_len=100

        "encoder"
        # self.encoder_input_conv=Conv_Encoder_net([3,64,64])
        self.encoder_input_fc=dense_block([self.conv_feature_dim+self.action_dim,self.gru_dim,self.gru_dim],act_output=nn.LeakyReLU(0.1))
        self.encoder_gru=nn.GRU(self.gru_dim,self.gru_dim)
        self.encoder_gauss=dense_block([self.gru_dim,2*self.gru_dim,2*self.gru_dim,2*self.gru_dim])
        "decoder"
        # self.decoder_input_fc=dense_block([self.action_dim,32,32,32])
        self.decoder_input_fc=dense_block([self.action_dim,self.gru_dim,self.gru_dim],act_output=nn.LeakyReLU(0.1))
        self.decoder_gru=nn.GRU(self.gru_dim,self.gru_dim)
        self.decoder_output_fc=dense_block([self.gru_dim,self.gru_dim,self.gru_dim,self.gru_dim,self.gru_dim,self.gru_dim,self.conv_feature_dim],act=nn.Tanh())
        # self.decoder_output_conv=Conv_Decoder_net()
        
        "pred kl"
        self.kl_pred_input_fc=dense_block([self.action_dim+self.state_dim,32,32,32])
        self.kl_pred_gru=nn.GRU(32,self.gru_dim)
        self.kl_pred_output_fc=dense_block([self.gru_dim+self.action_dim,32,1],act_output=nn.Softplus())

        self.lr_decay=paddle.optimizer.lr.MultiStepDecay(learning_rate=0.001, milestones=[3000], gamma=0.3)

        self.optimizer=optim.Adam(0.0003,parameters=[
            # *self.encoder_input_conv.parameters(),
            *self.encoder_input_fc.parameters(),
            *self.encoder_gru.parameters(),
            *self.encoder_gauss.parameters(),

            *self.decoder_input_fc.parameters(),
            *self.decoder_gru.parameters(),
            *self.decoder_output_fc.parameters(),
            # *self.decoder_output_conv.parameters(),

            *self.kl_pred_input_fc.parameters(),
            *self.kl_pred_gru.parameters(),
            *self.kl_pred_output_fc.parameters(),
            ])
        
        self.rpm=ReplayMemory(
            [
                DataFormat("action_history",[self.history_len,self.action_dim],np.float32),
                DataFormat("obs_feature_history",[self.history_len,self.conv_feature_dim],np.float32),
                # DataFormat("obs_last",[])
                ],100000)

    def rpm_collect(self,data_dict):
        self.rpm.collect(data_dict)
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

    def learn(self):
        if len(self.rpm)>=self.train_start_size:
            train_dict=self.rpm.sample_batch(self.train_batchsize)
            return self.train_history(train_dict["obs_feature_history"],train_dict["action_history"])
        else:
            # print("rmp size=",len(self.rpm),self.rpm._max_size)
            pass
    def save_model(self,save_dir,iter_num):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        paddle.save(self.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")

    def load_model(self,save_dir,iter_num):
        self.set_state_dict(paddle.load(f"{save_dir}/{iter_num}_model.pdparams"))
    def train_history(self,s_history,a_history):
        s_his=paddle.to_tensor(s_history)
        # s_his=paddle.cast(s_his,'float32')/255
        a_his=paddle.to_tensor(a_history)


        _b,_t,_=s_his.shape

        "feature"
        # s_feature=self.btchw_layer(self.encoder_input_conv,s_his).reshape([_b,_t,-1])
        sa_his=paddle.concat([s_his,a_his],axis=-1)
        sa_feature_encoder=self.btd_layer(self.encoder_input_fc,sa_his)

        "gru"
        encoder_output_his,_=self.encoder_gru(sa_feature_encoder,paddle.zeros([1,sa_his.shape[0],self.gru_dim]))

        "gauss_sample"
        "[init，step0,step1 ... stepn-1]"
        encoder_gru_hid=paddle.concat([paddle.zeros([sa_his.shape[0],1,self.gru_dim]),encoder_output_his],axis=-2)
        encoder_gauss_param=self.btd_layer(self.encoder_gauss,encoder_gru_hid)
        gauss_param_mean,gauss_param_log_var=encoder_gauss_param[:,:,:self.gru_dim],encoder_gauss_param[:,:,self.gru_dim:]
        encoder_hid=ReparamNormal(gauss_param_mean.shape).sample(gauss_param_mean,gauss_param_log_var)

        "decode"
        "[step0,step1 ... stepn-1]"
        sa_feature_decoder=self.btd_layer(self.decoder_input_fc,a_his)
        decoder_gru_hid,_=self.decoder_gru(sa_feature_decoder,paddle.reshape(encoder_hid[:,-1:,:],[1,_b,self.gru_dim]))
        decoder_output=self.btd_layer(self.decoder_output_fc,decoder_gru_hid)
        # print(decoder_gru_hid.shape)
        # decoder_pred_img=self.btchw_layer(self.decoder_output_conv,paddle.reshape(decoder_gru_hid,[_b,_t,*self.conv_feature_shape]))

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

        mean2=gauss_param_mean[:,1:]
        sigma2=paddle.exp(0.5*gauss_param_log_var[:,1:])
        mean3=gauss_param_mean[:,:-1]
        sigma3=paddle.exp(0.5*gauss_param_log_var[:,:-1])
        kl_loss2=paddle.mean(gauss_KL(mean2,sigma2,mean3,sigma3))

        # "kl pred"
        # kl_pred_sa_his_feature=self.btd_layer(self.kl_pred_input_fc,sa_his)
        # kl_pred_gru_hid,_=self.kl_pred_gru(kl_pred_sa_his_feature,paddle.zeros([1,kl_pred_sa_his_feature.shape[0],self.gru_dim]))
        # "[init，step0,step1 ... stepn-1]"
        # kl_pred_gru_hid=paddle.concat([paddle.zeros([kl_pred_sa_his_feature.shape[0],1,self.gru_dim]),kl_pred_gru_hid],axis=-2)
        # "[step0,step1 ... stepn-1]"
        # kl_pred_output=self.btd_layer(self.kl_pred_output_fc,paddle.concat([kl_pred_gru_hid[:,:-1],a_his],axis=-1))

        # "kl_pred_loss"
        # "[stepn-1]"
        # mean2=gauss_param_mean[:,1:]
        # sigma2=paddle.exp(0.5*gauss_param_log_var[:,1:])
        # mean3=gauss_param_mean[:,:-1]
        # sigma3=paddle.exp(0.5*gauss_param_log_var[:,:-1])

        # kl_target=paddle.sum(gauss_KL(mean2,sigma2,mean3,sigma3),axis=-1,keepdim=True).detach()
        # # print(kl_target.shape)
        # kl_pred_loss=paddle.mean((kl_target-kl_pred_output)**2)

        total_loss=recon_loss+0.000001*kl_loss1#+0.1*kl_loss2#+kl_pred_loss

        self.optimizer.clear_grad()
        total_loss.backward()
        self.optimizer.step()
        self.lr_decay.step()
        # print(total_loss)
        # self.pred_result=np.transpose(F.softmax(pred).numpy(),[1,0])
        # self.label=np.transpose(np.array(s_history),[1,0])
        # print(F.softmax(decoder_pred).numpy(),np.array(s_history))
        return [recon_loss.numpy()[0],kl_loss1.numpy()[0],kl_loss2.numpy()[0]]#,kl_pred_loss.numpy()[0]]

    def test(self,s_history,a_history,vae_net):
        s_his=paddle.to_tensor(s_history)
        # s_his=paddle.cast(s_his,'float32')/255
        a_his=paddle.to_tensor(a_history)


        _b,_t,_=s_his.shape

        "feature"
        # s_feature=self.btchw_layer(self.encoder_input_conv,s_his).reshape([_b,_t,-1])
        sa_his=paddle.concat([s_his,a_his],axis=-1)
        sa_feature_encoder=self.btd_layer(self.encoder_input_fc,sa_his)

        "gru"
        encoder_output_his,_=self.encoder_gru(sa_feature_encoder,paddle.zeros([1,sa_his.shape[0],self.gru_dim]))

        "gauss_sample"
        "[init，step0,step1 ... stepn-1]"
        encoder_gru_hid=paddle.concat([paddle.zeros([sa_his.shape[0],1,self.gru_dim]),encoder_output_his],axis=-2)
        encoder_gauss_param=self.btd_layer(self.encoder_gauss,encoder_gru_hid)
        gauss_param_mean,gauss_param_log_var=encoder_gauss_param[:,:,:self.gru_dim],encoder_gauss_param[:,:,self.gru_dim:]
        encoder_hid=ReparamNormal(gauss_param_mean.shape).sample(gauss_param_mean,gauss_param_log_var)

        "decode"
        "[step0,step1 ... stepn-1]"
        sa_feature_decoder=self.btd_layer(self.decoder_input_fc,a_his)
        decoder_gru_hid,_=self.decoder_gru(sa_feature_decoder,paddle.reshape(encoder_hid[:,-1:,:],[1,_b,self.gru_dim]))
        decoder_output=self.btd_layer(self.decoder_output_fc,decoder_gru_hid)

        "recon last step"
        label=s_his[:,1:]
        recon_loss=paddle.mean((decoder_output[:,:-1]-label)**2)
        print("recon_loss=",recon_loss.numpy()[0])

        "recon"
        recon_img=self.btchw_layer(vae_net.decoder,decoder_output.reshape([_b,_t,4,8,8]))

        mean2=gauss_param_mean[:,1:]
        sigma2=paddle.exp(0.5*gauss_param_log_var[:,1:])
        mean3=gauss_param_mean[:,:-1]
        sigma3=paddle.exp(0.5*gauss_param_log_var[:,:-1])
        kl_loss2=gauss_KL(mean2,sigma2,mean3,sigma3)
        kl_loss2=paddle.sum(kl_loss2,axis=-1,keepdim=False)

        return recon_img.numpy(),kl_loss2.numpy()
