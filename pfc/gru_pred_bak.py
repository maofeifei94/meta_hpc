import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
import numpy as np
import os
from mlutils.ml import dense_block,ReparamNormal,gauss_KL,DataFormat,ReplayMemory
from train_pred_net import train_pred
class GruPredNext(nn.Layer):
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        gru_dim=128
        box_dim=10
        reward_dim=2
        self.gru_dim=gru_dim
        self.box_dim=box_dim
        self.reward_dim=reward_dim
        self.feature_net=dense_block([box_dim+reward_dim,32,32,32,32])
        self.gru=nn.GRU(32,128)
        self.pred_net=dense_block([box_dim+gru_dim,64,64,64,64,reward_dim])

        self.optimizer=optim.Adam(0.0001,parameters=[*self.feature_net.parameters(),*self.gru.parameters(),*self.pred_net.parameters()])
    
    def train_history(self,history):
        history_paddle=paddle.to_tensor(np.array(history,np.float32),'float32')
        history_feature=self.feature_net(history_paddle)
        # print(history_feature.shape)
        gru_result,_=self.gru(paddle.reshape(history_feature,[1,*history_feature.shape]),paddle.zeros([1,1,self.gru_dim]))
        # print(gru_result.shape)
        
        "forward pred"
        gru_forward_result=gru_result[0,:-1,:]
        gru_forward_result=paddle.concat([paddle.zeros([1,self.gru_dim]),gru_forward_result],axis=0)
        forward_query=history_paddle.clone()[:,:self.box_dim]
        forward_label=history_paddle.clone()[:,-self.reward_dim:]
        forward_pred=self.pred_net(paddle.concat([gru_forward_result,forward_query],axis=-1))

        "mask pred"
        history_mask=history_paddle.clone()[:-1]
        history_mask[:,-self.reward_dim:]=-1
        history_mask_feature=self.feature_net(history_mask)
        # print(history_mask[0],history_paddle[0])
        gru_mask_result,_=self.gru(
            paddle.reshape(history_mask_feature,[history_mask_feature.shape[0],1,history_mask_feature.shape[1]]),
            paddle.concat([paddle.zeros([1,1,self.gru_dim]),gru_result[:,:-2]],axis=1)
            )
        gru_mask_result=gru_mask_result[:,0]
        mask_query=history_paddle.clone()[1:,:self.box_dim]
        mask_label=history_paddle.clone()[1:,-self.reward_dim:]
        # print(gru_forward_result.shape,gru_mask_result.shape,query.shape,label.shape)

        mask_pred=self.pred_net(paddle.concat([gru_mask_result,mask_query],axis=-1))

        loss_forward=paddle.mean(F.softmax_with_cross_entropy(forward_pred,forward_label,True))
        loss_mask=paddle.mean(F.softmax_with_cross_entropy(mask_pred,mask_label,True))
        total_loss=loss_forward+loss_mask

        self.optimizer.clear_grad()
        total_loss.backward()
        self.optimizer.step()

        return [loss_forward.numpy()[0],loss_mask.numpy()[0]]

    def reset_pred(self):
        self.h_t_1=paddle.zeros([1,1,self.gru_dim])
        self.h_t_2=None
        # self.feature_t_1=None
        self.obs_t_1=None

    def pred(self,obs):
        obs_t=paddle.to_tensor(np.array(obs,np.float32),'float32').reshape([1,-1])
        obs_feature_t=self.feature_net(obs_t)

        if self.h_t_2 is not None and self.obs_t_1 is not None:
            mask_obs_t_1=self.obs_t_1.clone()
            mask_obs_t_1[:,-self.reward_dim:]=-1
            _,mask_h_t_1=self.gru(self.feature_net(mask_obs_t_1).reshape([1,1,-1]),self.h_t_2)
            mask_pred_t=self.pred_net(paddle.concat([mask_h_t_1[0],obs_t[:,:self.box_dim]],axis=-1))

            forward_pred_t=self.pred_net(paddle.concat([self.h_t_1[0],obs_t[:,:self.box_dim]],axis=-1))

            mask_pred_prob=F.softmax(mask_pred_t)
            forward_pred_prob=F.softmax(forward_pred_t)
        #update
        _,h_t=self.gru(obs_feature_t.reshape([1,1,-1]),self.h_t_1)
        self.h_t_2=self.h_t_1
        self.h_t_1=h_t
        self.obs_t_1=obs_t

        return forward_pred_prob,mask_pred_prob
# class GruPredKL(nn.Layer):
#     def __init__(self):
#         super().__init__()
#         self.train_batchsize=1024
#         self.gru_dim=256
#         self.out_dim=1

#         self.encoder_input_fc=dense_block([self.action_dim+self.state_dim,32,32,32])
#         self.encoder_gru=nn.GRU(32,self.gru_dim)

class KLPredNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.train_batchsize=1024
        self.train_start_size=100000
        self.gru_dim=256
        self.action_dim=10
        self.state_dim=1

        self.kl_pred_input_fc=dense_block([self.action_dim+self.state_dim,64,64])
        self.kl_pred_gru=nn.GRU(64,self.gru_dim)
        self.kl_pred_output_fc=dense_block([self.gru_dim+self.action_dim,64,64,1],act_output=nn.Softplus())
        
        self.lr_decay=paddle.optimizer.lr.MultiStepDecay(learning_rate=0.001, milestones=[10000,30000], gamma=1/10**0.5)
        self.optimizer=optim.Adam(0.0003,parameters=[
            *self.kl_pred_input_fc.parameters(),
            *self.kl_pred_gru.parameters(),
            *self.kl_pred_output_fc.parameters(),
            ])

        self.rpm=ReplayMemory(
            [
                DataFormat("action_history",[20,10],np.float32),
                DataFormat("obs_history",[20,1],np.float32)
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
            # for key in train_dict.keys():
            #     print(key,np.shape(train_dict[key]))
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

        "kl pred"
        kl_pred_sa_his_feature=self.btd_layer(self.kl_pred_input_fc,sa_his)
        kl_pred_gru_hid,_=self.kl_pred_gru(kl_pred_sa_his_feature,paddle.zeros([1,kl_pred_sa_his_feature.shape[0],self.gru_dim]))
        # kl_pred_output_input=
        "[init，step0,step1 ... stepn-1]"
        kl_pred_gru_hid=paddle.concat([paddle.zeros([kl_pred_sa_his_feature.shape[0],1,self.gru_dim]),kl_pred_gru_hid],axis=-2)
        
        # ###################  entropy loss
        # "[step0,step1 ... stepn-1]"
        # kl_pred_output=self.btd_layer(self.kl_pred_output_fc,paddle.concat([kl_pred_gru_hid[:,:-1],a_his],axis=-1))
        # kl_pred_label=paddle.cast(s_his,'int64')
        # loss=paddle.mean(F.softmax_with_cross_entropy(kl_pred_output,kl_pred_label))

        ###################  L2 loss
        "[step0,step1 ... stepn-1]"
        kl_pred_output=self.btd_layer(self.kl_pred_output_fc,paddle.concat([kl_pred_gru_hid[:,:-1],a_his],axis=-1))
        kl_pred_label=s_his
        loss=paddle.mean((kl_pred_output-kl_pred_label)**2)

        # loss=paddle.mean((kl_pred_output-kl_pred_label)**2)
        # print(kl_pred_output,kl_pred_label)
        
        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_decay.step()
        return loss.numpy()[0]
    def test(self,s_history,a_history):
        s_his=paddle.to_tensor(s_history)
        a_his=paddle.to_tensor(a_history)
        sa_his=paddle.concat([s_his,a_his],axis=-1)

        "kl pred"
        kl_pred_sa_his_feature=self.btd_layer(self.kl_pred_input_fc,sa_his)
        kl_pred_gru_hid,_=self.kl_pred_gru(kl_pred_sa_his_feature,paddle.zeros([1,kl_pred_sa_his_feature.shape[0],self.gru_dim]))
        # kl_pred_output_input=
        "[init，step0,step1 ... stepn-1]"
        kl_pred_gru_hid=paddle.concat([paddle.zeros([kl_pred_sa_his_feature.shape[0],1,self.gru_dim]),kl_pred_gru_hid],axis=-2)
        
        # ###################  entropy loss
        # "[step0,step1 ... stepn-1]"
        # kl_pred_output=self.btd_layer(self.kl_pred_output_fc,paddle.concat([kl_pred_gru_hid[:,:-1],a_his],axis=-1))
        # kl_pred_label=paddle.cast(s_his,'int64')
        # loss=paddle.mean(F.softmax_with_cross_entropy(kl_pred_output,kl_pred_label))

        ###################  L2 loss
        "[step0,step1 ... stepn-1]"
        kl_pred_output=self.btd_layer(self.kl_pred_output_fc,paddle.concat([kl_pred_gru_hid[:,:-1],a_his],axis=-1))
        kl_pred_label=s_his
        loss=paddle.mean((kl_pred_output-kl_pred_label)**2)
        
        # kl_pred_output=F.softmax(kl_pred_output)[:,:,1:]

        return kl_pred_output,loss

class GruPredHid(nn.Layer):
    def __init__(self):
        super().__init__()
        self.train_batchsize=1024
        self.gru_dim=256
        self.action_dim=10
        self.state_dim=2

        # 
        # self.feature_net=dense_block([box_dim+reward_dim,32,32,32,32])
        self.encoder_input_fc=dense_block([self.action_dim+self.state_dim,32,32,32])
        # self.encoder=dense_block([self.action_dim+self.state_dim,self.gru_dim,self.gru_dim,self.gru_dim,self.gru_dim])
        self.encoder_gru=nn.GRU(32,self.gru_dim)
        self.encoder_gauss=dense_block([self.gru_dim,2*self.gru_dim,2*self.gru_dim,2*self.gru_dim])

        self.decoder_input_fc=dense_block([self.action_dim,32,32,32])
        self.decoder_gru=nn.GRU(32,self.gru_dim)
        self.decoder_output_fc=dense_block([self.gru_dim,32,32,32,self.state_dim])
        # self.decoder=dense_block([self.action_dim+self.gru_dim,self.gru_dim,self.gru_dim,self.state_dim])
        # self.future_pred_fc=dense_block([self.gru_dim+self.action_dim,32,32,32,self.state_dim])
        self.dis_pred_fc=dense_block([self.gru_dim,32,32,32,self.action_dim])
        # self.pred_net=dense_block([box_dim+gru_dim,64,64,64,64,reward_dim])

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
                ],50000)

    def rpm_collect(self,data_dict):
        self.rpm.collect(data_dict)
    def btd_layer(self,net,x):
        batch,time,dim=x.shape
        return net(x.reshape([batch*time,dim])).reshape([batch,time,-1])
    def bttd_layer(self,net,x):
        batch,time,time,dim=x.shape
        return net(x.reshape([batch*time*time,dim])).reshape([batch,time,time,-1])
    def learn(self):
        train_dict=self.rpm.sample_batch(self.train_batchsize)
        # for key in train_dict.keys():
        #     print(key,np.shape(train_dict[key]))
        return self.train_history(train_dict["obs_history"],train_dict["action_history"])
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

        # print("s_his,a_his",s_his.shape,a_his.shape)
        "encode"
        sa_his_feature=self.btd_layer(self.encoder_input_fc,sa_his)
        # print(sa_his)
        encoder_output_his,_=self.encoder_gru(sa_his_feature,paddle.zeros([1,sa_his_feature.shape[0],self.gru_dim]))

        "[init，step0,step1 ... stepn-1]"
        encoder_gru_hid=paddle.concat([paddle.zeros([sa_his_feature.shape[0],1,self.gru_dim]),encoder_output_his],axis=-2)
        encoder_gauss_param=self.btd_layer(self.encoder_gauss,encoder_gru_hid)
        gauss_param_mean,gauss_param_log_var=encoder_gauss_param[:,:,:self.gru_dim],encoder_gauss_param[:,:,self.gru_dim:]
        encoder_hid=ReparamNormal(gauss_param_mean.shape).sample(gauss_param_mean,gauss_param_log_var)

        
        # "recon all
        # a_his_feature=self.btd_layer(self.decoder_input_fc,a_his)
        # _b,_t,_d=a_his_feature.shape
        # a_his_feature=paddle.reshape(a_his_feature,[_b,1,_t,_d])
        # a_his_feature=paddle.tile(a_his_feature,[1,_t,1,1]).reshape([_b*_t,_t,_d])

        # decoder_pred_gru,_=self.decoder_gru(a_his_feature,paddle.reshape(encoder_hid,[1,_b*_t,self.gru_dim]))
        # decoder_pred_gru=decoder_pred_gru.reshape([_b,_t,_t,-1])
        # pred=self.bttd_layer(self.decoder_output_fc,decoder_pred_gru)

        # recon_loss_mask=paddle.tril(paddle.ones([_t,_t]))
        # recon_loss_mask=paddle.reshape(recon_loss_mask,[_t,_t,1])
        
        # label=paddle.tile(s_his.reshape([_b,1,_t,-1]),[1,_t,1,1])

        # loss_ratio=s_his[:,:,1:]*(5-5.0/9.0)+5.0/9.0
        # loss_ratio=paddle.tile(loss_ratio.reshape([_b,1,_t,-1]),[1,_t,1,1])
        # recon_loss=paddle.mean(F.softmax_with_cross_entropy(pred,label,soft_label=True)*recon_loss_mask*loss_ratio)*2

        "recon last step"
        a_his_feature=self.btd_layer(self.decoder_input_fc,a_his)
        _b,_t,_d=a_his_feature.shape
        # a_his_feature=paddle.reshape(a_his_feature,[_b,1,_t,_d])
        # a_his_feature=paddle.tile(a_his_feature,[1,_t,1,1]).reshape([_b*_t,_t,_d])

        "[step0,step1 ... stepn-1]"
        decoder_pred_gru,_=self.decoder_gru(a_his_feature,paddle.reshape(encoder_hid[:,-1:,:],[1,_b,self.gru_dim]))
        pred=self.btd_layer(self.decoder_output_fc,decoder_pred_gru)

        
        label=s_his

        loss_ratio=paddle.cast(paddle.sum(s_his[:,:,1:],axis=-2,keepdim=True)>0.5,'float32')
        # print(a_his)
        # print(s_his)
        # print(loss_ratio)
        # loss_ratio=paddle.tile(loss_ratio.reshape([_b,1,_t,-1]),[1,_t,1,1])
        recon_loss=paddle.mean(F.softmax_with_cross_entropy(pred,label,soft_label=True)*loss_ratio)*2


        # print(loss_ratio.shape,s_his.shape,F.softmax_with_cross_entropy(pred,s_his,soft_label=True).shape)
        # print(a_his.shape,s_his.shape,pred.shape,encoder_hid.shape,hid_repeat.shape)
        # print(F.sigmoid(pred),s_his)
        # recon_loss=paddle.mean((F.sigmoid(pred)-s_his)**2)
        "obs pred"

        # count_vec=paddle.tile(a_his.reshape([_b,1,_t,self.action_dim]),[1,_t,1,1])
        # count_mask=paddle.tril(paddle.ones([_t,_t]))
        # count_mask=paddle.reshape(count_mask,[_t,_t,1])
        # count_label=paddle.cast(paddle.sum(count_vec*count_mask,axis=-2,keepdim=False)>0.5,'float32')
        # count_label=paddle.cast(count_label.reshape([_b,_t,self.action_dim,1]),'int64')

        # pred_count=self.btd_layer(self.obs_count_fc,encoder_hid)
        
        # pred_count=pred_count.reshape([_b,_t,self.action_dim,2])
       
        # count_loss=paddle.mean(F.softmax_with_cross_entropy(pred_count,count_label))
        # print(a_his.shape,s_his.shape,encoder_hid.shape)


        # a_tile=a_his.reshape([_b,1,_t,-1]).tile([1,_t,1,1])
        # s_tile=s_his.reshape([_b,1,_t,-1]).tile([1,_t,1,1])
        # hid_tile=encoder_hid.reshape([_b,_t,1,-1]).tile([1,1,_t,1])

        # pred_future=self.bttd_layer(self.future_pred_fc,paddle.concat([hid_tile,a_tile],axis=-1))
        # # print(pred_future.shape,s_tile.shape)
        # future_loss=paddle.mean(F.softmax_with_cross_entropy(pred_future,s_tile,soft_label=True))
        a_tile=a_his.reshape([_b,1,_t,-1]).tile([1,_t,1,1])
        a_mask=paddle.tril(paddle.ones([_t,_t])).reshape([_t,_t,1])

        s_tile=s_his.reshape([_b,1,_t,-1]).tile([1,_t,1,1])

        box_opened=paddle.cast(paddle.sum(a_tile*a_mask,axis=-2,keepdim=False)>0.5,'float32')
        box_not_opened=1-box_opened
        box_contained=paddle.cast(paddle.sum(a_tile*a_mask*s_tile[:,:,:,1:],axis=-2,keepdim=False)>0.5,'float32')

        find_contained=paddle.tile(paddle.sum(box_contained,axis=-1,keepdim=True),[1,1,self.action_dim])

        dis_label=paddle.where(find_contained>0.5,box_contained,box_not_opened/paddle.sum(box_not_opened,axis=-1,keepdim=True))

        dis_pred=self.btd_layer(self.dis_pred_fc,encoder_hid)
        # dis_loss=paddle.mean(F.softmax_with_cross_entropy(dis_pred,dis_label,soft_label=True))

        # print(a_his,box_opened,box_contained,dis_label)

        
        # print(recon_loss_mask[5])
        "kl loss"

        "[stepn-1]"
        mean0=gauss_param_mean[:,-1:]
        sigma0=paddle.exp(0.5*gauss_param_log_var[:,-1:])
        "[init，step0,step1 ... stepn-1]"
        mean1=gauss_param_mean
        sigma1=paddle.exp(0.5*gauss_param_log_var)

        # print(s_history[0])
        kl_mask=paddle.cast(paddle.sum(s_his[:,:,1:],axis=1,keepdim=True)>0.5,'float32')
        # print(gauss_KL(mean0,sigma0,mean1,sigma1).shape,kl_mask.shape)
        kl_loss1=paddle.mean(gauss_KL(mean0,sigma0,mean1,sigma1)*kl_mask)

        mean2=gauss_param_mean[:,1:]
        sigma2=paddle.exp(0.5*gauss_param_log_var[:,1:])
        mean3=gauss_param_mean[:,:-1]
        sigma3=paddle.exp(0.5*gauss_param_log_var[:,:-1])
        kl_loss2=paddle.mean(gauss_KL(mean2,sigma2,mean3,sigma3))

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
        kl_target=paddle.where(kl_target>1.0,kl_target*0+1,kl_target*0).detach()
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
