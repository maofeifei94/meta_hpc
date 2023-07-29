from re import X
from turtle import forward

# from sympy import rational_interpolate
import os
import myf_ML_util
from myf_ML_util import Basic_Module, ReplayMemory,dense_block,DataFormat
import numpy as np
import paddle.nn as nn
import paddle
import paddle.optimizer as optim
import paddle.nn.functional as F

def nl_func():
    return nn.LeakyReLU(negative_slope=0.1)
class V_Conv_Encoder_net(nn.Layer):
    def __init__(self,img_shape):
        super(V_Conv_Encoder_net,self).__init__()
        self.conv_net=nn.Sequential(
            nn.Conv2D(
                in_channels=img_shape[0],out_channels=4,
                kernel_size=[3,3],stride=[2,2],padding='SAME'),nl_func(),
            nn.Conv2D(
                in_channels=4,out_channels=4,
                kernel_size=[3,3],stride=[2,2],padding='SAME'),nl_func(),
            nn.Conv2D(
                in_channels=4,out_channels=4,
                kernel_size=[3,3],stride=[2,2],padding='SAME'),nl_func()
        )
        self.mean=nn.Linear(256,256)
        self.std=nn.Linear(256,256)
    def forward(self,img):
        x=img
        x=self.conv_net(x)
        x=paddle.reshape(x,[x.shape[0],-1])
        mean=self.mean(x)
        logvar2=self.std(x) #logvar2=log(var**2)
        return mean,logvar2

class V_Conv_Decoder_net(nn.Layer):
    def __init__(self):
        super(V_Conv_Decoder_net,self).__init__()
        self.deconv_net=nn.Sequential(
            nn.Conv2DTranspose(
                in_channels=4,out_channels=4,
                kernel_size=[3,3],stride=2,padding="SAME"),nl_func(),
            nn.Conv2DTranspose(
                in_channels=4,out_channels=4,
                kernel_size=[3,3],stride=1,padding="SAME"),nl_func(),
            nn.Conv2DTranspose(
                in_channels=4,out_channels=4,
                kernel_size=[3,3],stride=2,padding="SAME"),nl_func(),
            nn.Conv2DTranspose(
                in_channels=4,out_channels=4,
                kernel_size=[3,3],stride=1,padding="SAME"),nl_func(),
            nn.Conv2DTranspose(
                in_channels=4,out_channels=4,
                kernel_size=[3,3],stride=2,padding="SAME"),nl_func(),
            nn.Conv2DTranspose(
                in_channels=4,out_channels=3,
                kernel_size=[3,3],stride=1,padding="SAME"),nn.Sigmoid(),
            )
    def forward(self,img):
        return self.deconv_net(img)
class V_Auto_Encoder(Basic_Module):
    def __init__(self) -> None:
        super(V_Auto_Encoder,self).__init__()
        self.train_batch_size=128
        self.hid_dim=64

        self.encoder=V_Conv_Encoder_net([3,64,64])
        # self.sigmoid_net=nn.Sequential(nn.Linear(64,64),nn.Sigmoid())
        self.hid_net=nn.Sequential(nn.Linear(256,256),nn.LeakyReLU(negative_slope=0.1),nn.Linear(256,256),nn.LeakyReLU(negative_slope=0.1))
        self.decoder=V_Conv_Decoder_net()
        self.optimizer=optim.Adam(
            learning_rate=0.0003,
            parameters=[*self.encoder.parameters(),*self.hid_net.parameters(),*self.decoder.parameters()]
            )
        self.rpm=ReplayMemory([DataFormat("img",[64,64,3],np.float32)],50000)

        self.avg_loss_recon=myf_ML_util.moving_avg()
        self.avg_loss_kl=myf_ML_util.moving_avg()
    def reparameterize(self, mu, logvar2):
        std = paddle.exp(0.5*logvar2)
        eps = paddle.randn(shape=std.shape)
        return mu + eps*std
    def reset(self):
        pass
    def input(self,img,pred_mean=False):
        "接受外界输入，返回处理结果"
        mean,logvar2=self.encoder(img)
        if pred_mean:
            return mean
        else:
            encoder_hid=self.reparameterize(mean,logvar2)
            # hid_sigmoid=F.sigmoid(encoder_hid)
            hid_sigmoid=encoder_hid
            return hid_sigmoid

    def rpm_collect(self,img):
        "收集训练数据"
        self.rpm.collect({'img':img})
        pass
    def rpm_clear(self):
        self.rpm.clear()
        pass

    def encode(self,img):
        mean,logvar2=self.encoder(img)
        encoder_hid=self.reparameterize(mean,logvar2)
        hid_sigmoid=encoder_hid
        return hid_sigmoid
    def decode(self,hid_sigmoid):
        hid_after_fc=self.hid_net(hid_sigmoid)
        hid_img=paddle.reshape(hid_after_fc,[-1,4,8,8])
        recon_img=self.decoder(hid_img)
        return recon_img

    def learn(self):
        "更新网络"
        train_data=self.rpm.sample_batch(self.train_batch_size)
        img=paddle.to_tensor(train_data['img'])
        img=paddle.transpose(img,perm=[0,3,1,2])

        mean,logvar2=self.encoder(img)
        encoder_hid=self.reparameterize(mean,logvar2)
        # hid_sigmoid=F.sigmoid(encoder_hid)
        hid_sigmoid=encoder_hid

        hid_after_fc=self.hid_net(hid_sigmoid)
        hid_img=paddle.reshape(hid_after_fc,[-1,4,8,8])
        recon_img=self.decoder(hid_img)


        loss_recon=paddle.mean((img-recon_img)**2)
        # loss_recon=paddle.mean(paddle.abs(img-recon_img))
        loss_kl=0.5*paddle.mean(mean.pow(2)+logvar2.exp()-logvar2-1)

        loss=loss_recon+loss_kl*0.00001

        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()

        # print(loss_recon.numpy(),loss_kl.numpy())

        self.avg_loss_kl.update(np.mean(loss_kl.numpy()))
        self.avg_loss_recon.update(np.mean(loss_recon.numpy()))

        return img,recon_img

    def save_model(self,save_dir,iter_num):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"start_save_model {save_dir}/{iter_num}_model.pdparams")
        paddle.save(self.encoder.state_dict(),f"{save_dir}/{iter_num}_encoder_model.pdparams")
        paddle.save(self.hid_net.state_dict(),f"{save_dir}/{iter_num}_hid_net_model.pdparams")
        paddle.save(self.decoder.state_dict(),f"{save_dir}/{iter_num}_decoder_model.pdparams")
        paddle.save(self.optimizer.state_dict(),f"{save_dir}/{iter_num}_optimizer.pdparams")
        print(f"save model {save_dir}/{iter_num}_model.pdparams success")

        iter_num="newest"
        paddle.save(self.encoder.state_dict(),f"{save_dir}/{iter_num}_encoder_model.pdparams")
        paddle.save(self.hid_net.state_dict(),f"{save_dir}/{iter_num}_hid_net_model.pdparams")
        paddle.save(self.decoder.state_dict(),f"{save_dir}/{iter_num}_decoder_model.pdparams")
        paddle.save(self.optimizer.state_dict(),f"{save_dir}/{iter_num}_optimizer.pdparams")
        
        
    def load_model(self,save_dir,iter_num):
        print(f"start load_model {save_dir}/{iter_num}_model.pdparams")
        self.encoder.set_state_dict(paddle.load(path=f"{save_dir}/{iter_num}_encoder_model.pdparams"))
        self.hid_net.set_state_dict(paddle.load(path=f"{save_dir}/{iter_num}_hid_net_model.pdparams"))
        self.decoder.set_state_dict(paddle.load(path=f"{save_dir}/{iter_num}_decoder_model.pdparams"))
        self.optimizer.set_state_dict(paddle.load(path=f"{save_dir}/{iter_num}_optimizer.pdparams"))
        print(f"load model {save_dir}/{iter_num}_model.pdparams success")
class Reslayer(nn.Layer):
    def __init__(self,dim):
        super(Reslayer,self).__init__()
        self.layer=nn.Sequential(nn.Linear(dim,dim),nl_func())
    def forward(self,x):
        return x+self.layer(x)



class Place_imging_module(Basic_Module):
    def __init__(self,input_dim,train_batch_size=128):
        self.imaging_net=nn.Sequential(
            nn.Linear(input_dim,256),nl_func(),
            Reslayer(256),
            Reslayer(256),
            Reslayer(256),
            Reslayer(256),
            Reslayer(256),
            Reslayer(256),
            Reslayer(256),
            Reslayer(256),
            Reslayer(256),
            Reslayer(256),
            nn.Linear(256,256)
        )
        self.optimizer=optim.Adam(learning_rate=0.0003,parameters=self.imaging_net.parameters())
        self.rpm=myf_ML_util.ReplayMemory([
            DataFormat('data',[input_dim],np.float32),
            DataFormat('label',[256],np.float32)
            ],20000)

        self.avg_loss=myf_ML_util.moving_avg()
        self.train_batch_size=train_batch_size
    def reset(self):
        self.rpm_clear()
    def input(self,hpc_input):
        "接受外界输入，返回处理结果"
        return self.imaging_net(hpc_input)
    def rpm_collect(self,hpc_input,label):
        "收集训练数据"
        self.rpm.collect({'data':hpc_input,'label':label})

    def rpm_clear(self):
        self.rpm.clear()
        "清空rpm的训练数据"
        pass
    def learn(self):
        "更新网络"
        train_data_dict=self.rpm.sample_batch(self.train_batch_size)
        data=train_data_dict['data']
        label=train_data_dict['label']
        data=paddle.to_tensor(data)
        label=paddle.to_tensor(label)

        pred_img_hid=self.imaging_net(data)
        loss=paddle.mean((pred_img_hid-label)**2)

        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()

        self.avg_loss.update(np.mean(loss.numpy()))
        
        pass
    def save_model(self,save_dir,iter_num):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"start_save_model {save_dir}/{iter_num}_model.pdparams")
        paddle.save(self.imaging_net.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")
        paddle.save(self.optimizer.state_dict(),f"{save_dir}/{iter_num}_optimizer.pdparams")
        print(f"save model {save_dir}/{iter_num}_model.pdparams success")

        iter_num="newest"
        paddle.save(self.imaging_net.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")
        paddle.save(self.optimizer.state_dict(),f"{save_dir}/{iter_num}_optimizer.pdparams")
        
    def load_model(self,save_dir,iter_num):
        print(f"start load_model {save_dir}/{iter_num}_model.pdparams")
        self.imaging_net.set_state_dict(paddle.load(path=f"{save_dir}/{iter_num}_model.pdparams"))
        self.optimizer.set_state_dict(paddle.load(path=f"{save_dir}/{iter_num}_optimizer.pdparams"))
        print(f"load model {save_dir}/{iter_num}_model.pdparams success")

        
class Reward_HPC_Module(Basic_Module):
    def __init__(self,input_dim,train_batch_size=128):
        self.reward_net=nn.Sequential(
            nn.Linear(input_dim,256),nl_func(),
            nn.Linear(256,256),nl_func(),
            nn.Linear(256,128),nl_func(),
            nn.Linear(128,64),nl_func(),
            nn.Linear(64,1)
        )
        self.optimizer=optim.Adam(learning_rate=0.0003,parameters=self.reward_net.parameters())
        self.rpm=myf_ML_util.ReplayMemory([
            DataFormat('data',[input_dim],np.float32),
            DataFormat('label',[1],np.float32),
            DataFormat('ratio',[1],np.float32)
            ],20000)

        self.avg_loss=myf_ML_util.moving_avg()
        self.train_batch_size=train_batch_size
    def reset(self):
        self.rpm_clear()
    def input(self,hpc_input):
        "接受外界输入，返回处理结果"
        return self.reward_net(hpc_input)
    def rpm_collect(self,hpc_input,label):
        "收集训练数据"
        self.rpm.collect({'data':hpc_input,'label':label,'ratio':1})

    def rpm_clear(self):
        self.rpm.clear()
        "清空rpm的训练数据"
        pass
    def count_rpm_label(self):
        all_label=self.rpm._data_dict['label'][:self.rpm._size]
        num_dict={}
        for _label in all_label:
            l=_label[0]
            # print(l,num_dict.keys())
            if l in num_dict.keys():
                num_dict[l]+=1
            else:
                num_dict[l]=1
        print("num_dict=",num_dict)


        # all_label_keys=list(num_dict.keys())
        # min_label,max_label=np.min(all_label_keys),np.max(all_label_keys)
        ratio_dict={}
        total_num=np.sum([num_dict[key] for key in num_dict.keys()])

        for key in num_dict.keys():
            ratio_dict[key]=len(list(num_dict.keys()))*min(total_num/num_dict[key],10)/np.sum([min(total_num/num_dict[_key],10) for _key in num_dict.keys()])

        # print("ori ratio_dict",ratio_dict)
        # ratio_dict={key:min(2,ratio_dict[key]) for key in ratio_dict.keys()}
        # print("step1 ratio_dict",ratio_dict)
        # ratio_dict={key:len(list(ratio_dict.keys))*ratio_dict[key]/np.sum(ratio_dict.keys()) for key in ratio_dict.keys()}
        print("final ratio_dict",ratio_dict)

        for i in range(self.rpm._size):
            # print(f"label {i}",self.rpm._data_dict['label'][i])
            self.rpm._data_dict['ratio'][i]=ratio_dict[self.rpm._data_dict['label'][i][0]]
            # print(self.rpm._data_dict['label'][i],self.rpm._data_dict['ratio'][i])

    def learn(self):
        "更新网络"
        train_data_dict=self.rpm.sample_batch(self.train_batch_size)
        data=train_data_dict['data']
        label=train_data_dict['label']
        ratio=train_data_dict['ratio']
        data=paddle.to_tensor(data)
        label=paddle.to_tensor(label)
        ratio=paddle.to_tensor(ratio)

        pred_reward=self.reward_net(data)
        # print("train ratio=",ratio)
        loss=paddle.mean(ratio*(pred_reward-label)**2)

        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()
        
        self.avg_loss.update(np.mean(loss.numpy()))

        pass
    def save_model(self,save_dir,iter_num):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"start_save_model {save_dir}/{iter_num}_model.pdparams")
        paddle.save(self.reward_net.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")
        paddle.save(self.optimizer.state_dict(),f"{save_dir}/{iter_num}_optimizer.pdparams")
        print(f"save model {save_dir}/{iter_num}_model.pdparams success")

        iter_num="newest"
        paddle.save(self.reward_net.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")
        paddle.save(self.optimizer.state_dict(),f"{save_dir}/{iter_num}_optimizer.pdparams")
        
    def load_model(self,save_dir,iter_num):
        print(f"start load_model {save_dir}/{iter_num}_model.pdparams")
        self.reward_net.set_state_dict(paddle.load(path=f"{save_dir}/{iter_num}_model.pdparams"))
        self.optimizer.set_state_dict(paddle.load(path=f"{save_dir}/{iter_num}_optimizer.pdparams"))
        print(f"load model {save_dir}/{iter_num}_model.pdparams success")
    


    