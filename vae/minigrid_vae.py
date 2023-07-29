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
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),nl_func()
            ),
            # Reslayer(
            #     nn.Conv2D(
            #         in_channels=hid_c,out_channels=hid_c,
            #         kernel_size=[3,3],stride=[1,1],padding='SAME'),nl_func()
            # ),
            nn.Conv2D(
                in_channels=hid_c,out_channels=v_z_c,
                kernel_size=[3,3],stride=[1,1],padding='SAME')
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


class MinigridVae(nn.Layer):
    def __init__(self):
        super().__init__()
        self.train_batchsize=1024
        self.train_start_size=100000

        self.vae_var=1.0**2

        self.encoder=Conv_Encoder_net([3,64,64])
        self.decoder=Conv_Decoder_net()

        self.lr_decay=paddle.optimizer.lr.MultiStepDecay(learning_rate=0.001, milestones=[3000], gamma=0.3)
        self.optimizer=optim.Adam(self.lr_decay,parameters=[
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            ])
        
        self.rpm=ReplayMemory(
            [
                # DataFormat("img",[self.history_len,self.action_dim],np.float32),
                DataFormat("img",[3,64,64],np.uint8),
                ],1000000)

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
            return self.train_history(train_dict['img'])
        else:
            # print("rmp size=",len(self.rpm),self.rpm._max_size)
            pass
    def save_model(self,save_dir,iter_num):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        paddle.save(self.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")

    def load_model(self,save_dir,iter_num):
        self.set_state_dict(paddle.load(f"{save_dir}/{iter_num}_model.pdparams"))

    def train_history(self,train_img):
        train_img=paddle.to_tensor(train_img)
        train_img=paddle.cast(train_img,'float32')/255

        encoder_hid_conv=self.encoder(train_img)
        mean=encoder_hid_conv
        log_var=paddle.zeros_like(mean)+np.log(self.vae_var)

        encoder_hid=ReparamNormal(mean.shape).sample(mean,log_var)
        decoder_result=self.decoder(encoder_hid)

        loss1=paddle.mean((decoder_result-train_img)**2)
        # print(paddle.sum(mean**2,axis=[1,2,3]).shape)
        loss2=paddle.mean(
            F.softplus(
                (
                    paddle.sum(mean**2,axis=[1,2,3])+1e-8)**0.5
                )
            )

        loss3=paddle.mean(mean**2)
        loss4=paddle.mean(
            (paddle.sum(mean**2,axis=[1,2,3])+1e-8)**0.5
            )

        loss=loss1+loss4*0.0001

        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_decay.step()
        return [loss1.numpy()[0],loss4.numpy()[0]]
    def pred(self,img_batch):
        _b=len(img_batch)
        img_batch=paddle.to_tensor(img_batch)
        img_batch=paddle.cast(img_batch,'float32')/255

        mean=self.encoder(img_batch).reshape([_b,-1])
        return mean.numpy()

        

    def test(self,test_img):
        test_img=paddle.to_tensor(test_img)
        test_img=paddle.cast(test_img,'float32')/255

        encoder_hid_conv=self.encoder(test_img)

        mean=encoder_hid_conv
        log_var=paddle.zeros_like(mean)+np.log(self.vae_var)
        encoder_hid=ReparamNormal(mean.shape).sample(mean,log_var)


        decoder_result=self.decoder(mean)
        return decoder_result.numpy()


       