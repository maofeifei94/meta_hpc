import os
import myf_ML_util
from myf_ML_util import Basic_Module, ReplayMemory,dense_block,DataFormat
import numpy as np
import paddle.nn as nn
import paddle
import paddle.optimizer as optim
import paddle.nn.functional as F

v_encoder_hid_c=8
v_z_c=2
v_decoder_hic_c=8

def nl_func():
    return nn.Tanh()
class Reslayer(nn.Layer):
    def __init__(self,layer,func):
        super(Reslayer,self).__init__()
        self.layer=layer
        self.func=func
    def forward(self,x):
        return x+self.func(self.layer(x))
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
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),nl_func()
            ),
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),nl_func()
            ),
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),nl_func()
            ),
            nn.Conv2D(
            in_channels=hid_c,out_channels=v_z_c,
            kernel_size=[3,3],stride=[1,1],padding='SAME'),nn.Sigmoid()

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
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),nl_func()
            ),
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),nl_func()
            ),
            nn.Conv2DTranspose(
                in_channels=hid_c,out_channels=hid_c,
                kernel_size=[2,2],stride=2,padding='VALID',),nl_func(),
            nn.Conv2DTranspose(
                in_channels=hid_c,out_channels=hid_c,
                kernel_size=[2,2],stride=2,padding="VALID"),nl_func(),
            nn.Conv2D(
                in_channels=hid_c,out_channels=hid_c,
                kernel_size=[3,3],stride=1,padding="SAME"),nl_func(),
            nn.Conv2D(
                in_channels=hid_c,out_channels=1,
                kernel_size=[3,3],stride=1,padding="SAME"),nn.Sigmoid(),
            )
    def forward(self,img):
        return self.deconv_net(img)

class Conv_Auto_Encoder(nn.Layer):
    def __init__(self) -> None:
        super(Conv_Auto_Encoder,self).__init__()
        self.train_batch_size=128

        self.encoder=Conv_Encoder_net([1,28,28])
        self.decoder=Conv_Decoder_net()
        self.optimizer=optim.Adam(
            learning_rate=0.0003,
            parameters=[*self.encoder.parameters(),*self.decoder.parameters()]
            )

        self.rpm=ReplayMemory([DataFormat("img",[28,28,1],np.float32)],50000)
        self.avg_loss_recon=myf_ML_util.moving_avg()

    def reset(self):
        pass

    def rpm_collect(self,img):
        "收集训练数据"
        self.rpm.collect({'img':img})
        pass
    def rpm_clear(self):
        self.rpm.clear()
        pass

    def encode(self,img):
        z=self.encoder(img)
        return z
    def decode(self,z):
        recon_img=self.decoder(z)
        return recon_img
    
    def ae(self,img):
        img=paddle.to_tensor(img)
        img=paddle.transpose(img,perm=[0,3,1,2])
        z=self.encoder(img)
        recon_img=self.decoder(z)
        return recon_img


    def learn(self):
        "更新网络"
        train_data=self.rpm.sample_batch(self.train_batch_size)
        img=paddle.to_tensor(train_data['img'])
        img=paddle.transpose(img,perm=[0,3,1,2])

        z=self.encoder(img)

        recon_img=self.decoder(z)


        loss_recon=0.5*paddle.mean((img-recon_img)**2)
        # loss_recon=paddle.mean(paddle.abs(img-recon_img))
        loss=loss_recon

        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()

        # print(loss_recon.numpy(),loss_kl.numpy())

        self.avg_loss_recon.update(np.mean(loss_recon.numpy()))

        return img,recon_img

    def save_model(self,save_dir,iter_num):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"start_save_model {save_dir}/{iter_num}_model.pdparams")
        paddle.save(self.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")
        print(f"save model {save_dir}/{iter_num}_model.pdparams success")
        paddle.save(self.state_dict(),f"{save_dir}/newest_model.pdparams")

    def load_model(self,save_dir,iter_num):
        print(f"start load_model {save_dir}/{iter_num}_model.pdparams")
        self.set_state_dict(paddle.load(path=f"{save_dir}/{iter_num}_model.pdparams"))
        print(f"load model {save_dir}/{iter_num}_model.pdparams success")
