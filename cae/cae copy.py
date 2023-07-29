import os
import numpy as np
import paddle.nn as nn
import paddle
import paddle.optimizer as optim
import paddle.nn.functional as F

from mlutils.ml import ReparamNormal,moving_avg,ModelSaver


class Reslayer(nn.Layer):
    def __init__(self,layer,func):
        super(Reslayer,self).__init__()
        self.layer=layer
        self.func=func
    def forward(self,x):
        return x+self.func(self.layer(x))
"""
卷积自编码器，输入为图片，输出为降维后的向量
"""
class Conv_Encoder_net(nn.Layer):
    def __init__(self,img_shape,cae_hp):
        super(Conv_Encoder_net,self).__init__()
        self.cae_hp=cae_hp
        self.nl_func={'Tanh':nn.Tanh(),"LeakyRelu":nn.LeakyReLU(0.1)}[self.cae_hp.nl_func]

        hid_c=self.cae_hp.v_encoder_hid_c
        print(img_shape,hid_c)
        self.conv_net=nn.Sequential(
            nn.Conv2D(
                in_channels=img_shape[0],out_channels=hid_c,
                kernel_size=[3,3],stride=[2,2],padding='SAME'),self.nl_func,
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),self.nl_func
            ),
            nn.Conv2D(
                in_channels=hid_c,out_channels=hid_c,
                kernel_size=[3,3],stride=[2,2],padding='SAME'),self.nl_func,
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),self.nl_func
            ),
            nn.Conv2D(
                in_channels=hid_c,out_channels=hid_c,
                kernel_size=[3,3],stride=[2,2],padding='SAME'),self.nl_func,
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),self.nl_func
            ),
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),self.nl_func
            ),
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),self.nl_func
            ),
            nn.Conv2D(
            in_channels=hid_c,
            #若使用vae，则channel变为2倍
            out_channels=self.cae_hp.v_z_c*2,
            kernel_size=[3,3],stride=[1,1],padding='SAME')
        )
        print("finish init cae_encoder")

    def forward(self,img):
        x=img
        x=self.conv_net(x)

        return x

class Conv_Decoder_net(nn.Layer):
    def __init__(self,cae_hp):
        super(Conv_Decoder_net,self).__init__()
        self.cae_hp=cae_hp
        self.nl_func={'Tanh':nn.Tanh(),"LeakyRelu":nn.LeakyReLU(0.1)}[self.cae_hp.nl_func]
        hid_c=self.cae_hp.v_decoder_hic_c
        self.deconv_net=nn.Sequential(
            nn.Conv2D(
                in_channels=self.cae_hp.v_z_c,out_channels=hid_c,
                kernel_size=[3,3],stride=[1,1],padding='SAME'),self.nl_func,
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),self.nl_func
            ),
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),self.nl_func
            ),
            nn.Conv2DTranspose(
                in_channels=hid_c,out_channels=hid_c,
                kernel_size=[3,3],stride=2,padding="SAME"),self.nl_func,
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),self.nl_func
            ),
            nn.Conv2DTranspose(
                in_channels=hid_c,out_channels=hid_c,
                kernel_size=[3,3],stride=2,padding="SAME"),self.nl_func,
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),self.nl_func
            ),
            nn.Conv2DTranspose(
                in_channels=hid_c,out_channels=hid_c,
                kernel_size=[3,3],stride=2,padding="SAME"),self.nl_func,
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),self.nl_func
            ),
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),self.nl_func
            ),
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),self.nl_func
            ),
            nn.Conv2D(
                in_channels=hid_c,out_channels=3,
                kernel_size=[3,3],stride=1,padding="SAME"),nn.Sigmoid(),
            )
    def forward(self,img):
        return self.deconv_net(img)

class Conv_Auto_Encoder(nn.Layer,ModelSaver):
    def __init__(self,cae_hp) -> None:
        super(Conv_Auto_Encoder,self).__init__()
        self.cae_hp=cae_hp

        self.encoder=Conv_Encoder_net([3,64,64],cae_hp)
        self.decoder=Conv_Decoder_net(cae_hp)
        self.optimizer=optim.Adam(
            learning_rate=0.0003,
            parameters=[*self.encoder.parameters(),*self.decoder.parameters()]
            )
        self.avg_loss_recon=moving_avg(gamma=0.995)
        self.avg_loss_kl=moving_avg(gamma=0.995)

        # self.weight_mask=self.get_weight_mask()
        self.weight_mask=self.get_gauss_weight_mask()
    def get_weight_mask(self):
        mask=paddle.zeros([64,64],'float32')
        mask[28:36][28:36]=0.999
        mask+=0.001
        mask*=paddle.sum(paddle.ones_like(mask))/paddle.sum(mask)
        return mask
    def get_gauss_weight_mask(self):
        site_max=64
        center=paddle.to_tensor(np.array([31.5,31.5]),'float32')
        gauss_ratio=0.05

        site_list=paddle.to_tensor(list(range(site_max)),'float32')

        h_site=paddle.reshape(site_list,[-1,1,1])
        w_site=paddle.reshape(site_list,[1,-1,1])

        hw_site=paddle.concat([paddle.tile(h_site,[1,site_max,1]),paddle.tile(w_site,[site_max,1,1])],axis=2)
        # print(hw_site)
        distance=paddle.sum((hw_site-center)**2,axis=-1)
        mask=paddle.exp(-distance*gauss_ratio)
        mask+=0.001
        mask*=paddle.sum(paddle.ones_like(mask))/paddle.sum(mask)
        return mask



    def reset(self):
        pass
    def encode(self,img):
        z=self.gauss_sample(self.encoder(img)) if self.cae_hp.use_vae else self.encoder(img)
        return z
    def gauss_sample(self,z):
        "z.shape=[b,c,8,8]"
        mean=z[:,:self.cae_hp.v_z_c]
        log_var=paddle.full_like(z[:,self.cae_hp.v_z_c:],np.log(self.cae_hp.var),'float32')
        
        gauss_sample=ReparamNormal(mean.shape).sample(mean,log_var)
        return gauss_sample

    def decode(self,z):
        recon_img=self.decoder(z)
        return recon_img
    
    def ae(self,img):
        img=paddle.to_tensor(img)
        img=paddle.transpose(img,perm=[0,3,1,2])
        z=self.encoder(img)
        recon_img=self.decoder(z)
        return recon_img
    def train_with_pred(self,img_seq):

        # print(f"img_seq:shape={img_seq.shape},type={type(img_seq)}")
        _b,_t,_c,_h,_w=img_seq.shape
        img=paddle.cast(paddle.to_tensor(img_seq,'uint8'),'float32')/255
        img=paddle.reshape(img,[_b*_t,_c,_h,_w])
        

        z=self.encoder(img)
        mean=z[:,:self.cae_hp.v_z_c]
        log_var=paddle.full_like(z[:,self.cae_hp.v_z_c:],np.log(self.cae_hp.var),'float32')
        gauss_sample=ReparamNormal(mean.shape).sample(mean,log_var)
        recon_img=self.decode(gauss_sample)

        loss_recon=0.5*paddle.mean((img-recon_img)**2*self.weight_mask)
        loss_kl=paddle.mean(paddle.mean(mean**2,axis=[1,2,3],keepdim=True)**0.5)
        loss=loss_recon+loss_kl*self.cae_hp.kl_loss_ratio

        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()

        self.avg_loss_kl.update(np.mean(loss_kl.numpy()))
        self.avg_loss_recon.update(np.mean(loss_recon.numpy()))

        return paddle.reshape(mean,[_b,-1]).numpy(),recon_img.detach()


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
