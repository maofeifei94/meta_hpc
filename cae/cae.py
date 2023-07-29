import os
import numpy as np
import paddle.nn as nn
import paddle
import paddle.optimizer as optim
import paddle.nn.functional as F

from mlutils.ml import ReparamNormal,moving_avg,ModelSaver

c_ratio=2

nl_func_dict={
    'Tanh':nn.Tanh(),
    "LeakyRelu":nn.LeakyReLU(0.01),
    "Selu":nn.SELU(),
    "Swish":nn.Swish(),
    "Gelu":nn.GELU()
    }


class Reslayer(nn.Layer):
    def __init__(self,layer,func):
        super(Reslayer,self).__init__()
        self.layer=layer
        self.func=func
    def forward(self,x):
        return x+self.func(self.layer(x))
class EqualLayer(nn.Layer):
    def forward(self,x):
        return x
class ResBlock(nn.Layer):
    def __init__(self,input_shape,shortcut_layer=None,first_conv_layer=None,output_shape=None):
        super().__init__()
        "eg:input_shape=[3,64,64]"
        self.act=nn.GELU()
        if shortcut_layer is None:
            self.layer_norm1=nn.LayerNorm(input_shape[-2:],weight_attr=False,bias_attr=False)
            self.conv1=nn.Conv2D(in_channels=input_shape[0],out_channels=input_shape[0],kernel_size=[3,3],stride=1,padding="SAME",weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01)))
            self.layer_norm2=nn.LayerNorm(input_shape[-2:],weight_attr=False,bias_attr=False)
            self.conv2=nn.Conv2D(in_channels=input_shape[0],out_channels=input_shape[0],kernel_size=[1,1],stride=1,padding="SAME",weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01)))
            self.seq_layer=nn.Sequential(
                self.layer_norm1,
                self.act,
                self.conv1,
                self.layer_norm2,
                self.act,
                self.conv2,
            )
            self.shortcut_layer=EqualLayer()
        else:
            self.layer_norm1=nn.LayerNorm(input_shape[-2:],weight_attr=False,bias_attr=False)
            self.conv1=first_conv_layer
            self.layer_norm2=nn.LayerNorm(output_shape[-2:],weight_attr=False,bias_attr=False)
            self.conv2=nn.Conv2D(in_channels=output_shape[0],out_channels=output_shape[0],kernel_size=[1,1],stride=1,padding="SAME",weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01)))
            self.seq_layer=nn.Sequential(
                self.layer_norm1,
                self.act,
                self.conv1,
                self.layer_norm2,
                self.act,
                self.conv2,
            )
            self.shortcut_layer=shortcut_layer

    def forward(self, x):
        return self.shortcut_layer(x)+self.seq_layer(x)
"""
卷积自编码器，输入为图片，输出为降维后的向量
"""
class Conv_Encoder_net(nn.Layer):
    def __init__(self,img_shape,cae_hp):
        super(Conv_Encoder_net,self).__init__()
        self.cae_hp=cae_hp
        hid_c=self.cae_hp.v_encoder_hid_c

        layer_list=[]
        img_width=64
        "downsample"
        for i in range(3):
            inc=img_shape[0] if i==0 else hid_c*2**i
            outc=hid_c*2**(i+1)
            layer_list.append(ResBlock(
                input_shape=[inc,img_width,img_width],
                shortcut_layer=nn.Conv2D(in_channels=inc,out_channels=outc, kernel_size=[5,5],stride=[2,2],padding='SAME'),
                first_conv_layer=nn.Conv2D(in_channels=inc,out_channels=outc, kernel_size=[5,5],stride=[2,2],padding='SAME'),
                output_shape=[outc,img_width//2,img_width//2],
                ))
            img_width=img_width//2
            for j in range(3):
                layer_list.append(ResBlock([outc,img_width,img_width]))
        "normal conv"
        for i in range(3):
            layer_list.append(ResBlock([outc,img_width,img_width]))
        "out conv"
        layer_list.append(ResBlock(
            input_shape=[outc,img_width,img_width],
            shortcut_layer=nn.Conv2D(in_channels=outc,out_channels=self.cae_hp.v_z_c*2, kernel_size=[3,3],stride=[1,1],padding='SAME'),
            first_conv_layer=nn.Conv2D(in_channels=outc,out_channels=self.cae_hp.v_z_c*2, kernel_size=[3,3],stride=[1,1],padding='SAME'),
            output_shape=[self.cae_hp.v_z_c*2,img_width,img_width],
            ))
        for i in range(3):
            layer_list.append(ResBlock([self.cae_hp.v_z_c*2,img_width,img_width]))
        self.conv_net=nn.Sequential(*layer_list)
    def forward(self,img):
        return self.conv_net(img)

class Conv_Decoder_net(nn.Layer):
    def __init__(self,cae_hp):
        super(Conv_Decoder_net,self).__init__()
        self.cae_hp=cae_hp
        hid_c=self.cae_hp.v_decoder_hid_c

        layer_list=[]
        img_width=8
        "channel conv"
        layer_list.append(ResBlock(
            input_shape=[self.cae_hp.v_z_c,img_width,img_width],
            shortcut_layer=nn.Conv2D(in_channels=self.cae_hp.v_z_c,out_channels=hid_c*2**3, kernel_size=[3,3],stride=[1,1],padding='SAME'),
            first_conv_layer=nn.Conv2D(in_channels=self.cae_hp.v_z_c,out_channels=hid_c*2**3, kernel_size=[3,3],stride=[1,1],padding='SAME'),
            output_shape=[hid_c*2**3,img_width,img_width],
        ))
        "normal_conv before upsample"
        for i in range(2):
            inc=hid_c*2**3
            outc=hid_c*2**3
            layer_list.append(ResBlock([inc,img_width,img_width]))
        "upsample"
        for i in range(3):
            inc=hid_c*2**(3-i)
            outc=hid_c*2**(2-i)
            layer_list.append(ResBlock(
                input_shape=[inc,img_width,img_width],
                shortcut_layer=nn.Conv2DTranspose(in_channels=inc,out_channels=outc,kernel_size=[5,5],stride=2,padding="SAME"),
                first_conv_layer=nn.Conv2DTranspose(in_channels=inc,out_channels=outc,kernel_size=[5,5],stride=2,padding="SAME"),
                output_shape=[outc,img_width*2,img_width*2],
            ))
            img_width*=2
            for j in range(3):
                layer_list.append(ResBlock([outc,img_width,img_width]))
        "normal conv"
        for i in range(3):
            layer_list.append(ResBlock([outc,img_width,img_width]))
        "out conv"
        layer_list.append(nn.Conv2D(in_channels=outc,out_channels=3, kernel_size=[3,3],stride=[1,1],padding='SAME'))
        layer_list.append(nn.Sigmoid())

        self.conv_net=nn.Sequential(*layer_list)

    def forward(self,img):
        return self.conv_net(img)


class Conv_Auto_Encoder(nn.Layer,ModelSaver):
    def __init__(self,cae_hp) -> None:
        super(Conv_Auto_Encoder,self).__init__()
        self.cae_hp=cae_hp

        self.encoder=Conv_Encoder_net([3,64,64],cae_hp)
        self.decoder=Conv_Decoder_net(cae_hp)
        self.optimizer=optim.Adam(
            learning_rate=self.cae_hp.lr,
            parameters=[*self.encoder.parameters(),*self.decoder.parameters()]
            )
        self.avg_loss_recon=moving_avg(gamma=0.99)
        self.avg_loss_kl=moving_avg(gamma=0.99)

        self.weight_mask=self.get_weight_mask()
        # self.weight_mask=self.get_gauss_weight_mask()
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
