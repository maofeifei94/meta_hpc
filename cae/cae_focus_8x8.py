import os
import numpy as np
import paddle.nn as nn
import paddle
import paddle.optimizer as optim
import paddle.nn.functional as F
from paddle.jit import to_static
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
class Conv_Encoder_net_8x8(nn.Layer):
    def __init__(self,img_shape,cae_hp):
        super(Conv_Encoder_net_8x8,self).__init__()
        hidc=cae_hp.v_encoder_hid_c_8x8
        layer_list=[]
        layer_list.append(ResBlock(
            input_shape=[3,8,8],
            shortcut_layer=nn.Conv2D(in_channels=3,out_channels=hidc, kernel_size=[3,3],stride=[1,1],padding='SAME',weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))),
            first_conv_layer=nn.Conv2D(in_channels=3,out_channels=hidc, kernel_size=[3,3],stride=[1,1],padding='SAME',weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))),
            output_shape=[hidc,8,8],
            ))
        for i in range(3):
            layer_list.append(ResBlock(
                input_shape=[hidc,8,8],
                shortcut_layer=EqualLayer(),
                first_conv_layer=nn.Conv2D(in_channels=hidc,out_channels=hidc, kernel_size=[3,3],stride=[1,1],padding='SAME',weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))),
                output_shape=[hidc,8,8],
                ))
        layer_list.append(ResBlock(
            input_shape=[hidc,8,8],
            shortcut_layer=nn.Conv2D(in_channels=hidc,out_channels=cae_hp.v_z_c_8x8, kernel_size=[3,3],stride=[1,1],padding='SAME',weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))),
            first_conv_layer=nn.Conv2D(in_channels=hidc,out_channels=cae_hp.v_z_c_8x8, kernel_size=[3,3],stride=[1,1],padding='SAME',weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))),
            output_shape=[cae_hp.v_z_c_8x8,8,8],
            ))
        self.conv_net=nn.Sequential(*layer_list)
    def forward(self,img):
        return self.conv_net(img)
class Conv_Decoder_net_8x8(nn.Layer):
    def __init__(self,cae_hp):
        super(Conv_Decoder_net_8x8,self).__init__()
        hidc=cae_hp.v_decoder_hid_c_8x8
        layer_list=[]
        layer_list.append(ResBlock(
            input_shape=[cae_hp.v_z_c_8x8,8,8],
            shortcut_layer=nn.Conv2D(in_channels=cae_hp.v_z_c_8x8,out_channels=hidc, kernel_size=[3,3],stride=[1,1],padding='SAME',weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))),
            first_conv_layer=nn.Conv2D(in_channels=cae_hp.v_z_c_8x8,out_channels=hidc, kernel_size=[3,3],stride=[1,1],padding='SAME',weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))),
            output_shape=[hidc,8,8],
            ))
        for i in range(4):
            layer_list.append(ResBlock(
                input_shape=[hidc,8,8],
                shortcut_layer=EqualLayer(),
                first_conv_layer=nn.Conv2D(in_channels=hidc,out_channels=hidc, kernel_size=[3,3],stride=[1,1],padding='SAME',weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))),
                output_shape=[hidc,8,8],
                ))
        layer_list.append(nn.Conv2D(in_channels=hidc,out_channels=3, kernel_size=[3,3],stride=[1,1],padding='SAME',weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))))
        layer_list.append(nn.Sigmoid())

        self.conv_net=nn.Sequential(*layer_list)
    def forward(self,img):
        return self.conv_net(img)

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
                shortcut_layer=nn.Conv2D(in_channels=inc,out_channels=outc, kernel_size=[5,5],stride=[2,2],padding='SAME',weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))),
                first_conv_layer=nn.Conv2D(in_channels=inc,out_channels=outc, kernel_size=[5,5],stride=[2,2],padding='SAME',weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))),
                output_shape=[outc,img_width//2,img_width//2],
                ))
            img_width=img_width//2
            for j in range(1):
                layer_list.append(ResBlock([outc,img_width,img_width]))
        "normal conv"
        for i in range(2):
            layer_list.append(ResBlock([outc,img_width,img_width]))
        "out conv"
        layer_list.append(ResBlock(
            input_shape=[outc,img_width,img_width],
            shortcut_layer=nn.Conv2D(in_channels=outc,out_channels=self.cae_hp.v_z_c, kernel_size=[3,3],stride=[1,1],padding='SAME',weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))),
            first_conv_layer=nn.Conv2D(in_channels=outc,out_channels=self.cae_hp.v_z_c, kernel_size=[3,3],stride=[1,1],padding='SAME',weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))),
            output_shape=[self.cae_hp.v_z_c,img_width,img_width],
            ))
        for i in range(2):
            layer_list.append(ResBlock([self.cae_hp.v_z_c,img_width,img_width]))
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
            shortcut_layer=nn.Conv2D(in_channels=self.cae_hp.v_z_c,out_channels=hid_c*2**3, kernel_size=[3,3],stride=[1,1],padding='SAME',weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))),
            first_conv_layer=nn.Conv2D(in_channels=self.cae_hp.v_z_c,out_channels=hid_c*2**3, kernel_size=[3,3],stride=[1,1],padding='SAME',weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))),
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
                shortcut_layer=nn.Conv2DTranspose(in_channels=inc,out_channels=outc,kernel_size=[5,5],stride=2,padding="SAME",weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))),
                first_conv_layer=nn.Conv2DTranspose(in_channels=inc,out_channels=outc,kernel_size=[5,5],stride=2,padding="SAME",weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))),
                output_shape=[outc,img_width*2,img_width*2],
            ))
            img_width*=2
            for j in range(1):
                layer_list.append(ResBlock([outc,img_width,img_width]))
        "normal conv"
        for i in range(3):
            layer_list.append(ResBlock([outc,img_width,img_width]))
        "out conv"
        layer_list.append(nn.Conv2D(in_channels=outc,out_channels=3, kernel_size=[3,3],stride=[1,1],padding='SAME',weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0,std=0.01))))
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

        self.encoder_8x8=Conv_Encoder_net_8x8([3,8,8],cae_hp)
        self.decoder_8x8=Conv_Decoder_net_8x8(cae_hp)
        
        self.optimizer=optim.Adam(
            learning_rate=self.cae_hp.lr,
            parameters=[*self.encoder.parameters(),*self.decoder.parameters(),*self.encoder_8x8.parameters(),*self.decoder_8x8.parameters()]
            )
        self.avg_loss_recon=moving_avg(gamma=0.99)
        self.avg_loss_kl=moving_avg(gamma=0.99)

    def get_8x8_img(self,img):
        return img[:,:,28:36,28:36]
    def encode_static(self,img):
        with paddle.no_grad():
            z=self.encoder(img)
            z_8x8=self.encoder_8x8(self.get_8x8_img(img))
            return paddle.concat([z,z_8x8],axis=1)
    def encode(self,img):
        z=self.encoder(img)
        z_8x8=self.encoder_8x8(self.get_8x8_img(img))
        return paddle.concat([z,z_8x8],axis=1)
    def decode(self,z):
        recon_img=self.decoder(z)
        return recon_img
    def decode_8x8(self,z_8x8):
        recon_img_8x8=self.decoder_8x8(z_8x8)
        return recon_img_8x8
    
    def train_with_pred(self,img_seq):

        # print(f"img_seq:shape={img_seq.shape},type={type(img_seq)}")
        _b,_t,_c,_h,_w=img_seq.shape
        img=paddle.cast(paddle.to_tensor(img_seq,'uint8'),'float32')/255
        img=paddle.reshape(img,[_b*_t,_c,_h,_w])
        img_8x8=self.get_8x8_img(img)
        

        z=self.encoder(img)
        z_8x8=self.encoder_8x8(img_8x8)

        mean=z
        mean_8x8=z_8x8

        log_var=paddle.full_like(mean,np.log(self.cae_hp.var),'float32')
        log_var_8x8=paddle.full_like(mean_8x8,np.log(self.cae_hp.var),'float32')

        gauss_sample=ReparamNormal(mean.shape).sample(mean,log_var)
        gauss_sample_8x8=ReparamNormal(mean_8x8.shape).sample(mean_8x8,log_var_8x8)

        recon_img=self.decode(gauss_sample)
        recon_img_8x8=self.decode_8x8(gauss_sample_8x8)

        loss_recon=0.5*paddle.mean((img-recon_img)**2)+0.5*paddle.mean((img_8x8-recon_img_8x8)**2)
        loss_kl=paddle.mean(paddle.mean(mean**2,axis=[1,2,3],keepdim=True)**0.5)+paddle.mean(paddle.mean(mean_8x8**2,axis=[1,2,3],keepdim=True)**0.5)

        loss=(loss_recon)+loss_kl*self.cae_hp.kl_loss_ratio

        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()

        self.avg_loss_kl.update(np.mean(loss_kl.numpy()))
        self.avg_loss_recon.update(np.mean(loss_recon.numpy()))

        out_img=recon_img.detach()
        out_img[:,:,28:36,28:36]=recon_img_8x8.detach()
        return paddle.reshape(paddle.concat([z,z_8x8],axis=1),[_b,_t,-1]).numpy(),out_img
    def pred(self,img_seq):
        with paddle.no_grad():
            _b,_t,_c,_h,_w=img_seq.shape
            img=paddle.cast(paddle.to_tensor(img_seq,'uint8'),'float32')/255
            img=paddle.reshape(img,[_b*_t,_c,_h,_w])
            img_8x8=self.get_8x8_img(img)
            
            z=self.encoder(img)
            z_8x8=self.encoder_8x8(img_8x8)
            return paddle.reshape(paddle.concat([z,z_8x8],axis=1),[_b,_t,-1]).numpy()


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
