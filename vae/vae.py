import os
import myf_ML_util
from myf_ML_util import Basic_Module, ReplayMemory,dense_block,DataFormat
import numpy as np
import paddle.nn as nn
import paddle
import paddle.optimizer as optim
import paddle.nn.functional as F

v_encoder_hid_c=8
v_z_c=4
v_decoder_hic_c=8

def nl_func():
    return nn.LeakyReLU(negative_slope=0.1)
class Reslayer(nn.Layer):
    def __init__(self,layer,func):
        super(Reslayer,self).__init__()
        self.layer=layer
        self.func=func
    def forward(self,x):
        return x+self.func(self.layer(x))
class V_Conv_Encoder_net(nn.Layer):
    def __init__(self,img_shape):
        super(V_Conv_Encoder_net,self).__init__()

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
            Reslayer(
                nn.Conv2D(
                    in_channels=hid_c,out_channels=hid_c,
                    kernel_size=[3,3],stride=[1,1],padding='SAME'),nl_func()
            ),
        )
        self.mean=nn.Conv2D(
            in_channels=hid_c,out_channels=v_z_c,
            kernel_size=[3,3],stride=[1,1],padding='SAME')
        self.log_var=nn.Conv2D(
            in_channels=hid_c,out_channels=v_z_c,
            kernel_size=[3,3],stride=[1,1],padding='SAME')
    def forward(self,img):
        x=img
        x=self.conv_net(x)
        mean=self.mean(x)
        
        ######################
        # log_var=self.log_var(x)
        log_var=paddle.zeros_like(mean)-10
        #######################
        return mean,log_var

class V_Conv_Decoder_net(nn.Layer):
    def __init__(self):
        super(V_Conv_Decoder_net,self).__init__()
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
                kernel_size=[3,3],stride=2,padding="SAME"),nl_func(),
            nn.Conv2DTranspose(
                in_channels=hid_c,out_channels=hid_c,
                kernel_size=[3,3],stride=2,padding="SAME"),nl_func(),
            nn.Conv2DTranspose(
                in_channels=hid_c,out_channels=hid_c,
                kernel_size=[3,3],stride=1,padding="SAME"),nl_func(),
            nn.Conv2DTranspose(
                in_channels=hid_c,out_channels=3,
                kernel_size=[3,3],stride=1,padding="SAME"),nn.Sigmoid(),
            )
    def forward(self,img):
        return self.deconv_net(img)
class V_Auto_Encoder(nn.Layer):
    def __init__(self) -> None:
        super(V_Auto_Encoder,self).__init__()
        self.train_batch_size=128

        self.encoder=V_Conv_Encoder_net([3,64,64])
        self.decoder=V_Conv_Decoder_net()
        self.optimizer=optim.Adam(
            learning_rate=0.0003,
            parameters=[*self.encoder.parameters(),*self.decoder.parameters()]
            )

        self.rpm=ReplayMemory([DataFormat("img",[64,64,3],np.float32)],50000)
        self.avg_loss_recon=myf_ML_util.moving_avg()
        self.avg_loss_kl=myf_ML_util.moving_avg()
    def reparameterize(self, mu, log_var):
        std = paddle.exp(0.5*log_var)
        eps = paddle.randn(shape=std.shape)
        # print(mu,std,paddle.sum(eps**2))
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
        mean,log_var=self.encoder(img)
        z=self.reparameterize(mean,log_var)
        return z
    def decode(self,z):
        recon_img=self.decoder(z)
        return recon_img
    
    def ae(self,img):
        img=paddle.to_tensor(img)
        img=paddle.transpose(img,perm=[0,3,1,2])
        mean,log_var=self.encoder(img)
        z=self.reparameterize(mean,log_var)
        recon_img=self.decoder(z)
        return recon_img


    def learn(self):
        "更新网络"
        train_data=self.rpm.sample_batch(self.train_batch_size)
        img=paddle.to_tensor(train_data['img'])
        img=paddle.transpose(img,perm=[0,3,1,2])

        mean,log_var=self.encoder(img)
        z=self.reparameterize(mean,log_var)
        
        recon_img=self.decoder(z)


        loss_recon=0.5*paddle.mean((img-recon_img)**2)
        # loss_recon=paddle.mean(paddle.abs(img-recon_img))
        loss_kl=paddle.mean(paddle.mean(mean**2,axis=[1,2,3],keepdim=True)**0.5)

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
        paddle.save(self.state_dict(),f"{save_dir}/{iter_num}_model.pdparams")
        print(f"save model {save_dir}/{iter_num}_model.pdparams success")
        paddle.save(self.state_dict(),f"{save_dir}/newest_model.pdparams")

    def load_model(self,save_dir,iter_num):
        print(f"start load_model {save_dir}/{iter_num}_model.pdparams")
        self.set_state_dict(paddle.load(path=f"{save_dir}/{iter_num}_model.pdparams"))
        print(f"load model {save_dir}/{iter_num}_model.pdparams success")
