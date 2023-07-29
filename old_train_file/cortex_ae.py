from re import X
from turtle import forward
import myf_ML_util
from myf_ML_util import Basic_Module,dense_block
import paddle.nn as nn
import paddle
import paddle.optimizer as optim
import paddle.nn.functional as F

def nl_func(x):
    return F.leaky_relu(x,negative_slope=0.1)
class Conv_Encoder_net(nn.Layer):
    def __init__(self,img_shape):
        super(Conv_Encoder_net,self).__init__()
        self.conv1=nn.Conv2D(
            in_channels=img_shape[0],out_channels=4,
            kernel_size=[3,3],stride=[2,2],padding='SAME')
        self.conv2=nn.Conv2D(
            in_channels=4,out_channels=4,
            kernel_size=[3,3],stride=[2,2],padding='SAME')
        self.conv3=nn.Conv2D(
            in_channels=4,out_channels=1,
            kernel_size=[3,3],stride=[2,2],padding='SAME')
        self.l4=nn.Linear(64,64)
    def forward(self,img):
        x=img
        x=nl_func(self.conv1(x))
        x=nl_func(self.conv2(x))
        x=nl_func(self.conv3(x))
        x=nn.Sigmoid(self.l4(x))
        return x
class Conv_Decoder_net(nn.Layer):
    def __init__(self):
        super(Conv_Decoder_net,self).__init__()
        self.deconv1=nn.Conv2DTranspose(
            in_channels=1,out_channels=4,
            kernel_size=[3,3],stride=2,padding="SAME")
        self.deconv2=nn.Conv2DTranspose(
            in_channels=4,out_channels=4,
            kernel_size=[3,3],stride=2,padding="SAME")
        self.deconv3=nn.Conv2DTranspose(
            in_channels=4,out_channels=4,
            kernel_size=[3,3],stride=1,padding="SAME")
        self.deconv4=nn.Conv2DTranspose(
            in_channels=4,out_channels=4,
            kernel_size=[3,3],stride=2,padding="SAME") 
        self.deconv5=nn.Conv2DTranspose(
            in_channels=4,out_channels=3,
            kernel_size=[3,3],stride=1,padding="SAME") 
    def forward(self,img):
        x=img
        x=nl_func(self.deconv1(x))
        x=nl_func(self.deconv2(x))
        x=nl_func(self.deconv3(x))
        x=nl_func(self.deconv4(x))
        x=F.sigmoid(self.deconv5(x))
        return x
class Auto_Encoder(Basic_Module):
    def __init__(self) -> None:
        super(Auto_Encoder,self).__init__()
        self.encoder=Conv_Encoder_net([3,64,64])
        self.decoder=Conv_Decoder_net()
        self.optimizer=optim.Adam(
            learning_rate=0.0003,
            parameters=[*self.encoder.parameters(),*self.decoder.parameters()]
            )
    def reset(self):
        pass
    def input(self):
        "接受外界输入，返回处理结果"
        pass
    def rpm_collect(self):
        "收集训练数据"
        pass
    def rpm_clear(self):
        "清空rpm的训练数据"
        pass
    def learn(self):
        "更新网络"
        pass
    def save_model(self):
        pass
    def load_model(self):
        pass