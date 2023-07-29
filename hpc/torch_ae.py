import gym
import math
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image



class replay_buffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def push(self, x):
        self.buffer.append(x)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
    def get_random_data(self):
        index=np.random.randint(0,len(self.buffer))
        # print("index=",index)
        rand_data=self.buffer[index]
        return rand_data

class dense_block(nn.Module):
    def __init__(self,fc_num_list,act=nn.LeakyReLU(negative_slope=0.1),act_output=None):
        super(dense_block,self).__init__()
        self.fc_num_list=fc_num_list
        #get dense_layer_list
        dense_layer_list=[]
        for i in range(1, len(fc_num_list)):
            linear_part=nn.Linear(fc_num_list[i-1],fc_num_list[i])
            act_part=act_output if i==len(fc_num_list)-1 else act
            layer=linear_part if act_part==None else nn.Sequential(linear_part,act_part)
            dense_layer_list.append(layer)
        #get block
        self.block=nn.Sequential(*dense_layer_list)
        # print(self.block)
    def forward(self,x):
        # print(self.fc_num_list)
        # print(x.size())
        return self.block(x)

class hippocampus_sparse_autoencoder():
    def __init__(self,data_dim,
                 encoder_fc_num_list,decoder_fc_num_list,
                 sparse_dim=1000,sparsity=0.05,
                 lr = 0.0001,gamma_fire=0.9999,gamma_fire_fast=0.99
                 ):
        "hyper parameter"
        self.batch_size=64
        self.data_dim=data_dim
        self.encoder_fc_num_list = encoder_fc_num_list
        self.decoder_fc_num_list = decoder_fc_num_list
        self.sparse_dim=sparse_dim
        self.sparsity=sparsity
        self.k=int(sparse_dim*sparsity)
        self.lr=lr
        self.gamma_fire=gamma_fire
        self.gamma_fire_fast=gamma_fire_fast

        "buffer"
        self.buffer=replay_buffer(max_size=50000)

        "variable"
        self.hid_neuron_avg_fire = torch.from_numpy(np.ones([self.sparse_dim]) * self.sparsity)
        self.hid_neuron_avg_fire = Variable(self.hid_neuron_avg_fire)
        self.hid_neuron_avg_fire.requires_grad = False
        # self.hid_neuron_avg_fire=to_cuda(self.hid_neuron_avg_fire)

        self.hid_neuron_avg_fire_fast = torch.from_numpy(np.ones([self.sparse_dim]) * self.sparsity)
        self.hid_neuron_avg_fire_fast = Variable(self.hid_neuron_avg_fire_fast)
        self.hid_neuron_avg_fire_fast.requires_grad = False
        # self.hid_neuron_avg_fire_fast=to_cuda(self.hid_neuron_avg_fire_fast)

        "net"
        self.encoder=dense_block(encoder_fc_num_list,act_output=nn.Sigmoid())

        # self.decoder=dense_block(decoder_fc_num_list,act_output=None)
        self.decoder=nn.Linear(decoder_fc_num_list[0],decoder_fc_num_list[1],bias=False)
        self.decoder.weight.data.uniform_(-1/5000,1/5000)
        "decoder weight"
        self.decoder_weight = torch.transpose(list(self.decoder.parameters())[0], 1, 0)
        "cuda"
        # self.encoder=to_cuda(self.encoder)
        # self.encoder_old_version = to_cuda(self.encoder_old_version)
        # self.decoder = to_cuda(self.decoder)
        "optimizer"
        self.optimizer=optim.RMSprop(
            [*list(self.encoder.parameters()),*list(self.decoder.parameters())],
            lr=lr)
        "state"
        self.get_test_img=True
        self.test_img=None
    def get_avg_fire_numpy(self):
        return self.hid_neuron_avg_fire.data.cpu().numpy()
    def imshow_img(self,img,name):
        if type(img)==torch.Tensor:
            img=img.data.cpu().numpy()
        print(type(img))
        img=cv2.resize(np.reshape(img,[28,28]),(400,400),interpolation=cv2.INTER_NEAREST)
        cv2.imshow(name,img)
    def get_large_img(self,img_list,width_img_num=6):
        single_img_size=[280,280]
        small_img_list=[]
        for img in img_list:
            small_img=np.reshape(img,[28,28])
            small_img_list.append(small_img)

        line_img_list=[]
        for i in range(len(small_img_list)//width_img_num):
           line_img_list.append(np.concatenate(small_img_list[i*width_img_num:(i+1)*width_img_num],axis=1))
        if len(small_img_list)%width_img_num==0:
            pass
        else:
            append_num=len(small_img_list)%width_img_num
            last_line=small_img_list[-(len(small_img_list)%width_img_num):]
            for k in range(width_img_num-len(last_line)):
                last_line.append(np.zeros([28,28],dtype=np.float32))
            line_img_list.append(np.concatenate(last_line,axis=1))
            # print([np.shape(li) for li in last_line])
        # print([np.shape(li) for li in line_img_list])
        large_img=np.concatenate(line_img_list,axis=0)
        return large_img
    def get_vec_from_action(self,action):
        vec=self.decoder_weight[action].data.numpy()
        return vec
    def get_hid_code(self,data):
        if type(data)==np.ndarray or type(data)==list:
            # print(type(data))
            data_torch = torch.from_numpy(np.reshape(data,[-1,self.data_dim]).astype(np.float32))
        else:
            # print(type(data))
            data_torch=data
        # data_torch=to_cuda(data_torch)
        "forward"
        hid_code = self.encoder(data_torch)
        return hid_code.data.cpu().numpy()


    def save_model(self,count,dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(self.encoder.state_dict(),dir+"/encoder_{}".format(count))
        torch.save(self.decoder.state_dict(),dir+"/decoder_{}".format(count))
        print("save model {} success!".format(count))
    def load_model(self,count,dir):
        self.encoder.load_state_dict(torch.load(dir+"./encoder_{}".format(count),map_location=torch.device('cpu')))
        print(list(self.decoder.state_dict()))
        print(list(torch.load(dir+"./decoder_{}".format(count))))
        self.decoder.load_state_dict(torch.load(dir+"./decoder_{}".format(count),map_location=torch.device('cpu')))

        print("load model {} success!".format(count))
    def get_output_from_hidcode(self,hid_code):
        hid_code_torch=torch.from_numpy(hid_code)
        hid_code_norm=self.norm_hid(hid_code_torch)
        output=self.decoder(hid_code_norm)
        return output.data.numpy()


    def norm_hid(self,hid_code):
        # return self.k*hid_code/torch.sum(hid_code,dim=-1,keepdim=True)
        hid2=hid_code**2
        return self.k*hid2/torch.sum(hid2,dim=-1,keepdim=True)
        
    def show_hid(self,hid,name):
        data=np.reshape(hid,[40,25])
        img=cv2.resize(data,(250,400),interpolation=cv2.INTER_NEAREST)
        cv2.imshow(name,img)
    def train_one_step(self):
        if len(self.buffer.buffer)>self.batch_size:
            # print(self.buffer)
            data=random.sample(self.buffer.buffer,self.batch_size)
        else:
            return 0,0
        "test_img"
        # if self.get_test_img:
        #     self.test_img=data[:1].data.cpu().numpy()
        #     self.get_test_img=False

        self.optimizer.zero_grad()
        if type(data)==np.array or type(data)==list:
            data_torch=torch.from_numpy(np.reshape(data,[-1,self.data_dim]).astype(np.float32))
        else:
            data_torch=data
        # print(type(data_torch))
        ##################################################################
        # print("data_torch",data_torch)
        "forward"
        hid_code=self.encoder(data_torch)
        self.train_hid=hid_code[:1].data.cpu().numpy()

        hid_code_norm=self.norm_hid(hid_code)
        self.train_hid_norm=hid_code_norm[:1].data.cpu().numpy()
        output=self.decoder(hid_code_norm)

        "update hid avg fire"
        mean_fire_hid_code = torch.mean(hid_code, dim=0).detach()
        self.train_mean_fire =mean_fire_hid_code.data.cpu().numpy()
        self.hid_neuron_avg_fire.mul_(self.gamma_fire)
        self.hid_neuron_avg_fire.add_((1 - self.gamma_fire) * mean_fire_hid_code)
        self.hid_neuron_avg_fire_fast.mul_(self.gamma_fire_fast)
        self.hid_neuron_avg_fire_fast.add_((1 - self.gamma_fire_fast) * mean_fire_hid_code)
        ##############################################################
        "get loss"
        "loss_restore"
        loss_restore = torch.mean((output - data_torch) ** 2)
        loss_restore *= 1000


        "loss_top_k"
        loss_top_k= torch.mean(torch.abs(torch.sum(hid_code ** 2, dim=-1, keepdim=True) - self.k) +
                               torch.abs(torch.sum((1 - hid_code) ** 2, dim=-1, keepdim=True) - (self.sparse_dim-self.k)))
        # loss_top_k=torch.mean((1-self.sparsity)*torch.abs(torch.sum(hid_code ** 2, dim=-1, keepdim=True) - self.k) +
        #                        self.sparsity*torch.abs(torch.sum((1 - hid_code) ** 2, dim=-1, keepdim=True) - (self.sparse_dim-self.k)))
        loss_top_k *= 0.01
        "loss_KL"

        loss_KL_slow = torch.mean((-self.sparsity / (self.hid_neuron_avg_fire + 1e-8) + (1 - self.sparsity) / (
                    1 - self.hid_neuron_avg_fire + 1e-8)) * hid_code)
        loss_KL_fast=torch.mean((-self.sparsity / (self.hid_neuron_avg_fire_fast + 1e-8) + (1 - self.sparsity) / (
                    1 - self.hid_neuron_avg_fire_fast + 1e-8)) * hid_code)
        loss_KL=0.9*loss_KL_slow+0.1*loss_KL_fast
        loss_KL *= 10
        "all loss"
        loss = loss_top_k + loss_restore + loss_KL
        loss_list=[float(loss_restore.data.cpu().numpy()),
                   float(loss_top_k.data.cpu().numpy()),
                   float(loss_KL.data.cpu().numpy())]
        ############################################################
        "backward"
        loss.backward()
        self.optimizer.step()
        return loss_list,output














