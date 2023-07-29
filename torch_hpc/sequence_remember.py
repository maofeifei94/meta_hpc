
import os
import gym
import math
import random
import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from util import hippocampus_sparse_autoencoder,dense_block,mnist_reader
from show import *




show_img_window=False
use_cuda=True


def to_cuda(x,use_cuda):
    if use_cuda:
        return x.cuda()
    else:
        return x
def to_np(x):
    return x.data.cpu().numpy()

def save_model(count,net,dir,name):
    torch.save(net.state_dict(),dir+"/"+name+"_{}".format(count))
def load_model(count,net,dir,name):
    # if name=='hippo_decoder':
    #     load_dict = torch.load(dir + "/" + name + "_{}".format(count))
    #     # load_dict['weight']
    #     print(load_dict)

    # print(dir+"/"+name+"_{}".format(count))
    # load_dict=torch.load(dir+"/"+name+"_{}".format(count))
    # print([n for n in list(load_dict)])
    # print(list(net.state_dict()))
    net.load_state_dict(torch.load(dir+"/"+name+"_{}".format(count)))#, map_location=torch.device('cpu')))
def show_tmat(tmat,winname,shape=[40,25]):
    cpu_mat=tmat.data.cpu().numpy()
    cpu_mat/=np.max(cpu_mat)
    cv2.imshow(winname,cv2.resize(np.reshape(cpu_mat,shape),None,fx=1,fy=1,interpolation=cv2.INTER_AREA))


def memory_sequence(seq_engram):
    seq_engram=torch.from_numpy(seq_engram.astype(np.float32)).cuda()
    # memory_mat=np.zeros([1000,1000])
    memory_mat=torch.zeros([1000,1000]).cuda()
    thred=0.9
    "memory"
    # seq_engram_binary=np.where(seq_engram>thred,1.0,0.0)
    seq_engram_binary=torch.where(seq_engram>thred,torch.ones_like(seq_engram),torch.zeros_like(seq_engram))

    pre_engram=seq_engram_binary[0]
    run_num=1000
    # seq_engram.shape[0]
    for i in range(1,run_num):
        print(i)
        now_engram=seq_engram_binary[i]

        # LTP_mat=np.matmul(np.reshape(pre_engram,[1000,1]),np.reshape(now_engram,[1,1000]))
        LTP_mat=torch.matmul(torch.reshape(pre_engram,[1000,1]),torch.reshape(now_engram,[1,1000]))
        # memory_mat=np.minimum(np.ones_like(memory_mat),memory_mat+LTP_mat)

        memory_mat=torch.clamp_max(memory_mat+LTP_mat,1.0)

        pre_engram=now_engram

    # cv2.imshow("memory_mat",memory_mat.data.cpu().numpy())
    # cv2.waitKey()

    "recall"
    recall_pre_engram=seq_engram_binary[0]

    for i in range(1,run_num):
        label_engram = seq_engram_binary[i]


        recall_pre_num=torch.sum(recall_pre_engram)

        recall_memory=torch.matmul(torch.reshape(recall_pre_engram,[1,1000]),memory_mat)[0]
        # print(recall_memory)
        print(recall_pre_num,torch.sum(recall_memory>=(recall_pre_num-0.5)))
        recall_now_engram=torch.where(recall_memory>=recall_pre_num-1,torch.ones_like(recall_memory),torch.zeros_like(recall_memory))

        plt.hist(recall_memory.data.cpu().numpy(),bins=100)
        plt.show()
        show_tmat(recall_pre_engram,"pre")
        show_tmat(recall_memory,"memory")
        show_tmat(recall_now_engram,"now")
        show_tmat(label_engram,"label")
        cv2.waitKey()


        # recall_pre_engram=recall_now_engram
        recall_pre_engram=label_engram
        loss=torch.sum(torch.abs(recall_now_engram-label_engram))
        print(i,"loss=",loss)



def fast_memory(hid_code_list,label_list):
    print(np.shape(hid_code_list),np.shape(label_list))
    seq_engram = torch.from_numpy(hid_code_list.astype(np.float32)).cuda()
    # memory_mat=np.zeros([1000,1000])
    count_mat = torch.zeros([1000, 10]).cuda()
    thred = 0.9
    "memory"
    # seq_engram_binary=np.where(seq_engram>thred,1.0,0.0)
    seq_engram_binary = torch.where(seq_engram > thred, torch.ones_like(seq_engram), torch.zeros_like(seq_engram))
    # print(count_mat)
    run_num = 50
    for i in range(run_num):
        engram=seq_engram_binary[i]
        label=torch.zeros([1,10]).cuda()
        label[0,label_list[i]]=1
        # print(label)

        count_mat+=torch.matmul(torch.reshape(engram,[1000,1]),label)
    # print(torch.max(count_mat))
    # print(count_mat)
    # show_tmat(count_mat,"count_mat",shape=[10,1000])
    # cv2.waitKey()

    norm_count_mat=count_mat/torch.sum(count_mat+1e-8,dim=-1,keepdim=True)
    # norm_count_mat=count_mat
    # print(norm_count_mat[:10])

    # remember
    right_count=0
    count=0
    for i in range(run_num):
        engram = seq_engram_binary[i]
        label=label_list[i]
        remem=torch.matmul(torch.reshape(engram,[1,-1]),norm_count_mat)
        # print(remem)
        max_class=torch.argmax(remem,dim=-1)
        remem_class=max_class.data.cpu().numpy()[0]

        right=remem_class==label
        right_count+=right
        # print(remem_class==label)
        count+=1
        print(remem_class,label)
    print("acc=",right_count/count)




        # torch.matmul()


def sequence_learn():
    save_dir = "./model"
    load_dir=save_dir
    batch_size=128
    hid_dim=128
    "mnist autoencoder"
    mnist_encoder_fc_num_list=[784,256,256,256,hid_dim]
    mnist_decoder_fc_num_list=[hid_dim,256,256,256,784]

    mnist_encoder=to_cuda(dense_block(mnist_encoder_fc_num_list,act_output=nn.Sigmoid()),use_cuda)
    mnist_decoder=to_cuda(dense_block(mnist_decoder_fc_num_list,act_output=nn.Sigmoid()),use_cuda)


    "hippocampus"
    hid_neuron_num=1000
    encoder_fc_num_list=[hid_dim,64,64,256,256,256,256,hid_neuron_num]
    decoder_fc_num_list=[hid_neuron_num,hid_dim]
    # decoder_fc_num_list=[hid_neuron_num,2]
    hippocampus = hippocampus_sparse_autoencoder(
        2, encoder_fc_num_list, decoder_fc_num_list,
        sparse_dim=hid_neuron_num)


    "load_model"
    load_num=500000
    init_iter=0
    if load_num!=None:
        load_model(load_num, mnist_encoder, load_dir, "mnist_encoder")
        load_model(load_num, mnist_decoder, load_dir, "mnist_decoder")
        load_model(load_num, hippocampus.encoder, load_dir, "hippo_encoder")
        # print(hippocampus.decoder.state_dict())
        load_model(load_num, hippocampus.decoder, load_dir, "hippo_decoder")
        init_iter=load_num+1




    "env"
    img_drawer = show_img_class(use_plt=False, show=True)
    data_reader=mnist_reader()


    test_grid_size=22
    test_grid=np.meshgrid(np.linspace(0,1,test_grid_size),np.linspace(0,1,test_grid_size))
    test_grid=np.transpose(test_grid,[1,2,0])
    test_grid=np.reshape(test_grid,[-1,2]).astype(np.float32)


    test_batch_data1,_=data_reader.get_batch_data_label(1)



    "remember_sequence"

    hid_code_list=[]
    label_list=[]
    for iter in range(1):
        # optimizer.zero_grad()
        loss_list=[]
        data_batch,label_batch=data_reader.get_batch_data_label(batch_size)
        mnist_ae_hid_code=mnist_encoder(data_batch)

        "hippocampus"
        hippocampus_restore,loss_top_k,loss_restore,loss_KL,hid_code=hippocampus.pred_with_loss(mnist_ae_hid_code,get_engram=True)
        hid_code_list.append(hid_code.data.cpu().numpy())
        label_list.append(label_batch.data.cpu().numpy())
        # print(hid_code.shape)
    hid_code_list=np.concatenate(hid_code_list,axis=0)
    label_list=np.concatenate(label_list,axis=0)


    # print(np.sum(np.where(hid_code_list>0.95,1,0))/np.prod(np.shape(hid_code_list)))

    # print(hid_code_list.shape)
    # memory_sequence(hid_code_list)

    fast_memory(hid_code_list,label_list)
    # input()




if __name__=="__main__":
    sequence_learn()