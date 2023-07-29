
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

    data_batch, label_batch = data_reader.get_batch_data_label(batch_size)
    mnist_ae_hid_code = mnist_encoder(data_batch)

    "hippocampus"
    hippocampus_restore, loss_top_k, loss_restore, loss_KL, hid_code = hippocampus.pred_with_loss(mnist_ae_hid_code,
                                                                                                  get_engram=True)
    avg_hid = (hippocampus_restore + mnist_ae_hid_code) / 2
    mnist_restore = mnist_decoder(avg_hid)
    print(np.shape(hid_code))

    # input()

    show_code = hid_code.data.cpu().numpy()
    mnist_batch=data_batch.data.cpu().numpy()
    restore_mnist=mnist_restore.data.cpu().numpy()
    for i in range(len(show_code)):
        the_minst_img=np.reshape(np.array(mnist_batch[i]*255,np.uint8),[28,28])
        plt.hist(show_code[i],bins=100)
        print(np.sum(show_code[i]>0.95),np.sum(show_code[i]<0.05))
        plt.show()
        cv2.imshow("data",cv2.resize(the_minst_img,(300,300)))
        cv2.imwrite("engram_code/{}.png".format(i),the_minst_img)

        the_mnist_restore=np.reshape(np.array(restore_mnist[i]*255,np.uint8),[28,28])
        cv2.imshow("resotore",cv2.resize(the_mnist_restore,(300,300)))
        cv2.imwrite(f"engram_code/{i}r.png",the_mnist_restore)
        cv2.waitKey(1)



        plt.axis('off')
        plt.imshow(np.reshape(show_code[i],[25,40]))
        plt.savefig("engram_code/{}.svg".format(i))
        plt.close()
        # plt.show()




if __name__=="__main__":
    sequence_learn()