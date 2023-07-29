
import os
import gym
import math
import random
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
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
    print("save success",count)
    torch.save(net.state_dict(),dir+"/"+name+"_{}".format(count))
def load_model(count,net,dir,name):
    net.load_state_dict(torch.load(dir+"/"+name+"_{}".format(count)))#, map_location=torch.device('cpu')))

def train_place_cell():
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
    load_num=None
    init_iter=50000
    if load_num!=None:
        load_model(load_num, mnist_encoder, load_dir, "mnist_encoder")
        load_model(load_num, mnist_decoder, load_dir, "mnist_decoder")
        load_model(load_num, hippocampus.encoder, load_dir, "hippo_encoder")
        load_model(load_num, hippocampus.decoder, load_dir, "hippo_decoder")
        init_iter=load_num+1




    "env"
    img_drawer = show_img_class(use_plt=False, show=True)
    data_reader=mnist_reader()

    "train"
    optimizer=optim.RMSprop(params=[*list(mnist_encoder.parameters()),
                                    *list(mnist_decoder.parameters()),
                                    *list(hippocampus.encoder.parameters()),
                                    *list(hippocampus.decoder.parameters())],
                            lr=0.0001)


    show_iter=1000
    save_iter=10000

    test_grid_size=21
    test_grid=np.meshgrid(np.linspace(0,1,test_grid_size),np.linspace(0,1,test_grid_size))
    test_grid=np.transpose(test_grid,[1,2,0])
    test_grid=np.reshape(test_grid,[-1,2]).astype(np.float32)
    avg_loss_list=None
    loss_decay=0.999

    test_batch_data1,_=data_reader.get_batch_data_label(1)

    for iter in range(init_iter,1000000000000000):
        optimizer.zero_grad()
        loss_list=[]
        data_batch,label_batch=data_reader.get_batch_data_label(batch_size)


        mnist_ae_hid_code=mnist_encoder(data_batch)

        "hippocampus"
        hippocampus_restore,loss_top_k,loss_restore,loss_KL=hippocampus.pred_with_loss(mnist_ae_hid_code)
        avg_hid=(hippocampus_restore+mnist_ae_hid_code)/2
        loss_list=[loss_top_k,loss_restore,loss_KL]

        # avg_hid=mnist_ae_hid_code

        mnist_restore=mnist_decoder(avg_hid)
        loss_mnist_ae=torch.mean((mnist_restore-data_batch)**2)
        loss_mnist_ae*=100.0

        loss_list.append(loss_mnist_ae)

        loss=0
        for l in loss_list:
            loss+=l

        loss.backward()
        optimizer.step()

        loss_list=np.array([abs(float(l.data.cpu().numpy())) for l in loss_list])

        isNone=type(avg_loss_list)==type(None)
        avg_loss_list=loss_list if isNone else avg_loss_list*loss_decay+(1-loss_decay)*loss_list

        if iter%show_iter==0:
            print(iter,[round(agl,3) for agl in avg_loss_list])

            if show_img_window:

                img_drawer.show_mnist_img(to_np(data_batch),name="data",ratio=10)
                img_drawer.show_mnist_img(to_np(mnist_restore), name="mnist_restore")
                cv2.waitKey()

                with torch.no_grad():
                    test_batch=torch.cat([test_batch_data1,data_batch[:1]],dim=0)
                    hid_code=mnist_encoder(test_batch)

        if iter%save_iter==0:

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_model(iter,mnist_encoder,save_dir,"mnist_encoder")
            save_model(iter, mnist_decoder, save_dir, "mnist_decoder")
            save_model(iter, hippocampus.encoder, save_dir, "hippo_encoder")
            save_model(iter, hippocampus.decoder, save_dir, "hippo_decoder")



        # if count%1==0:
        #     loss_list,_=hippocampus.train_one_step()
        #     if type(avg_loss_list)==type(None):
        #         avg_loss_list=np.array(loss_list)
        #     else:
        #         avg_loss_list=avg_loss_list*0.999+0.001*np.array(loss_list)
        # if count%save_iter==0:
        #     hippocampus.save_model(count,"model")
        # if count%show_iter==0:
        #     print(count,avg_loss_list)
        #
        #     test_data=[0.5,0.5]
        #     test_data2=np.random.uniform(0,1,[2])
        #
        #     "hid code"
        #     hid_code=hippocampus.get_hid_code(test_data)
        #     # print("here")
        #     hid_hist_img=img_drawer.show_hist(hid_code,"test_hid")
        #
        #
        #     hid_code2=hippocampus.get_hid_code(test_data2)
        #
        #     hippocampus.show_hid(hid_code,"hid_code")
        #     hippocampus.show_hid(hid_code2,"hid_code2")
        #     "get place field"
        #     grid_hid_code=hippocampus.get_hid_code(test_grid)
        #     # print(np.shape(grid_hid_code))
        #     rand_cell=np.random.randint(0,hid_neuron_num)
        #     cell_place_field=grid_hid_code[:,rand_cell:rand_cell+1]
        #     cell_place_img=np.reshape(cell_place_field,[test_grid_size,test_grid_size])
        #     # cv2.putText(cell_place_img,str(rand_cell),)
        #     # print(hid_code[0,0])
        #     # print(np.max(cell_place_img))
        #     cv2.imshow("cell_place_img",cv2.resize(cell_place_img,(300,300),interpolation=cv2.INTER_AREA))
        #     "avg_fire"
        #     avg_fire=hippocampus.get_avg_fire_numpy()
        #     avg_fire_img=img_drawer.show_hist(avg_fire,"avg_fire")
        #
        #
        #     img_count+=1
        #
        # count+=1

if __name__=="__main__":
    train_place_cell()