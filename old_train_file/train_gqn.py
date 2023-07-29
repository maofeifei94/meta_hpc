from gqn.model import gqnmodel 
from env.env_war_frog_gqn import multi_circle_env
import numpy as np
import cv2
import random
import paddle
from myf_ML_util import moving_avg

def train():
    img_num=20
    history_num_range=[3,12]

    env=multi_circle_env()
    model=gqnmodel()
    model_num=220000
    if model_num==0 or model_num is None:
        pass
    else:
        model.load_model("all_models/gqn_model",model_num)

    avg_loss=moving_avg()

    for iter in range(model_num,999999999):
        "get data"
        env.reset()
        img_list=[]
        view_list=[]
        for i in range(img_num):
            rand_site=np.random.uniform(0,1,[2])
            env_info,reward,done=env.step(None,rand_site,True)
            # print(env_info.keys())
            img=env_info['ask_ball_obs']
            view=env_info['ask_site_norm']
            # print(view,np.max(img),np.shape(img))
            # cv2.imshow("img",img)
            # cv2.waitKey(10)
            img_list.append(img)
            view_list.append(view)

        "get train data"
        history_len=np.random.randint(*history_num_range)
        index_list=list(range(img_num))
        random.shuffle(index_list)
        history_img=[img_list[index] for index in index_list[:history_len]]
        history_view=[view_list[index] for index in index_list[:history_len]]

        target_img=img_list#[img_list[index] for index in index_list[history_len:]]
        target_view=view_list#[view_list[index] for index in index_list[history_len:]]

        "to_paddle"
        history_img=paddle.transpose(paddle.to_tensor(np.array(history_img)),[0,3,1,2])#[b,3,64,64]
        target_img=paddle.transpose(paddle.to_tensor(np.array(target_img)),[0,3,1,2]) #[b,3,64,64]
        history_view=paddle.tile(
            paddle.reshape(
                paddle.to_tensor(
                    np.array(history_view,dtype=np.float32)
                    ),
                [len(history_view),2,1,1]
                ),
                [1,1,16,16]
            ) #[b,2,16,16]
        target_view=paddle.tile(
            paddle.reshape(
                paddle.to_tensor(
                    np.array(target_view,dtype=np.float32)
                    ),
                    [len(target_view),2,1,1]
                    ),
                [1,1,16,16]
            ) #[b,2,16,16]

        train_data={
            "history_img_batch":history_img,
            "history_view_batch":history_view,
            "target_img":target_img,
            "target_view":target_view,
        }
        # print("history_viewshape=",history_view.shape)
        # print("target_img.shape",target_img.shape)

        loss_list=model.train_one_step(train_data,iter)
        avg_loss.update(loss_list)

        if iter%100==0:
            print(iter,avg_loss)
        if iter%10000==0:
            model.save_model("all_models/gqn_model",iter)
        # print()

    # pass

if __name__=="__main__":
    train()