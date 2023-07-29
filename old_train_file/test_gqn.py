from cProfile import label
from gqn.model import gqnmodel 
from env.env_war_frog_gqn import multi_circle_env
import numpy as np
import cv2
import random
import paddle
import copy
from myf_ML_util import moving_avg,img_cross_area
def restore_full_img(history_img_list,history_view_list,full_img):
    restore_img=np.full([600,600,3],0.3,np.float32)
    for i in range(history_view_list.shape[0]):
        img=cv2.resize(np.transpose(history_img_list[i].numpy(),[1,2,0]),(256,256),interpolation=cv2.INTER_AREA)
        view=np.array(history_view_list[i,:,0,0].numpy()*600,np.int)
        print(view)
        view_minx,view_miny,view_maxx,view_maxy=view[0]-128,view[1]-128,view[0]+128,view[1]+128
        full_img_area,sub_img_area=img_cross_area(600,600,view[0]-128,view[1]-128,view[0]+128,view[1]+128)
        restore_img[full_img_area[0]:full_img_area[1],full_img_area[2]:full_img_area[3]]=img[sub_img_area[0]:sub_img_area[1],sub_img_area[2]:sub_img_area[3]]
        cv2.rectangle(restore_img,(view_minx,view_miny),(view_maxx,view_maxy),(255,0,0),1)
    return restore_img



def test():
    img_num=40
    history_num_range=[5,6]
    env=multi_circle_env()
    model=gqnmodel()
    model.load_model("all_models/gqn_model",500000)

    avg_loss=moving_avg()

    for iter in range(999999999):
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

        "get test data"
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

        test_data={
            "history_img_batch":history_img,
            "history_view_batch":history_view,
            "target_img":target_img,
            "target_view":target_view,
        }
        test_pred_dict=model.pred_for_test(test_data)
        train_pred_dict=model.pred_for_train(test_data)

        full_img=env.render_full_img_cv()
        full_img_restore=restore_full_img(history_img,history_view,None)

        

        # print(train_pred_dict["recon_img"].shape)
        for _i in range(test_data['target_img'].shape[0]):
            label_img=np.transpose(test_data['target_img'][_i].numpy(),[1,2,0])

            recon_img_test=np.transpose(test_pred_dict['recon_img'][_i].numpy(),[1,2,0])
            recon_img_train=np.transpose(train_pred_dict['recon_img'][_i].numpy(),[1,2,0])

            target_view=np.array(test_data['target_view'][_i].numpy()[:,0,0]*600,np.int)

            full_img_show=copy.deepcopy(full_img)
            full_img_restore_show=copy.deepcopy(full_img_restore)
            cv2.rectangle(
                full_img_restore_show,
                (target_view[0]-128,target_view[1]-128),
                (target_view[0]+128,target_view[1]+128),
                (0,0,255),
                2)
            cv2.rectangle(
                full_img_show,
                (target_view[0]-128,target_view[1]-128),
                (target_view[0]+128,target_view[1]+128),
                (0,0,255),
                2)
            cv2.imshow("full",full_img_show)
            cv2.imshow("full restore",full_img_restore_show)

            cv2.imshow("1",cv2.resize(np.concatenate([label_img,recon_img_train,recon_img_test],axis=1),None,fx=5,fy=5))
            cv2.waitKey()
        # print("history_viewshape=",history_view.shape)
        # print("target_img.shape",target_img.shape)

        # loss_list=model.train_one_step(train_data)
        # avg_loss.update(loss_list)

        # if iter%100==0:
        #     print(iter,avg_loss)
        # if iter%1000==0:
        #     model.save_model("all_models/gqn_model",iter)
        # print()


    # pass

if __name__=="__main__":
    test()