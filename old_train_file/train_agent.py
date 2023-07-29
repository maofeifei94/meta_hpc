
import numpy as np
import cv2
import paddle
from env.env import multi_circle_env
from hpc import V_Auto_Encoder,Place_imging_module,Reward_HPC_Module

def collect_env_img_reward(_env:multi_circle_env,data_per_goal=10):
    grid_num=20.0
    block_size=1/grid_num

    obs=_env.reset()

    img_list=[]
    place_list=[]
    reward_list=[]

    for i in range(1,19):
        print('collect_env_img_reward',i)
        for j in range(1,19):
            for n in range(data_per_goal):
                y=np.random.uniform(i*block_size,(i+1)*block_size)
                x=np.random.uniform(j*block_size,(j+1)*block_size)


                target_site=[y,x]
                obs,reward,done=_env.step(None,place=target_site,hpc_train_mode=True)

                
                ball_img=obs["ball_obs"]
                ball_place=obs['ball_site_norm']

                img_list.append(ball_img)
                place_list.append(ball_place)
                reward_list.append(reward)

                # print(ball_place,reward)
                # cv2.imshow("ball_img",cv2.resize(ball_img,dsize=None,fx=4,fy=4,interpolation=cv2.INTER_AREA))
                # cv2.waitKey()
    return img_list,place_list,reward_list

def train_vae(env_data,vae:V_Auto_Encoder,train_vae_steps):
    img_list,_,_=env_data
    print_loss_iter=1000
    save_model_iter=10000
    "train vae"
    for i in range(len(img_list)):
        vae.rpm_collect(img_list[i])
    print(f"vae rpm size={len(vae.rpm)}")
    for i in range(train_vae_steps):
        img,recon_img=vae.learn()
        # print(img.shape)
        if i %print_loss_iter==0:
            show_img=np.concatenate([np.transpose(img[0],[1,2,0]),np.transpose(recon_img[0],[1,2,0])],axis=1)
            show_img=cv2.resize(show_img,dsize=None,fx=5,fy=5,interpolation=cv2.INTER_AREA)
            print('iter',i," vae_loss=",vae.avg_loss_recon,vae.avg_loss_kl)
            # print(np.shape(show_img),np.max(show_img))
            # cv2.imshow("img",show_img)
            # cv2.waitKey(10)
            cv2.imwrite(f"img/vae_recon_img{i}.png",np.array(show_img*255,dtype=np.uint8))
        if (i+1)%save_model_iter==0:
            vae.save_model("vae_model","newest")
def train_rhm(p_num,env_data,vae:V_Auto_Encoder,rhm:Reward_HPC_Module,train_rhm_steps):
    img_list,place_list,reward_list=env_data
    print_loss_iter=1000
    save_model_iter=10000

    "collect"
    for i in range(len(img_list)):
        img=img_list[i]
        place=place_list[i]
        reward=reward_list[i]

        img_paddle=paddle.to_tensor(np.expand_dims(np.transpose(img,[2,0,1]),axis=0),'float32')

        img_hid_vec2=vae.input(img_paddle).numpy()[0]
        rhm.rpm_collect(np.concatenate([place,img_hid_vec2]),reward)

    
    "train_Reward_HPC_Module"
    rhm.count_rpm_label()
    for i in range(train_rhm_steps):
        rhm.learn()
        if i %print_loss_iter==0:
            print(f"process {p_num} iter {i} reward_hpc_loss={rhm.avg_loss}")
        if (i+1)%save_model_iter==0:
            rhm.save_model(f"rhm_rand_goal_model/{p_num}","newest")

def train_hpc(p_num,env_data,vae:V_Auto_Encoder,pim:Place_imging_module,rhm:Reward_HPC_Module,train_pim_steps,train_rhm_steps):
    img_list,place_list,reward_list=env_data
    print_loss_iter=1000
    save_model_iter=10000

    "collect"
    for i in range(len(img_list)):
        img=img_list[i]
        place=place_list[i]
        reward=reward_list[i]

        img_paddle=paddle.to_tensor(np.expand_dims(np.transpose(img,[2,0,1]),axis=0),'float32')
        img_hid_vec=vae.input(img_paddle,pred_mean=True).numpy()[0]
        pim.rpm_collect(place,img_hid_vec)

        img_hid_vec2=vae.input(img_paddle).numpy()[0]
        rhm.rpm_collect(np.concatenate([place,img_hid_vec2]),reward)

    "train_imaging_net"
    for i in range(train_pim_steps):
        pim.learn()
        if i %print_loss_iter==0:
            print(f"process {p_num} iter {i} imaging_loss={pim.avg_loss}")
        if (i+1)%save_model_iter==0:
            pim.save_model(f"pim_model/{p_num}","newest")
    
    "train_Reward_HPC_Module"
    rhm.count_rpm_label()
    for i in range(train_rhm_steps):
        rhm.learn()
        if i %print_loss_iter==0:
            print(f"process {p_num} iter {i} reward_hpc_loss={rhm.avg_loss}")
        if (i+1)%save_model_iter==0:
            rhm.save_model(f"rhm_model/{p_num}","newest")

def test_hpc_and_vae(env_data,vae:V_Auto_Encoder,pim:Place_imging_module,rhm:Reward_HPC_Module):
    img_list,place_list,reward_list=env_data
    # print(place_list)

    for i in range(100):
        rand_index=np.random.randint(0,len(img_list))
        img,place,reward=img_list[rand_index],place_list[rand_index],reward_list[rand_index]

        place_paddle=paddle.to_tensor(np.reshape(place,[1,2]),'float32')
        img_paddle=paddle.to_tensor(np.expand_dims(np.transpose(img,[2,0,1]),axis=0))

        imaging_hid_vec=pim.input(place_paddle)
        real_hid_vec=vae.input(img_paddle,pred_mean=True)

        pred_reward=rhm.input(paddle.concat([place_paddle,imaging_hid_vec],axis=-1))

        decode_img=vae.decode(imaging_hid_vec)

        show_img=np.concatenate([img,np.transpose(decode_img[0],[1,2,0])],axis=1)
        show_img=cv2.resize(show_img,dsize=None,fx=5,fy=5,interpolation=cv2.INTER_AREA)

        # print(paddle.mean((imaging_hid_vec-real_hid_vec)**2))
        # print("loss=",)
        print("pred_reward=",pred_reward.numpy()[0][0],reward)
        cv2.imshow("show_img",show_img)
        cv2.waitKey()


def maze_rand_goal():
    _env=multi_circle_env()
    _env.load_maze_param(f"maze_param/maze_{0}.npy")

    for i in range(100):
        _env.rand_real_goal()
        _env.reset()
        _env.save_maze_param(f"0_randgoal_{i}","maze_rand_goal_param")

def test_maze_rand_goal():
    _env=multi_circle_env()
    for i in range(100):
        _env.load_maze_param(f"maze_rand_goal_param/maze_0_randgoal_{i}.npy")
        _env.reset()
        print(_env.goal_site_list)
        print(_env.real_goal_index_list)
        # print(_env.real_goal_index_list)
        print("******************")
    

if __name__=="__main__":
    # maze_rand_goal()
    test_maze_rand_goal()

#     # for i in range(5):
#     #     _env=multi_circle_env()
#     #     _env.save_maze_param(i)

#     env_list=[]
#     for i in range(3):
#         _env=multi_circle_env()
#         _env.load_maze_param(f"maze_param/maze_{i}.npy")
    


#     vae=V_Auto_Encoder()
#     vae.load_model("vae_model",'newest')

#     pim=Place_imging_module(2)
#     pim.load_model("pim_model",'newest')
    
#     rhm=Reward_HPC_Module(2+256)
#     rhm.load_model("rhm_model",'newest')

#     _env.reset()
#     _env_data=collect_env_img_reward(_env)

#     train_hpc_and_vae(_env_data,vae,pim,rhm)

    # test_hpc_and_vae(_env_data,vae,pim,rhm)


    



