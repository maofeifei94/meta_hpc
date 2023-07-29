import multiprocessing

# from itsdangerous import NoneAlgorithm

# from numpy import block
# from train_agent import train_rhm

def train_randgoal_rhm_process(p_num,total_p_num,total_env_num):
    import numpy as np
    import cv2
    import paddle
    from env.env import multi_circle_env
    from hpc import V_Auto_Encoder,Place_imging_module,Reward_HPC_Module
    from train_agent import collect_env_img_reward,train_hpc,train_rhm

    for i in range(total_env_num):
        if i%total_p_num==p_num:
            _env=multi_circle_env()
            _env.load_maze_param(f"maze_rand_goal_param/maze_0_randgoal_{p_num}.npy")
            _env.reset()

            vae=V_Auto_Encoder()
            vae.load_model(f"vae_model",'newest')

            # pim=Place_imging_module(2)
            # pim.load_model(f"pim_model/{p_num}",'newest')
            
            rhm=Reward_HPC_Module(2+256)
            # rhm.load_model(f"rhm_model/{p_num}",'newest')

            _env_data=collect_env_img_reward(_env,data_per_goal=10)

            train_rhm(i,_env_data,vae,rhm,100000)
def train_randgoal_rhm():
    total_p_num=5
    for i in range(total_p_num):
        multiprocessing.Process(target=train_randgoal_rhm_process,args=(i,total_p_num,100)).start()

def train_hpc_process(p_num):
    import numpy as np
    import cv2
    import paddle
    from env.env import multi_circle_env
    from hpc import V_Auto_Encoder,Place_imging_module,Reward_HPC_Module
    from train_agent import collect_env_img_reward,train_hpc

    _env=multi_circle_env()
    _env.load_maze_param(f"maze_param/maze_{p_num}.npy")
    _env.reset()

    vae=V_Auto_Encoder()
    vae.load_model(f"vae_model",'newest')

    pim=Place_imging_module(2)
    pim.load_model(f"pim_model/{p_num}",'newest')
    
    rhm=Reward_HPC_Module(2+256)
    rhm.load_model(f"rhm_model/{p_num}",'newest')

    
    _env_data=collect_env_img_reward(_env,data_per_goal=10)

    train_hpc(p_num,_env_data,vae,pim,rhm,train_pim_steps=200000,train_rhm_steps=200000)
    # train_hpc(p_num,_env_data,vae,pim,rhm,train_pim_steps=1000,train_rhm_steps=1000)
def train_vae_process():
    import numpy as np
    import cv2
    import paddle
    from env.env import multi_circle_env
    from hpc import V_Auto_Encoder,Place_imging_module,Reward_HPC_Module
    from train_agent import collect_env_img_reward,train_vae

    all_img_list=[]
    for i in range(3):
        _env=multi_circle_env()
        _env.load_maze_param(f"maze_param/maze_{i}.npy")
        _env.reset()

        img_list,place_list,reward_list=collect_env_img_reward(_env,data_per_goal=10)
        all_img_list=[*all_img_list,*img_list]

    vae=V_Auto_Encoder()
    vae.load_model(f"vae_model",'newest')
    
    train_vae([all_img_list,None,None],vae,train_vae_steps=200000)

def train_slpfc():
    import paddle
    import random
    import time
    import cv2
    import numpy as np
    from SL_pfc import slpfc,fakepfc
    from env.env import multi_circle_env
    from hpc import V_Auto_Encoder,Place_imging_module,Reward_HPC_Module
    from myf_ML_util import timer_tool

    tt=timer_tool("slpfc")


    pfc=slpfc()
    pfc.load_model("pfc_model")
    # pfc=fakepfc()

    vae=V_Auto_Encoder()
    vae.load_model(f"vae_model",'newest')

    pim=Place_imging_module(2)
    pim.load_model(f"pim_model/{0}",'newest')

    mazes_num=6

    rhm_list=[Reward_HPC_Module(2+256) for _r in range(mazes_num)]
    for _n in range(mazes_num):
        rhm_list[_n].load_model(f"rhm_rand_goal_model/{_n}",'newest')

    test_img=np.zeros([600,600,3],dtype=np.uint8)

    for train_step in range(10000):
        tt.start()
        "随机选择环境"
        _env=multi_circle_env()
        maze_rand_num=np.random.randint(0,mazes_num)
        # maze_rand_num=0
        print("rand maze=",maze_rand_num)
        _env.load_maze_param(f"maze_rand_goal_param/maze_0_randgoal_{maze_rand_num}.npy")
        _env.reset()
        tt.end_and_start("env")

        "选择对应的网络"
        rhm=rhm_list[maze_rand_num]
        # pim=Place_imging_module(2)
        # pim.load_model(f"pim_model/{maze_rand_num}",'newest')
        # rhm=Reward_HPC_Module(2+256)
        # rhm.load_model(f"rhm_model/{maze_rand_num}",'newest')

        # print(_env.goal_site_list,_env.real_goal_index_list)

        "env所有格子分成三类：real_goal,fake_goal,no_goal"
        all_goal_site_list=_env.goal_site_list
        real_goal_site_list=[_env.goal_site_list[rgi] for rgi in _env.real_goal_index_list]
        fake_goal_site_list=[]
        no_goal_site_list=[]
        for _x in range(1,19):
            for _y in range(1,19):
                site=[_x,_y]
                if site in all_goal_site_list:
                    if site in real_goal_site_list:
                        pass
                    else:
                        fake_goal_site_list.append(site)
                else:
                    no_goal_site_list.append(site)
        
        tt.end_and_start("block seperate")
        
        "从三类格子中均匀随机抽样"
        check_site_list=[]
        for i in range(64):
            real_goal_site=random.choice(real_goal_site_list)
            fake_goal_site=random.choice(fake_goal_site_list)
            no_goal_site=random.choice(no_goal_site_list)
            check_site_list.append(real_goal_site)
            check_site_list.append(fake_goal_site)
            check_site_list.append(no_goal_site)
        
        tt.end_and_start("choose block")

        "从每个抽中的格子，提取一个左边点"
        train_img_list=[]
        train_place_list=[]
        train_reward_list=[]
        for cs in check_site_list:
            x,y=cs
            gap=0.25
            block_size=0.05
            norm_y=np.random.uniform((y+gap)*block_size,(y+1-gap)*block_size)
            norm_x=np.random.uniform((x+gap)*block_size,(x+1-gap)*block_size)

            target_site=[norm_y,norm_x]
            obs,reward,done=_env.step(None,place=target_site,hpc_train_mode=True)
            ball_img=obs["ball_obs"]
            ball_place=obs['ball_site_norm']

            train_img_list.append(ball_img)
            train_place_list.append(ball_place)
            train_reward_list.append([reward])

            if reward==-1:
                color=(255,0,0)
            elif reward==0:
                color=(0,255,0)
            elif reward==1:
                color=(0,0,255)
            else:
                color=(255,255,255)
            
            
            # cv2.imshow("ball_img",ball_img)
            # print(reward)
            # cv2.waitKey()
        #     cv2.circle(test_img,(int(ball_place[1]*600),int(ball_place[0]*600)),3,color,-1)
        # cv2.imshow("test_img",test_img)
        # cv2.waitKey()
        
        tt.end_and_start("choose point")

        img_batch_paddle=paddle.to_tensor(np.transpose(np.array(train_img_list,dtype=np.float32),[0,3,1,2]),'float32')
        place_batch_paddle=paddle.to_tensor(np.array(train_place_list,dtype=np.float32),'float32')
        vae_vec_paddle=vae.input(img_batch_paddle,pred_mean=True)
        train_data=paddle.concat([vae_vec_paddle,place_batch_paddle],axis=-1)

        train_label=paddle.to_tensor(np.array(train_reward_list,dtype=np.float32),'float32')
        # print(place_batch_paddle,train_label)
        # t1=time.time()
        loss=pfc.train(train_data,train_label,vae,pim,rhm)
        print(f"train pfc {train_step} loss={pfc.avg_loss}")
        tt.end_and_start("train")

        if train_step%100==0:
            pfc.save_model("pfc_model")

def test_slpfc():
    import paddle
    import random
    import time
    import cv2
    import numpy as np
    from SL_pfc import slpfc,fakepfc
    from env.env import multi_circle_env
    from hpc import V_Auto_Encoder,Place_imging_module,Reward_HPC_Module
    from myf_ML_util import timer_tool

    tt=timer_tool("slpfc")

    pfc=slpfc()
    pfc.load_model("pfc_model")
    # pfc=fakepfc()

    vae=V_Auto_Encoder()
    vae.load_model(f"vae_model",'newest')

    pim=Place_imging_module(2)
    pim.load_model(f"pim_model/{0}",'newest')

    mazes_num=6

    rhm_list=[Reward_HPC_Module(2+256) for _r in range(mazes_num)]
    for _n in range(mazes_num):
        rhm_list[_n].load_model(f"rhm_rand_goal_model/{_n}",'newest')

    test_img=np.zeros([600,600,3],dtype=np.uint8)

    for train_step in range(10000):
        _env=multi_circle_env()
        maze_rand_num=np.random.randint(0,mazes_num)
        # maze_rand_num=0
        print("rand maze=",maze_rand_num)
        _env.load_maze_param(f"maze_rand_goal_param/maze_0_randgoal_{maze_rand_num}.npy")
        _env.reset()
        tt.end_and_start("env")

        "选择对应的网络"

        rhm=rhm_list[maze_rand_num]

        pfc.test(None,None,vae,pim,rhm,_env)




def test():
    import numpy as np
    import cv2
    import paddle
    from env.env import multi_circle_env
    from hpc import V_Auto_Encoder,Place_imging_module,Reward_HPC_Module
    from train_agent import collect_env_img_reward,train_vae,test_hpc_and_vae

    _env=multi_circle_env()
    _env.load_maze_param(f"maze_param/maze_{0}.npy")
    _env.reset()

    vae=V_Auto_Encoder()
    vae.load_model(f"vae_model",'newest')

    pim=Place_imging_module(2)
    pim.load_model(f"pim_model/{0}",'newest')
    
    rhm=Reward_HPC_Module(2+256)
    rhm.load_model(f"rhm_model/{0}",'newest')

    
    _env_data=collect_env_img_reward(_env,data_per_goal=1)
    test_hpc_and_vae(_env_data,vae,pim,rhm)

if __name__=="__main__":
    # train_randgoal_rhm()
    # train_vae_process()

    # for i in range(3):
    #     multiprocessing.Process(target=train_hpc_process,args=(i,)).start()

    # train_slpfc()

    # test()

    test_slpfc()










