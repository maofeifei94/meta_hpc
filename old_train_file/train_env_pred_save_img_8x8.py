

import numpy as np
import cv2
import time
from multiprocessing import Process,Queue,Pipe
import threading
import os
import zipfile
from mlutils.data import npy_compress

save_dir="all_data/minigrid_img8x8_history"

def env_process(send_pipe:Pipe,index):
    from env.minigrid import MiniGrid_fourroom_obs8x8

    env=MiniGrid_fourroom_obs8x8()
    while 1:
        obs=env.reset()
        done=False
        a_history=[]
        s_history=[]

        env_time=time.time()
        while not done:
            action=np.clip(np.random.randint(0,4),0,2)

            ball_obs=obs
            s_history.append(np.transpose(ball_obs,[2,0,1]))
            a_history.append([action])

            obs,reward,done=env.step(action)

            # print(np.shape(s_history))

        # img_feature=vae_net.pred(s_history)

        send_data={
            "action_history":np.array(a_history),
            "s_history":s_history,
        }
        # print("env generate cost",time.time()-env_time)
        t1=time.time()
        send_pipe.send(send_data)
def save_data_process(recv_pipe_list:Pipe,compress_pipe_list:Pipe):
    from vae.minigrid_vae import MinigridVae
    from mlutils.ml import one_hot
    import os
    save_per_history=1000
    history_pool_maxsize=2000


    # vae_net=MinigridVae()
    # vae_net.load_model("all_models/minigrid_vae_model",36000)
    global lock
    # global state
    global data_list
    lock=threading.Lock()
    data_list=[]
    def recv_pipe_thread(env_pipe):
        global data_list
        global lock
        # global agent_pipe

        while 1:
            data=env_pipe.recv()
            # data["obs_feature_history"]=vae_net.pred(data["s_history"])
            while 1:
                lock.acquire()
                if len(data_list)>history_pool_maxsize:
                    lock.release()
                    time.sleep(0.1)
                else:
                    lock.release()
                    break
            
            lock.acquire()
            # agent_pipe_send.send(data)
            data_list.append(data)
            lock.release()
    def save_data_thread():
        global lock
        global data_list
        for i in range(0,1000):
            while 1:
                lock.acquire()
                if len(data_list)<save_per_history:
                    lock.release()
                    time.sleep(0.1)
                    continue
                else:
                    sub_data_list=data_list[:save_per_history]
                    print("data size=",len(data_list))
                    data_list=data_list[save_per_history:]
                    lock.release()
                    break
            



            s_list=[_d['s_history'] for _d in sub_data_list]
            action_list=[_d['action_history'] for _d in sub_data_list]

            one_hot_action=np.squeeze(one_hot(action_list,3))
            print(np.shape(s_list),np.shape(action_list),np.shape(one_hot_action))

            save_data={
                "s_history":np.array(s_list),
                "action_history":np.array(one_hot_action)
            }
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            np.save(f"{save_dir}/{i}.npy",save_data)
            compress_pipe_index=i%len(compress_pipe_list)
            compress_pipe_list[compress_pipe_index].send(i)

            
    recv_thread_list=[threading.Thread(target=recv_pipe_thread,args=(recv_pipe,)) for recv_pipe in recv_pipe_list]
    thread2=threading.Thread(target=save_data_thread)

    [td.start() for td in recv_thread_list]
    thread2.start()

    [td.join() for td in recv_thread_list]
    thread2.join()



def compress_process(pipe):
    while 1:
        data_num=pipe.recv()
        npy_compress().compress(save_dir,f"{data_num}.npy")
        npy_compress().delete_f(f"{save_dir}/{data_num}.npy")
        print(f"compress data {data_num} success")



def collect_data_multiprocess():
    process_num=8
    compress_num=8
    pipe_list=[Pipe() for _p in range(process_num)]
    compress_pipe_list=[Pipe() for _p in range(compress_num)]
    # data_queue=Pipe()
    p_list=[]
    for i in range(process_num):
        p_list.append(Process(target=env_process,args=(pipe_list[i][0],i)))
    p_list.append(Process(target=save_data_process,args=([pipe[1] for pipe in pipe_list],[c_pipe[0] for c_pipe in compress_pipe_list])))

    for i in range(compress_num):
        p_list.append(Process(target=compress_process,args=(compress_pipe_list[i][1],)))
    [p.start() for p in p_list]
    [p.join() for p in p_list]

if __name__=="__main__":
    collect_data_multiprocess()


