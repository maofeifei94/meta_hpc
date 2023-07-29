

import numpy as np
import cv2
import time
from multiprocessing import Process,Queue,Pipe
import threading
class render():
    def __init__(self) -> None:
        pass
    def reset(self):
        pass
    def step(self,ball_index,action,pred_prob):
        pass
def render_pred(label,pred):
    ratio=50
    cv2.imshow("label",cv2.resize(label,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA))
    cv2.imshow("pred",cv2.resize(pred,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA))
    cv2.waitKey(100)

def env_process(send_pipe:Pipe,index):
    from env.minigrid import MiniGrid_fourroom

    env=MiniGrid_fourroom()
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

        # img_feature=vae_net.pred(s_history)

        send_data={
            "action_history":np.array(a_history),
            "s_history":s_history,
        }
        # print("env generate cost",time.time()-env_time)
        t1=time.time()
        send_pipe.send(send_data)


def agent_process(recv_pipe_list:Pipe):
    from mlutils.ml import moving_avg
    from pfc.gru_pred_conv import GruPredHid 
    import paddle
    from vae.minigrid_vae import MinigridVae

    vae_net=MinigridVae()
    vae_net.load_model("all_models/minigrid_vae_model",36000)

    prednet=GruPredHid()
    avg_loss=moving_avg(0.99)

    train_per_history=10
    history_pool_maxsize=200

    # global data_list
    global lock
    # global state
    global data_list
    # global agent_pipe_send
    lock=threading.Lock()
    data_list=[]
    # state="collect" #collect or use

    # agent_pipe_send,agent_pipe_recv=Pipe()


    def recv_pipe_thread(env_pipe):
        global data_list
        global lock
        # global agent_pipe

        while 1:
            data=env_pipe.recv()
            data["obs_feature_history"]=vae_net.pred(data["s_history"])
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

    def train_thread():
        global lock
        global data_list

        for i in range(1000000000000):
            t1=time.time()
            while 1:
                lock.acquire()
                if len(data_list)<train_per_history:
                    lock.release()
                    time.sleep(0.1)
                    continue
                else:
                    sub_data_list=data_list[:train_per_history]
                    data_list=data_list[train_per_history:]
                    lock.release()
                    break
            for _d,data in enumerate(sub_data_list):
                prednet.rpm_collect(data)
            # print("collect cost",time.time()-t1)
            t_learn=time.time()
            loss_list=prednet.learn()
            if loss_list is None:
                continue
            avg_loss.update(loss_list)
            # print("learn cost",time.time()-t_learn)
        
            if i%100==0:
                print(i,avg_loss,prednet.optimizer.get_lr())
            if i%200==0:
                # print(np.argmax(a_history,axis=-1),np.array(s_history)[:,1].astype(np.int))
                # render_pred(prednet.label,prednet.pred_result)
                pass
            if i%2000==0:
                prednet.save_model("all_models/minigrid_model",i)
    recv_thread_list=[threading.Thread(target=recv_pipe_thread,args=(recv_pipe,)) for recv_pipe in recv_pipe_list]
    thread2=threading.Thread(target=train_thread)

    [td.start() for td in recv_thread_list]
    thread2.start()

    [td.join() for td in recv_thread_list]
    thread2.join()
def train_multiprocess():
    process_num=8
    pipe_list=[Pipe() for _p in range(process_num)]
    # data_queue=Pipe()
    p_list=[]
    for i in range(process_num):
        p_list.append(Process(target=env_process,args=(pipe_list[i][0],i)))
    p_list.append(Process(target=agent_process,args=([pipe[1] for pipe in pipe_list],)))
    [p.start() for p in p_list]
    [p.join() for p in p_list]



if __name__=="__main__":
    # train_pred()
    train_multiprocess()
    # collect_data_multiprocess()
